import math
import os
from pathlib import Path
from typing import Callable, List, Optional, Set, Union

import h5py
import numpy as np

from ..core.species import Species
from ..core.utils.logger import logger
from ..simulation import Simulation, Simulation3D
from .callback import Callback


class SaveFieldsToHDF5(Callback):
    """Callback to save field data to HDF5 files.

    Creates a new HDF5 file for each save with name pattern:
    
    - `prefix/000100.h5`
    - `prefix/000200.h5`
    - ...

    The data structure in each file:

    - `/ex`, `/ey,` `/ez` (electric fields)
    - `/bx`, `/by`, `/bz` (magnetic fields)
    - `/jx`, `/jy`, `/jz` (currents)
    - `/rho` (charge density)

    Args:
        prefix (str): Prefix for output filenames. For example, if prefix is 'output', the files will be named 'output/t000100.h5', 'output/t000200.h5', etc.
        interval (Union[int, float, Callable], optional): Number of timesteps between saves, or a 
            function(sim) -> bool that determines when to save. Defaults to 100.
        components (Optional[List[str]], optional): List of field components to save. 
            Available: ['ex','ey','ez','bx','by','bz','jx','jy','jz','rho']. 
            If None, saves all components.
        slice (tuple[int | slice, ...] | None, optional): Subset of the domain to save, specified via
            ``np.s_`` indexing (e.g., ``slice=np.s_[:, :, 100]``, ``slice=np.s_[::2, ::2, ::5]``,
            ``slice=np.s_[500:, :, :]``). Accepts any ``np.s_``-style tuple of ints and/or slices.
            If None, saves the full domain. Defaults to None.
    """
    DEFAULT_STAGE = "end"
    def __init__(self, 
                 prefix: Union[str, Path]='', 
                 interval: Union[int, float, Callable] = 100,
                 components: Optional[List[str]] = None,
                 mpi: bool = False,
                 slice: tuple[int | slice, ...] | None = None) -> None:
        self.stage = self.DEFAULT_STAGE
        self.prefix = Path(prefix)
        self.interval = interval
        self.mpi = mpi

        self.prefix.mkdir(parents=True, exist_ok=True)
        
        # Available field components
        self.all_components = {'ex','ey','ez','bx','by','bz','jx','jy','jz','rho'}
        if components is None:
            self.components = list(self.all_components)
        else:
            # Validate field components
            invalid = set(components) - self.all_components
            if invalid:
                raise ValueError(f"Invalid field components: {invalid}")
            self.components = list(components)
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
        self.slice = slice
        self._normalized_slice = None
        
    def _call(self, sim: Union[Simulation, Simulation3D]):
        # Normalize np.s_-style slice input to explicit slice objects
        if self.slice is not None:
            if sim.dimension == 2:
                self._normalized_slice = self._normalize_slice(2, self.slice, (sim.nx, sim.ny))
            elif sim.dimension == 3:
                self._normalized_slice = self._normalize_slice(3, self.slice, (sim.nx, sim.ny, sim.nz))
        else:
            self._normalized_slice = None
        filename = self.prefix / f"{sim.itime:06d}.h5"
        if sim.dimension == 2:
            self._write_2d(sim, filename)
        elif sim.dimension == 3:
            self._write_3d(sim, filename)
        
    @staticmethod
    def _normalize_slice(sim_dim: int, user_slice: tuple[int | slice, ...], dims: tuple) -> tuple[slice, ...] | None:
        """Normalize a user-provided slice specification to full slice objects.

        Converts ints to ``slice(i, i+1, 1)`` and ``slice(None)`` to ``slice(0, dim, 1)``.

        Args:
            sim_dim (int): Number of dimensions (2 or 3).
            user_slice (tuple[int | slice, ...]): User-provided ``np.s_``-style slice tuple,
                e.g. ``np.s_[:, :, 100]`` or ``np.s_[::2, ::2, ::5]``.
            dims (tuple): Size of each dimension, e.g. ``(nx, ny)``.

        Returns:
            Optional[tuple]: Normalized tuple of ``slice`` objects, or ``None`` if
                ``user_slice`` is ``None``.

        Raises:
            ValueError: If the slice specification contains invalid types, out-of-bounds
                indices, or negative steps.
        """
        if user_slice is None:
            return None
        
        # Wrap single slice or int in tuple
        if isinstance(user_slice, (slice, int)):
            user_slice = (user_slice,)
        
        # Reject Ellipsis
        if any(isinstance(s, type(Ellipsis)) for s in user_slice):
            raise ValueError("Ellipsis (...) is not supported in slice specification")
        
        # Reject None/np.newaxis
        if any(s is None for s in user_slice):
            raise ValueError("None/np.newaxis is not supported in slice specification")
        
        # Validate length
        if len(user_slice) != sim_dim:
            raise ValueError(
                f"Slice tuple length {len(user_slice)} does not match "
                f"simulation dimension {sim_dim}"
            )
        
        result = []
        for i, s in enumerate(user_slice):
            dim = dims[i]
            if isinstance(s, int):
                # Normalize negative index
                if s < 0:
                    s = dim + s
                if s < 0 or s >= dim:
                    raise ValueError(
                        f"Index {s} out of bounds for dimension {i} with size {dim}"
                    )
                result.append(slice(s, s + 1, 1))
            elif isinstance(s, slice):
                start, stop, step = s.start, s.stop, s.step
                # Fill defaults
                if start is None:
                    start = 0
                if stop is None:
                    stop = dim
                if step is None:
                    step = 1
                # Validate step
                if step <= 0:
                    raise ValueError(f"Step must be positive, got {step}")
                # Resolve negatives
                if start < 0:
                    start = dim + start
                if stop < 0:
                    stop = dim + stop
                # Clamp to valid range
                start = max(0, min(start, dim))
                stop = max(0, min(stop, dim))
                # Validate at least 1 element
                if start >= stop:
                    raise ValueError(
                        f"Slice {s} has no elements for dimension {i} with size {dim}"
                    )
                result.append(slice(start, stop, step))
            else:
                raise ValueError(
                    f"Invalid slice element type: {type(s).__name__}. "
                    f"Expected int or slice."
                )
        
        return tuple(result)
    
    @staticmethod
    def _compute_patch_slice(axis_dim: int, global_slice: slice,
                             patch_offset: int, patch_size: int) -> Optional[tuple]:
        """Compute the local slice for a single patch along one axis.

        Determines which elements of a patch are selected by a global slice.

        Args:
            axis_dim (int): Total size of this axis (e.g., ``sim.nx``).
            global_slice (slice): Normalized slice for this axis (start, stop, step).
            patch_offset (int): Global offset of this patch (``ipatch * n_per_patch``).
            patch_size (int): Size of this patch (``n_per_patch``).

        Returns:
            Optional[tuple]: ``(local_slice, output_start, num_elements)`` or ``None``
                if the patch does not intersect the slice.
        """
        start, stop, step = global_slice.start, global_slice.stop, global_slice.step
        offset, size = patch_offset, patch_size
        
        first = start + math.ceil(max(0, offset - start) / step) * step
        last_global_in_patch = min(stop, offset + size) - 1
        last_selected = start + math.floor((last_global_in_patch - start) / step) * step
        
        if last_selected < first:
            return None
        
        local_first = first - offset
        output_start = (first - start) // step
        num = (last_selected - first) // step + 1
        local_slice = slice(local_first, local_first + num * step, step)
        
        return (local_slice, output_start, num)
    
    @staticmethod
    def _serialize_slice(normalized_slice: tuple, dims: tuple) -> str:
        """Convert a normalized slice tuple to a human-readable string.

        Args:
            normalized_slice (tuple): Normalized tuple of ``slice`` objects.
            dims (tuple): Full size of each axis in the simulation domain.

        Returns:
            str: Human-readable string such as ``"[:, :, 100]"``.
        """
        parts = []
        for s, dim in zip(normalized_slice, dims):
            if s.start == 0 and s.stop == dim and s.step == 1:
                parts.append(":")
            elif s.step == 1 and s.stop == s.start + 1:
                # Single element like slice(i, i+1, 1) -> "i"
                parts.append(str(s.start))
            else:
                start = "" if s.start == 0 else str(s.start)
                stop = "" if s.stop == dim else str(s.stop)
                if s.step == 1:
                    parts.append(f"{start}:{stop}")
                else:
                    parts.append(f"{start}:{stop}:{s.step}")
        return "[" + ", ".join(parts) + "]"
    
    def _write_2d(self, sim: Simulation, filename: Path):
        """Write 2D field data to HDF5 file in parallel.

        Args:
            sim (Simulation): 2D simulation object containing field data
            filename (str): Output HDF5 filename

        Note:
            - Creates HDF5 file with datasets for each field component
            - Uses MPI parallel I/O to write patch data
            - Includes simulation metadata as file attributes
        """
        # Get MPI communicator
        comm = sim.mpi.comm
        rank = comm.Get_rank()
        
        chunk_size = (sim.nx_per_patch, sim.ny_per_patch)
        # Create filename with timestep
        if self.mpi:
            if self._normalized_slice is not None:
                sx, sy = self._normalized_slice
                shape = (len(range(sx.start, sx.stop, sx.step)), len(range(sy.start, sy.stop, sy.step)))
                with h5py.File(filename, 'w', driver='mpio', comm=comm) as f:
                    for field in self.components:
                        dset = f.create_dataset(
                            field, 
                            data=np.zeros(shape),
                            dtype='f8',
                            chunks=tuple(min(c, s) for c, s in zip(chunk_size, shape))
                        )
                        for p in sim.patches:
                            px = p.ipatch_x * sim.nx_per_patch
                            py = p.ipatch_y * sim.ny_per_patch
                            x_info = self._compute_patch_slice(sim.nx, sx, px, sim.nx_per_patch)
                            y_info = self._compute_patch_slice(sim.ny, sy, py, sim.ny_per_patch)
                            if x_info is None or y_info is None:
                                continue
                            local_x, out_x, num_x = x_info
                            local_y, out_y, num_y = y_info
                            data = getattr(p.fields, field)[local_x, local_y]
                            dset[out_x:out_x+num_x, out_y:out_y+num_y] = data
            else:
                with h5py.File(filename, 'w', driver='mpio', comm=comm) as f:
                    for field in self.components:
                        dset = f.create_dataset(
                            field, 
                            data=np.zeros((sim.nx, sim.ny)),
                            dtype='f8',
                            chunks=chunk_size
                        )
                        for p in sim.patches:
                            start = p.ipatch_x * sim.nx_per_patch, p.ipatch_y * sim.ny_per_patch
                            end   = start[0] + sim.nx_per_patch, start[1] + sim.ny_per_patch
                            data = getattr(p.fields, field)
                            dset[start[0]:end[0], start[1]:end[1]] = data[:sim.nx_per_patch, :sim.ny_per_patch]
        else: 
            if self._normalized_slice is not None:
                sx, sy = self._normalized_slice
                shape = (len(range(sx.start, sx.stop, sx.step)), len(range(sy.start, sy.stop, sy.step)))
                if rank == 0:
                    with h5py.File(filename, 'w') as f:
                        for field in self.components:
                            dset = f.create_dataset(
                                field, 
                                data=np.zeros(shape),
                                dtype='f8',
                                chunks=tuple(min(c, s) for c, s in zip(chunk_size, shape))
                            )
                comm.Barrier()
            
                with h5py.File(filename, 'a', locking=False) as f:
                    for field in self.components:
                        dset = f[field]
                        for p in sim.patches:
                            px = p.ipatch_x * sim.nx_per_patch
                            py = p.ipatch_y * sim.ny_per_patch
                            x_info = self._compute_patch_slice(sim.nx, sx, px, sim.nx_per_patch)
                            y_info = self._compute_patch_slice(sim.ny, sy, py, sim.ny_per_patch)
                            if x_info is None or y_info is None:
                                continue
                            local_x, out_x, num_x = x_info
                            local_y, out_y, num_y = y_info
                            data = getattr(p.fields, field)[local_x, local_y]
                            dset[out_x:out_x+num_x, out_y:out_y+num_y] = data
                comm.Barrier()
            else:
                if rank == 0:
                    with h5py.File(filename, 'w') as f:
                        for field in self.components:
                            dset = f.create_dataset(
                                field, 
                                data=np.zeros((sim.nx, sim.ny)),
                                dtype='f8',
                                chunks=chunk_size
                            )
                comm.Barrier()
        
                with h5py.File(filename, 'a', locking=False) as f:
                    for field in self.components:
                        dset = f[field]
                        for p in sim.patches:
                            start = p.ipatch_x * sim.nx_per_patch, p.ipatch_y * sim.ny_per_patch
                            end   = start[0] + sim.nx_per_patch, start[1] + sim.ny_per_patch
                            data = getattr(p.fields, field)
                            dset[start[0]:end[0], start[1]:end[1]] = data[:sim.nx_per_patch, :sim.ny_per_patch]

                comm.Barrier()    
        # Only rank 0 writes metadata
        if rank == 0:
            with h5py.File(filename, 'a') as f:
                f.attrs['nx'] = sim.nx
                f.attrs['ny'] = sim.ny
                f.attrs['dx'] = sim.dx
                f.attrs['dy'] = sim.dy
                f.attrs['Lx'] = sim.Lx
                f.attrs['Ly'] = sim.Ly
                f.attrs['time'] = sim.time
                f.attrs['itime'] = sim.itime
                if self._normalized_slice is not None:
                    f.attrs['slice'] = self._serialize_slice(self._normalized_slice, (sim.nx, sim.ny))
                
    def _write_3d(self, sim: Simulation3D, filename: Path):
        """Write 3D field data to HDF5 file in parallel.

        Args:
            sim (Simulation3D): 3D simulation object containing field data
            filename (str): Output HDF5 filename

        Note:
            - Creates HDF5 file with datasets for each field component
            - Uses MPI parallel I/O to write patch data
            - Includes simulation metadata as file attributes
        """
        # Get MPI communicator
        comm = sim.mpi.comm
        rank = comm.Get_rank()
        
        chunk_size = (sim.nx_per_patch, sim.ny_per_patch, sim.nz_per_patch)
        # Create filename with timestep
        if self.mpi:
            if self._normalized_slice is not None:
                sx, sy, sz = self._normalized_slice
                shape = (len(range(sx.start, sx.stop, sx.step)), len(range(sy.start, sy.stop, sy.step)), len(range(sz.start, sz.stop, sz.step)))
                with h5py.File(filename, 'w', driver='mpio', comm=comm) as f:
                    for field in self.components:
                        dset = f.create_dataset(
                            field, 
                            data=np.zeros(shape),
                            dtype='f8',
                            # chunks=chunk_size
                        )
                        comm.Barrier()
                        for p in sim.patches:
                            px = p.ipatch_x * sim.nx_per_patch
                            py = p.ipatch_y * sim.ny_per_patch
                            pz = p.ipatch_z * sim.nz_per_patch
                            x_info = self._compute_patch_slice(sim.nx, sx, px, sim.nx_per_patch)
                            y_info = self._compute_patch_slice(sim.ny, sy, py, sim.ny_per_patch)
                            z_info = self._compute_patch_slice(sim.nz, sz, pz, sim.nz_per_patch)
                            if x_info is None or y_info is None or z_info is None:
                                continue
                            local_x, out_x, num_x = x_info
                            local_y, out_y, num_y = y_info
                            local_z, out_z, num_z = z_info
                            data = getattr(p.fields, field)[local_x, local_y, local_z]
                            dset[out_x:out_x+num_x, out_y:out_y+num_y, out_z:out_z+num_z] = data
            else:
                with h5py.File(filename, 'w', driver='mpio', comm=comm) as f:
                    for field in self.components:
                        dset = f.create_dataset(
                            field, 
                            data=np.zeros((sim.nx, sim.ny, sim.nz)),
                            dtype='f8',
                            # chunks=chunk_size
                        )
                        comm.Barrier()
                        for p in sim.patches:
                            start = p.ipatch_x * sim.nx_per_patch, p.ipatch_y * sim.ny_per_patch, p.ipatch_z * sim.nz_per_patch
                            end   = start[0] + sim.nx_per_patch, start[1] + sim.ny_per_patch, start[2] + sim.nz_per_patch
                            data = getattr(p.fields, field)
                            dset[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = data[:sim.nx_per_patch, :sim.ny_per_patch, :sim.nz_per_patch]
        else:
            if self._normalized_slice is not None:
                sx, sy, sz = self._normalized_slice
                shape = (len(range(sx.start, sx.stop, sx.step)), len(range(sy.start, sy.stop, sy.step)), len(range(sz.start, sz.stop, sz.step)))
                if rank == 0:
                    with h5py.File(filename, 'w') as f:
                        for field in self.components:
                            dset = f.create_dataset(
                                field, 
                                data=np.zeros(shape),
                                dtype='f8',
                                chunks=tuple(min(c, s) for c, s in zip(chunk_size, shape))
                            )
                comm.Barrier()
            
                # Create chunked dataset with parallel access
                with h5py.File(filename, 'a', locking=False) as f:
                    for field in self.components:
                        dset = f[field]
                        for p in sim.patches:
                            px = p.ipatch_x * sim.nx_per_patch
                            py = p.ipatch_y * sim.ny_per_patch
                            pz = p.ipatch_z * sim.nz_per_patch
                            x_info = self._compute_patch_slice(sim.nx, sx, px, sim.nx_per_patch)
                            y_info = self._compute_patch_slice(sim.ny, sy, py, sim.ny_per_patch)
                            z_info = self._compute_patch_slice(sim.nz, sz, pz, sim.nz_per_patch)
                            if x_info is None or y_info is None or z_info is None:
                                continue
                            local_x, out_x, num_x = x_info
                            local_y, out_y, num_y = y_info
                            local_z, out_z, num_z = z_info
                            data = getattr(p.fields, field)[local_x, local_y, local_z]
                            dset[out_x:out_x+num_x, out_y:out_y+num_y, out_z:out_z+num_z] = data
                comm.Barrier()
            else:
                if rank == 0:
                    with h5py.File(filename, 'w') as f:
                        for field in self.components:
                            dset = f.create_dataset(
                                field, 
                                data=np.zeros((sim.nx, sim.ny, sim.nz)),
                                dtype='f8',
                                chunks=chunk_size
                            )
                comm.Barrier()
            
                # Create chunked dataset with parallel access
                with h5py.File(filename, 'a', locking=False) as f:
                    for field in self.components:
                        dset = f[field]
                        for p in sim.patches:
                            start = p.ipatch_x * sim.nx_per_patch, p.ipatch_y * sim.ny_per_patch, p.ipatch_z * sim.nz_per_patch
                            end   = start[0] + sim.nx_per_patch, start[1] + sim.ny_per_patch, start[2] + sim.nz_per_patch
                            data = getattr(p.fields, field)
                            dset[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = data[:sim.nx_per_patch, :sim.ny_per_patch, :sim.nz_per_patch]
                comm.Barrier()    
        # Only rank 0 writes metadata
        if rank == 0:
            with h5py.File(filename, 'a') as f:
                f.attrs['nx'] = sim.nx
                f.attrs['ny'] = sim.ny
                f.attrs['nz'] = sim.nz
                f.attrs['dx'] = sim.dx
                f.attrs['dy'] = sim.dy
                f.attrs['dz'] = sim.dz
                f.attrs['Lx'] = sim.Lx
                f.attrs['Ly'] = sim.Ly
                f.attrs['Lz'] = sim.Lz
                f.attrs['time'] = sim.time
                f.attrs['itime'] = sim.itime
                if self._normalized_slice is not None:
                    f.attrs['slice'] = self._serialize_slice(self._normalized_slice, (sim.nx, sim.ny, sim.nz))
        comm.Barrier()


class SaveSpeciesDensityToHDF5(Callback):
    """Callback to save species density data to HDF5 files.

    Creates a new HDF5 file for each save with name pattern:

    - `prefix/speciesname_000100.h5`
    - `prefix/speciesname_000200.h5`
    - ...

    The data structure in each file:

    - `/density` (2D or 3D array)

    Args:
        species (Species): The species whose density will be saved
        prefix (str): Prefix for output filenames. For example, if prefix is 'output', the files will be named 'output/{species.name}_000100.h5', 'output/{species.name}_000200.h5', etc.
        interval (Union[int, float, Callable], optional): Number of timesteps between saves, or a 
            function(sim) -> bool that determines when to save. Defaults to 100.
    """
    stage = "current_deposition"
    def __init__(self, species: Species, prefix: Union[str, Path]='', interval: Union[int, float, Callable] = 100, mpi: bool = False):
        self.species = species
        self.prefix = Path(prefix)
        self.prefix.mkdir(parents=True, exist_ok=True)
        self.mpi = mpi

        self.interval = interval
        self.prev_rho = None
        self.species = species
        os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)

    @property
    def ispec_target(self) -> int:
        """Get the target species index.

        Returns:
            int: The species index

        Raises:
            AssertionError: If species has not been initialized
        """
        return self.species.ispec
        
    def _call(self, sim: Union[Simulation, Simulation3D]):
        filename = self.prefix / f"{self.species.name}_{sim.itime:06d}.h5"
            
        if self.ispec_target == 0:
            if sim.ispec == 0:
                sim.sync_currents()
                density = self._compute_density(sim)
                if sim.dimension == 2:
                    self._write_2d(sim, density, filename)
                elif sim.dimension == 3:
                    self._write_3d(sim, density, filename)
        else:
            if sim.ispec == self.ispec_target - 1:
                sim.sync_currents()
                self.prev_rho = []
                for p in sim.patches:
                    self.prev_rho.append(p.fields.rho.copy())
            elif sim.ispec == self.ispec_target:
                sim.sync_currents()
                density = self._compute_density(sim)
                if sim.dimension == 2:
                    self._write_2d(sim, density, filename)
                elif sim.dimension == 3:
                    self._write_3d(sim, density, filename)
                self.prev_rho = None

    def _compute_density(self, sim: Union[Simulation, Simulation3D]) -> List[np.ndarray]:
        """Compute density from charge density for all patches.

        Args:
            sim (Union[Simulation, Simulation3D]): The simulation object containing field data

        Returns:
            List[np.ndarray]: List of density arrays (one per patch)
        """
        density = []
        for ip, p in enumerate(sim.patches):
            if self.ispec_target == 0:
                d = p.fields.rho / self.species.q
            else:
                d = (p.fields.rho - self.prev_rho[ip]) / self.species.q
            density.append(d)
        return density

    def _write_2d(self, sim: Simulation, density_per_patch: List[np.ndarray], filename: Path):
        """Write 2D density data to HDF5 file in parallel.

        Args:
            sim (Simulation): 2D simulation object
            density_per_patch (List[np.ndarray]): List of density arrays (one per patch)

        Note:
            - Creates HDF5 file with density dataset
            - Uses MPI parallel I/O to write patch data
            - Includes simulation metadata as file attributes
        """
        comm = sim.mpi.comm
        rank = comm.Get_rank()

        if self.mpi:
            with h5py.File(filename, 'w', driver='mpio', comm=comm) as f:
                dset = f.create_dataset(
                    'density', 
                    data=np.zeros((sim.nx, sim.ny), dtype='f8'),
                    chunks=(sim.nx_per_patch, sim.ny_per_patch)
                )
                for ip, p in enumerate(sim.patches):
                    start = p.ipatch_x * sim.nx_per_patch, p.ipatch_y * sim.ny_per_patch
                    end   = start[0] + sim.nx_per_patch, start[1] + sim.ny_per_patch
                    dset[start[0]:end[0], start[1]:end[1]] = density_per_patch[ip][:sim.nx_per_patch, :sim.ny_per_patch]
        else:
            if rank == 0:
                with h5py.File(filename, 'w') as f:
                    dset = f.create_dataset(
                        'density', 
                        data=np.zeros((sim.nx, sim.ny), dtype='f8'),
                        chunks=(sim.nx_per_patch, sim.ny_per_patch)
                    )
            comm.Barrier()

            with h5py.File(filename, 'a', locking=False) as f:
                dset = f['density']
                for ip, p in enumerate(sim.patches):
                    start = p.ipatch_x * sim.nx_per_patch, p.ipatch_y * sim.ny_per_patch
                    end   = start[0] + sim.nx_per_patch, start[1] + sim.ny_per_patch
                    dset[start[0]:end[0], start[1]:end[1]] = density_per_patch[ip][:sim.nx_per_patch, :sim.ny_per_patch]
            comm.Barrier()

        if rank == 0:
            with h5py.File(filename, 'a') as f:
                f.attrs['time'] = sim.time
                f.attrs['itime'] = sim.itime
                f.attrs['species'] = self.species.name
                f.attrs['nx'] = sim.nx
                f.attrs['ny'] = sim.ny
                f.attrs['dx'] = sim.dx
                f.attrs['dy'] = sim.dy
                f.attrs['Lx'] = sim.Lx
                f.attrs['Ly'] = sim.Ly

    def _write_3d(self, sim: Simulation3D, density_per_patch: List[np.ndarray], filename: Path):
        """Write 3D density data to HDF5 file in parallel.

        Args:
            sim (Simulation3D): 3D simulation object
            density_per_patch (List[np.ndarray]): List of density arrays (one per patch)

        Note:
            - Creates HDF5 file with density dataset
            - Uses MPI parallel I/O to write patch data
            - Includes simulation metadata as file attributes
        """
        comm = sim.mpi.comm
        rank = comm.Get_rank()

        if self.mpi:
            with h5py.File(filename, 'w', driver='mpio', comm=comm) as f:
                dset = f.create_dataset(
                    'density', 
                    data=np.zeros((sim.nx, sim.ny, sim.nz), dtype='f8'),
                    dtype='f8',
                    chunks=(sim.nx_per_patch, sim.ny_per_patch, sim.nz_per_patch)
                )

                for ip, p in enumerate(sim.patches):
                    start = p.ipatch_x * sim.nx_per_patch, p.ipatch_y * sim.ny_per_patch, p.ipatch_z * sim.nz_per_patch
                    end   = start[0] + sim.nx_per_patch, start[1] + sim.ny_per_patch, start[2] + sim.nz_per_patch
                    dset[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = density_per_patch[ip][:sim.nx_per_patch, :sim.ny_per_patch, :sim.nz_per_patch]
        else:
            if rank == 0:
                with h5py.File(filename, 'w') as f:
                    dset = f.create_dataset(
                        'density', 
                        data=np.zeros((sim.nx, sim.ny, sim.nz), dtype='f8'),
                        dtype='f8',
                        chunks=(sim.nx_per_patch, sim.ny_per_patch, sim.nz_per_patch)
                    )
            comm.Barrier()

            with h5py.File(filename, 'a', locking=False) as f:
                dset = f['density']
                for ip, p in enumerate(sim.patches):
                    start = p.ipatch_x * sim.nx_per_patch, p.ipatch_y * sim.ny_per_patch, p.ipatch_z * sim.nz_per_patch
                    end   = start[0] + sim.nx_per_patch, start[1] + sim.ny_per_patch, start[2] + sim.nz_per_patch
                    dset[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = density_per_patch[ip][:sim.nx_per_patch, :sim.ny_per_patch, :sim.nz_per_patch]
            comm.Barrier()

        if rank == 0:
            with h5py.File(filename, 'a') as f:
                f.attrs['time'] = sim.time
                f.attrs['itime'] = sim.itime
                f.attrs['species'] = self.species.name
                f.attrs['nx'] = sim.nx
                f.attrs['ny'] = sim.ny
                f.attrs['nz'] = sim.nz
                f.attrs['dx'] = sim.dx
                f.attrs['dy'] = sim.dy
                f.attrs['dz'] = sim.dz
                f.attrs['Lx'] = sim.Lx
                f.attrs['Ly'] = sim.Ly
                f.attrs['Lz'] = sim.Lz

        comm.Barrier()

class SaveParticlesToHDF5(Callback):
    """Callback to save particle data to HDF5 files.

    Creates a new HDF5 file for each save with name pattern:

    - `prefix/{species.name}_particles_000100.h5`
    - `prefix/{species.name}_particles_000200.h5`
    - ...

    The data structure in each file:
    
    - `/id`
    - `/x, y` (positions)
    - `/w` (weights)
    - `/...` (other specified attributes)

    Args:
        species (Species): The particle species to save
        prefix (str): Prefix for output filenames. For example, if prefix is 'output', the files will be named 'output/{species.name}_particles_0000100.h5'.
        interval (Union[int, float, Callable], optional): Number of timesteps between saves, or a
            function(sim) -> bool that determines when to save. Defaults to 100.
        attrs (Optional[List[str]], optional): List of particle attributes to save.
            If None, saves all attributes.
    """
    DEFAULT_STAGE = "end"
    def __init__(self,
                 species: Species,
                 prefix: Union[str, Path]='',
                 interval: Union[int, float, Callable] = 100,
                 attrs: Optional[List[str]] = None) -> None:
        self.stage = self.DEFAULT_STAGE
        self.prefix = Path(prefix)
        self.prefix.mkdir(parents=True, exist_ok=True)

        self.interval = interval
        self.attrs = attrs
        self.species = species
        
        if self.attrs is None:
            logger.warning("No attributes specified, saving all attributes.")
        elif 'id' in self.attrs:
            self.attrs.remove('id')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    
    def _call(self, sim: Union[Simulation, Simulation3D]):
        if self.attrs is None:
            self.attrs = sim.patches[0].particles[self.species.ispec].attrs
            if 'id' in self.attrs:
                self.attrs.remove('id')

        comm = sim.mpi.comm
        rank = comm.Get_rank()
        # gather number of particles in each patch
        npart_patches = [p.particles[self.species.ispec].is_alive.sum() for p in sim.patches]
        npart_allpatches = comm.allgather(npart_patches)

        # Create new file for this timestep
        filename = self.prefix / f"{self.species.name}_particles_{sim.itime:06d}.h5"
        if rank == 0:
            npart_total = sum([sum(n) for n in npart_allpatches])
            with h5py.File(filename, 'w') as f:
                for attr in self.attrs:
                    f.create_dataset(attr, data=np.zeros((npart_total,)), dtype='f8')
                f.create_dataset('id', data=np.zeros((npart_total,)), dtype='u8')
        
        comm.Barrier()
        
        # Create chunked dataset with parallel access
        with h5py.File(filename, 'a', locking=False) as f:
            start = sum([sum(n) for n in npart_allpatches[:rank]])
            for ipatch, p in enumerate(sim.patches):
                is_alive = p.particles[self.species.ispec].is_alive
                for attr in self.attrs:
                    dset = f[attr]
                    data = getattr(p.particles[self.species.ispec], attr)
                    dset[start:start+npart_patches[ipatch]] = data[is_alive]
                f['id'][start:start+npart_patches[ipatch]] = p.particles[self.species.ispec].id[is_alive]
                start += npart_patches[ipatch]

        comm.Barrier()
        # Only rank 0 writes metadata
        if rank == 0:
            with h5py.File(filename, 'a') as f:
                f.attrs['time'] = sim.time
                f.attrs['itime'] = sim.itime
        comm.Barrier()
        
