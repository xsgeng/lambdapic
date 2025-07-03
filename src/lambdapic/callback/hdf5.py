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
    'prefix_t000100.h5', 'prefix_t000200.h5', etc.

    The data structure in each file:
    - /ex, /ey, /ez (electric fields)
    - /bx, /by, /bz (magnetic fields)
    - /jx, /jy, /jz (currents)
    - /rho (charge density)

    Args:
        prefix (str): Prefix for output filenames. For example, if prefix is 'output', the files will be named 'output/t000100.h5', 'output/t000200.h5', etc.
        interval (Union[int, Callable], optional): Number of timesteps between saves, or a 
            function(sim) -> bool that determines when to save. Defaults to 100.
        components (Optional[List[str]], optional): List of field components to save. 
            Available: ['ex','ey','ez','bx','by','bz','jx','jy','jz','rho']. 
            If None, saves all components.

    Attributes:
        stage (str): The simulation stage when this callback is executed.
        all_components (Set[str]): Set of all available field components.
        components (List[str]): List of components to actually save.
    """
    stage="maxwell second"
    def __init__(self, 
                 prefix: Union[str, Path]='', 
                 interval: Union[int, Callable] = 100,
                 components: Optional[List[str]] = None) -> None:
        self.prefix = Path(prefix)
        self.interval = interval

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
        
    def _call(self, sim: Union[Simulation, Simulation3D]):
        filename = self.prefix / f"{sim.itime:06d}.h5"
        if sim.dimension == 2:
            if not isinstance(sim, Simulation):
                raise TypeError("Expected Simulation for 2D case")
            self._write_2d(sim, filename)
        elif sim.dimension == 3:
            if not isinstance(sim, Simulation3D):
                raise TypeError("Expected Simulation3D for 3D case")
            self._write_3d(sim, filename)
        
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
        
        # Create chunked dataset with parallel access
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
        comm.Barrier()


class SaveSpeciesDensityToHDF5(Callback):
    """Callback to save species density data to HDF5 files.

    Creates a new HDF5 file for each save with name pattern:
    'prefix_speciesname_t000100.h5', 'prefix_speciesname_t000200.h5', etc.

    The data structure in each file:
    - /density (2D or 3D array)

    Args:
        species (Species): The species whose density will be saved
        prefix (str): Prefix for output filenames. For example, if prefix is 'output', the files will be named 'output/{species.name}_t000100.h5', 'output/{species.name}_t000200.h5', etc.
        interval (Union[int, Callable], optional): Number of timesteps between saves, or a 
            function(sim) -> bool that determines when to save. Defaults to 100.

    Attributes:
        stage (str): The simulation stage when this callback is executed.
        species (Species): The species being tracked.
        prev_rho (Optional[List[np.ndarray]]): Previous charge density values for computation.
    """
    stage = "current deposition"
    def __init__(self, species: Species, prefix: Union[str, Path]='', interval: Union[int, Callable] = 100):
        self.species = species
        self.prefix = Path(prefix)
        self.prefix.mkdir(parents=True, exist_ok=True)

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
        if self.species.ispec is None:
            raise ValueError(f"Species {self.species.name} has not been initialized.")
        return self.species.ispec
        
    def _call(self, sim: Union[Simulation, Simulation3D]):
        filename = self.prefix / f"{self.species.name}_{sim.itime:06d}.h5"
            
        if self.ispec_target == 0:
            if sim.ispec == 0:
                density = self._compute_density(sim)
                if sim.dimension == 2:
                    self._write_2d(sim, density, filename)
                elif sim.dimension == 3:
                    self._write_3d(sim, density, filename)
        else:
            if sim.ispec == self.ispec_target - 1:
                self.prev_rho = []
                for p in sim.patches:
                    self.prev_rho.append(p.fields.rho.copy())
            elif sim.ispec == self.ispec_target:
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

    The data structure in each file:
    - /id
    - /x, y (positions)
    - /w (weights)
    - /... (other specified attributes)

    Args:
        species (Species): The particle species to save
        prefix (str): Prefix for output filenames. For example, if prefix is 'output', the files will be named 'output/{species.name}_particles_0000100.h5'.
        interval (Union[int, Callable], optional): Number of timesteps between saves, or a
            function(sim) -> bool that determines when to save. Defaults to 100.
        attrs (Optional[List[str]], optional): List of particle attributes to save.
            If None, saves all attributes.

    Attributes:
        stage (str): The simulation stage when this callback is executed.
        species (Species): The particle species being tracked.
    """
    stage="maxwell second"
    def __init__(self,
                 species: Species,
                 prefix: Union[str, Path]='',
                 interval: Union[int, Callable] = 100,
                 attrs: Optional[List[str]] = None) -> None:
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
        
