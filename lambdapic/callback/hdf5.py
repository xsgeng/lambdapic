import os
from typing import Callable, List, Optional, Set, Union

import h5py
import numpy as np

from ..simulation import Simulation, Simulation3D
from .utils import get_fields
from ..core.species import Species

class SaveFieldsToHDF5:
    """
    Callback to save field data to HDF5 files.
    
    Creates a new HDF5 file for each save with name pattern:
    'prefix_t000100.h5', 'prefix_t000200.h5', etc.
    
    The data structure in each file:
    - /fields/
        - ex, ey, ez (electric fields)
        - bx, by, bz (magnetic fields)
        - jx, jy, jz (currents)
        - rho (charge density)
    
    Args:
        prefix: Prefix for output filenames
        interval: Number of timesteps between saves, or a function(sim) -> bool
                 that determines when to save
        components: List of field components to save.
                   Available: ['ex','ey','ez','bx','by','bz','jx','jy','jz','rho']
                   If None, saves all components
    """
    stage="maxwell second"
    def __init__(self, 
                 prefix: str, 
                 interval: Union[int, Callable] = 100,
                 components: Optional[List[str]] = None) -> None:
        self.prefix = prefix
        self.interval = interval
        
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
    
    def __call__(self, sim: Simulation|Simulation3D):
        """Save field data if current timestep is at save interval"""
        if callable(self.interval):
            if not self.interval(sim):
                return
        elif sim.itime % self.interval != 0:
            return
        
        filename = f"{self.prefix}_t{sim.itime:06d}.h5"
        if sim.dimension == 2:
            self._write_2d(sim, filename)
        elif sim.dimension == 3:
            self._write_3d(sim, filename)
    
    def _write_2d(self, sim: Simulation, filename: str):
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
                
    def _write_3d(self, sim: Simulation3D, filename: str):
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


class SaveSpeciesDensityToHDF5:
    stage = "current deposition"
    def __init__(self, species: Species, prefix: str, interval: Union[int, Callable] = 100):
        self.species = species
        self.prefix = prefix
        self.interval = interval
        self.prev_rho = None
        self.species = species
        os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)

    @property
    def ispec_target(self):
        assert self.species.ispec >= 0, f"Species {self.species.name} has not been initialized."
        return self.species.ispec
    
    def __call__(self, sim: Simulation):
        if callable(self.interval):
            if not self.interval(sim):
                return
        elif sim.itime % self.interval != 0:
            return
            
        if self.ispec_target == 0:
            if sim.ispec == 0:
                density = self._compute_density(sim)
                if sim.dimension == 2:
                    self._write_2d(sim, density)
                elif sim.dimension == 3:
                    self._write_3d(sim, density)
        else:
            if sim.ispec == self.ispec_target - 1:
                self.prev_rho = []
                for p in sim.patches:
                    self.prev_rho.append(p.fields.rho.copy())
            elif sim.ispec == self.ispec_target:
                density = self._compute_density(sim)
                if sim.dimension == 2:
                    self._write_2d(sim, density)
                elif sim.dimension == 3:
                    self._write_3d(sim, density)
                self.prev_rho = None

    def _compute_density(self, sim: Simulation) -> list:
        density = []
        for ip, p in enumerate(sim.patches):
            if self.ispec_target == 0:
                d = p.fields.rho / self.species.q
            else:
                d = (p.fields.rho - self.prev_rho[ip]) / self.species.q
            density.append(d)
        return density

    def _write_2d(self, sim: Simulation, density_per_patch: list):
        comm = sim.mpi.comm
        rank = comm.Get_rank()
        filename = f"{self.prefix}_t{sim.itime:06d}_{self.species.name}.h5"
        
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

    def _write_3d(self, sim: Simulation3D, density_per_patch: list):
        comm = sim.mpi.comm
        rank = comm.Get_rank()
        filename = f"{self.prefix}_t{sim.itime:06d}_{self.species.name}.h5"
        
        if rank == 0:
            with h5py.File(filename, 'w') as f:
                dset = f.create_dataset(
                    'density', 
                    (sim.nx, sim.ny, sim.nz),
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

class SaveParticlesToHDF5:
    """
    Callback to save particle data to HDF5 files.
    
    Creates a new HDF5 file for each save with name pattern:
    'prefix_t000100.h5', 'prefix_t000200.h5', etc.
    
    The data structure in each file:
    - /id
    - /x, y (positions)
    - /w (weights)
    - /...
    
    Args:
        prefix: Prefix for output filenames
        interval: Number of timesteps between saves, or a function(sim) -> bool
                 that determines when to save
        attrs: List of particle attributes to save.
               If None, saves all attributes
    """
    stage="maxwell second"
    def __init__(self,
                 species: Species,
                 prefix: str='',
                 interval: Union[int, Callable] = 100,
                 attrs: Optional[List[str]] = None) -> None:
        self.prefix = prefix
        self.interval = interval
        self.attrs = attrs
        self.species = species

        self.attrs.remove('id')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
            
    def __call__(self, sim: Simulation):
        """Save particle data if current timestep is at save interval"""
        if callable(self.interval):
            if not self.interval(sim):
                return
        elif sim.itime % self.interval != 0:
            return
        
        comm = sim.mpi.comm
        rank = comm.Get_rank()
        # gather number of particles in each patch
        npart_patches = [p.particles[self.species.ispec].is_alive.sum() for p in sim.patches]
        npart_allpatches = comm.allgather(npart_patches)

        # Create new file for this timestep
        filename = f"{self.prefix}_t{sim.itime:06d}_{self.species.name}.h5"
        attrs = sim.patches[0].particles[self.species.ispec].attrs if self.attrs is None else self.attrs
        if rank == 0:
            npart_total = sum([sum(n) for n in npart_allpatches])
            with h5py.File(filename, 'w') as f:
                for attr in attrs:
                    f.create_dataset(attr, data=np.zeros((npart_total,)), dtype='f8')
                f.create_dataset('id', data=np.zeros((npart_total,)), dtype='u8')
        
        comm.Barrier()
        
        # Create chunked dataset with parallel access
        with h5py.File(filename, 'a', locking=False) as f:
            start = sum([sum(n) for n in npart_allpatches[:rank]])
            for ipatch, p in enumerate(sim.patches):
                is_alive = p.particles[self.species.ispec].is_alive
                for attr in attrs:
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
