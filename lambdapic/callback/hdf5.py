import os
import h5py
import numpy as np
from typing import List, Optional, Set, Union, Callable
from .utils import get_fields

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
            
    def __call__(self, sim):
        """Save field data if current timestep is at save interval"""
        if callable(self.interval):
            if not self.interval(sim):
                return
        elif sim.itime % self.interval != 0:
            return
            
        # Get complete fields
        field_data = get_fields(sim, self.components)
            
        # Create new file for this timestep
        filename = f"{self.prefix}_t{sim.itime:06d}.h5"
        with h5py.File(filename, 'w') as f:
            # Save field data
            for field, data in zip(self.components, field_data):
                f.create_dataset(field, data=data)
                
            # Save metadata
            f.attrs['nx'] = sim.nx
            f.attrs['ny'] = sim.ny
            f.attrs['dx'] = sim.dx
            f.attrs['dy'] = sim.dy
            f.attrs['Lx'] = sim.Lx
            f.attrs['Ly'] = sim.Ly
            f.attrs['time'] = sim.time
            f.attrs['itime'] = sim.itime


class SaveParticlesToHDF5:
    """
    Callback to save particle data to HDF5 files.
    
    Creates a new HDF5 file for each save with name pattern:
    'prefix_t000100.h5', 'prefix_t000200.h5', etc.
    
    The data structure in each file:
    - /species/
        - species_0/
            - patch_0/
                - x, y (positions)
                - w (weights)
                - other particle attributes
            - patch_1/
        - species_1/
        ...
    
    Args:
        prefix: Prefix for output filenames
        interval: Number of timesteps between saves, or a function(sim) -> bool
                 that determines when to save
        attrs: List of particle attributes to save.
               If None, saves all attributes
    """
    stage="maxwell second"
    def __init__(self,
                 prefix: str,
                 interval: Union[int, Callable] = 100,
                 attrs: Optional[List[str]] = None) -> None:
        self.prefix = prefix
        self.interval = interval
        self.attrs = attrs
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
            
    def __call__(self, sim):
        """Save particle data if current timestep is at save interval"""
        if callable(self.interval):
            if not self.interval(sim):
                return
        elif sim.itime % self.interval != 0:
            return
            
        # Create new file for this timestep
        filename = f"{self.prefix}_t{sim.itime:06d}.h5"
        with h5py.File(filename, 'w') as f:
            species_group = f.create_group('species')
            
            for i, species in enumerate(sim.species):
                species_subgroup = species_group.create_group(f'species_{i}')
                species_subgroup.attrs['name'] = species.name
                
                for patch in sim.patches:
                    patch_group = species_subgroup.create_group(f'patch_{patch.index}')
                    particles = patch.particles[i]
                    
                    # Save number of particles
                    patch_group.attrs['npart'] = particles.npart
                    
                    # Determine which attributes to save
                    attrs_to_save = (self.attrs if self.attrs is not None 
                                   else particles.attrs)
                    
                    # Save selected particle attributes
                    for attr in attrs_to_save:
                        if attr not in particles.attrs:
                            continue
                        data = getattr(particles, attr)
                        if attr in ['x', 'y', 'w']:  # Only save active particles
                            data = data[:particles.npart]
                        patch_group.create_dataset(attr, data=data)
