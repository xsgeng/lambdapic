from ..simulation import Simulation
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi
import numpy as np

from typing import Callable, Union, List

from ..core.species import Species
from ..core.utils.logger import logger

from pathlib import Path


def get_fields(sim: Simulation, fields: List[str]) -> list[np.ndarray]:
    """
    Get fields from all patches.
    
    Only rank 0 will gather the data, other ranks will get None.
    
    Args:
        sim (Simulation): Simulation instance.
        fields (List[str]): List of fields to get.
        
    Returns:
        list[np.ndarray]: List of fields named as field.
    """
    ret = []
    patches = sim.patches
    nx_per_patch = sim.nx_per_patch
    ny_per_patch = sim.ny_per_patch
    npatch_x = sim.npatch_x
    npatch_y = sim.npatch_y
    nx = sim.nx
    ny = sim.ny
    ng = sim.n_guard

    assert sim.dimension == 2, "Only 2D simulation is supported"
    
    if not fields:
        return ret
    
    for field in fields:
        if sim.mpi.rank == 0:
            local_patches = {p.index: ipatch for ipatch, p in enumerate(patches)}
            field_ = np.zeros((nx, ny))
            
            buf = np.zeros((nx_per_patch+2*ng, ny_per_patch+2*ng))
            for ipatch_x in range(npatch_x):
                for ipatch_y in range(npatch_y):
                    s = np.s_[ipatch_x*nx_per_patch:ipatch_x*nx_per_patch+nx_per_patch,\
                              ipatch_y*ny_per_patch:ipatch_y*ny_per_patch+ny_per_patch]
                    # local
                    index = ipatch_y*npatch_x + ipatch_x
                    if index in local_patches:
                        p = patches[local_patches[index]]
                        field_[s] = getattr(p.fields, field)[:-2*ng, :-2*ng]
                    #remote
                    else:
                        sim.mpi.comm.Recv(buf, tag=index)
                        field_[s] = buf[:-2*ng, :-2*ng]
                        
            ret.append(field_)
        else: # other ranks
            req = []
            for p in patches:
                req_ = sim.mpi.comm.Isend(getattr(p.fields, field), dest=0, tag=p.index)
                req.append(req_)
            for req_ in req:
                req_.wait()

            ret.append(None)
        sim.mpi.comm.Barrier()
        
    return ret

class ExtractSpeciesDensity:
    stage = "current deposition"
    def __init__(self, sim: Simulation, species: Species, every: Union[int, Callable]):
        self.sim = sim
        self.species = species
        self.every = every
        self.ispec_target = sim.species.index(species)
        
        self.patches = sim.patches
        self.nx_per_patch = sim.nx_per_patch
        self.ny_per_patch = sim.ny_per_patch
        self.n_guard = sim.n_guard

        self.density = np.zeros((self.nx_per_rank, self.ny_per_rank))

    def _get_patch_slice(self, patch):
        return np.s_[
            patch.ipatch_x*self.nx_per_patch:(patch.ipatch_x+1)*self.nx_per_patch,
            patch.ipatch_y*self.ny_per_patch:(patch.ipatch_y+1)*self.ny_per_patch
        ]

    def __call__(self, sim: Simulation):
        if callable(self.every):
            if not self.every(sim):
                return
        elif sim.itime % self.every != 0:
            return

        ispec = sim.ispec
        if self.ispec_target == 0:
            if ispec == 0:
                for p in self.patches:
                    s = self._get_patch_slice(p)
                    self.density[s] = p.fields.rho[:-2*self.n_guard, :-2*self.n_guard] / self.sim.species[ispec].q
        else:
            if ispec == self.ispec_target - 1:
                for p in self.patches:
                    s = self._get_patch_slice(p)
                    # store previous rho
                    self.density[s] = p.fields.rho[:-2*self.n_guard, :-2*self.n_guard]
            if ispec == self.ispec_target:
                for p in self.patches:
                    s = self._get_patch_slice(p)
                    # subtract previous rho
                    self.density[s] = p.fields.rho[:-2*self.n_guard, :-2*self.n_guard] - self.density[s]
                    self.density[s] /= self.sim.species[ispec].q


def species_transfer(s1, s2):
    # if ..., s1 particles become s2
    pass


class MovingWindow:
    """Callback to implement a moving simulation window"""
    stage = "start"  # Apply at start of each time step
    
    def __init__(self, every: int, direction: str = 'x'):
        """
        Initialize moving window callback
        
        Args:
            every: Apply moving window every N time steps
            direction: Direction of movement ('x', 'y', or 'z')
        """
        self.every = every
        self.direction = direction
        
    def __call__(self, sim: Simulation):
        """Apply moving window shift if needed"""
        if sim.itime % self.every != 0:
            return
            
        # Only x-direction is currently implemented
        if self.direction != 'x':
            raise NotImplementedError("Only x-direction moving window is currently supported")
            
        # Shift patches
        for p in sim.patches:
            if p.ipatch_x == 0:
                p.ipatch_x = sim.npatch_x - 1
                p.x0 += sim.Lx
                p.fields.xaxis += sim.Lx
            else:
                p.ipatch_x -= 1
                p.x0 += sim.nx_per_patch * sim.dx
                p.fields.xaxis += sim.nx_per_patch * sim.dx
                
            # Update global index
            p.index = p.ipatch_x + p.ipatch_y * sim.npatch_x

        # Gather updated patch information from all ranks
        comm = sim.mpi.comm
        rank = comm.Get_rank()
        
        # Prepare local patch info
        local_info = []
        for p in sim.patches:
            local_info.append((p.ipatch_x, p.ipatch_y, p.index, rank))
            
        # Gather all patch info
        all_info = comm.allgather(local_info)
        
        # Build global mappings
        patch_index_map = {}
        patch_rank_map = {}
        for rank_info in all_info:
            for (ipx, ipy, idx, r) in rank_info:
                patch_index_map[(ipx, ipy)] = idx
                patch_rank_map[idx] = r
                
        # Reinitialize neighbor relationships
        sim.patches.init_rect_neighbor_index_2d(sim.npatch_x, sim.npatch_y, patch_index_map)
        sim.patches.init_neighbor_ipatch_2d()
        sim.patches.init_neighbor_rank_2d(patch_rank_map)
        
        # Log the shift
        if rank == 0:
            logger.info(f"Applied moving window shift at step {sim.itime}")
