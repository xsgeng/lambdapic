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
    stage = "start"
    
    def __init__(
        self, 
        velocity: Union[float, Callable[[float], float]], 
        direction: str = 'x', 
        start_time: float|None = None
    ):
        # numpy docstring
        """
        Initialize moving window callback

        Args:
            velocity (float or callable): 
                Window velocity in m/s, constant or function of time. When function of time, the time start after `start_time`.
            direction (str): Direction of movement ('x', 'y', or 'z')
            start_time (float): Time at which to start moving the window
        """
        self.velocity = velocity
        self.direction = direction
        self.start_time = start_time
        self.total_shift = 0.0
        self.patch_this_shift = 0.0
        self.num_shifts: int = 0

        # Only x-direction is currently implemented
        if self.direction != 'x':
            raise NotImplementedError("Only x-direction moving window is currently supported")
        
    def __call__(self, sim: Simulation):
        if self.start_time is None:
            self.start_time = sim.Lx/c

        if sim.time <= self.start_time:
            return
        
        # Calculate current window velocity
        if callable(self.velocity):
            current_velocity = self.velocity(sim.time)
        else:
            current_velocity = self.velocity
            
        # Calculate shift amount based on velocity and time interval
        shift_amount = current_velocity * sim.dt
        self.total_shift += shift_amount
        self.patch_this_shift += shift_amount
        
        if -sim.nx_per_patch * sim.dx < self.patch_this_shift < sim.nx_per_patch * sim.dx:
            return
        
        if self.patch_this_shift > sim.nx_per_patch * sim.dx:
            self._shift_right(sim)
            self.patch_this_shift -= sim.nx_per_patch * sim.dx
        elif self.patch_this_shift < -sim.nx_per_patch * sim.dx:
            self._shift_left(sim)
            self.patch_this_shift += sim.nx_per_patch * sim.dx
        self.num_shifts += 1
        
        self._update_patch_info(sim)

    def _shift_right(self, sim: Simulation):
        for p in sim.patches:
            if p.ipatch_x == 0:
                p.ipatch_x = sim.npatch_x - 1
                p.x0 += sim.Lx
                p.fields.xaxis += sim.Lx
            else:
                p.ipatch_x -= 1
                p.x0 += sim.nx_per_patch * sim.dx
                p.fields.xaxis += sim.nx_per_patch * sim.dx
                
    def _shift_left(self, sim: Simulation):
        for p in sim.patches:
            if p.ipatch_x == sim.npatch_x - 1:
                p.ipatch_x = 0
                p.x0 -= sim.Lx
                p.fields.xaxis -= sim.Lx
            else:
                p.ipatch_x += 1
                p.x0 -= sim.nx_per_patch * sim.dx
                p.fields.xaxis -= sim.nx_per_patch * sim.dx
                
    def _update_patch_info(self, sim: Simulation):
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
        
    def _fill_particles(self, sim: Simulation):
        pass