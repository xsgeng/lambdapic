from ast import If
from lambdapic.core.boundary.cpml import PMLX
from ..simulation import Simulation
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi
import numpy as np

from typing import Callable, Sequence, Union, List

from ..core.species import Species
from ..core.utils.logger import logger

from pathlib import Path

from ..core.patch.cpu import fill_particles_2d, get_num_macro_particles_2d
from ..core.patch.patch import Patch, Patch2D, Patch3D
from numba import typed


def get_fields(sim: Simulation, fields: Sequence[str]) -> Sequence[np.ndarray]:
    """
    Get fields from all patches.
    
    Only rank 0 will gather the data, other ranks will get None.
    
    Args:
        sim (Simulation): Simulation instance.
        fields (Sequence[str]): List of fields to get.
        
    Returns:
        Sequence[np.ndarray]: List of fields named as field.
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
    
    index_info = {(p.ipatch_x, p.ipatch_y): p.index for ipatch, p in enumerate(patches)}
    index_info: list[dict[tuple[int, int], int]] = sim.mpi.comm.gather(index_info)
    if sim.mpi.rank == 0:
        patch_index_map = {k: v for d in index_info for k, v in d.items()}
        local_patches = {p.index: ipatch for ipatch, p in enumerate(patches)}

    for field in fields:
        if sim.mpi.rank == 0:
            field_ = np.zeros((nx, ny))
            
            buf = np.zeros((nx_per_patch+2*ng, ny_per_patch+2*ng))
            for ipatch_x in range(npatch_x):
                for ipatch_y in range(npatch_y):
                    s = np.s_[ipatch_x*nx_per_patch:ipatch_x*nx_per_patch+nx_per_patch,\
                              ipatch_y*ny_per_patch:ipatch_y*ny_per_patch+ny_per_patch]
                    # local
                    index = patch_index_map[(ipatch_x, ipatch_y)]
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
    """Callback implementing moving window technique along x-direction.

    The moving window follows the laser or plasma flow by periodically shifting
    the simulation domain while maintaining proper boundary conditions.

    Args:
        velocity (Union[float, Callable[[float], float]]): Window velocity in m/s. 
            Can be constant or function of time (velocity=f(sim.time))
        start_time (Optional[float]): Time at which to start moving window. 
            Defaults to sim.Lx/c.
        inject_particles (bool): Whether to inject particles in new regions. 
            Defaults to True.
        stop_inject_time (Optional[float]): Time to stop particle injection. 
            Defaults to None.

    Attributes:
        stage (str): The simulation stage when this callback is executed.
        total_shift (Optional[float]): Total accumulated shift distance.
        patch_this_shift (Optional[float]): Shift amount within current patch.
        num_shifts (int): Number of shifts performed.

    Note:
        - Handles both forward (positive) and backward (negative) moving windows
        - Maintains proper particle distributions in new regions
        - Updates patch neighbor relationships after shifts
        - Removes PML boundaries when moving starts
    """
    stage = "start"
    
    def __init__(
        self, 
        velocity: Union[float, Callable[[float], float]], 
        start_time: float|None = None,
        inject_particles: bool = True,
        stop_inject_time: float|None = None,
    ):
        self.velocity = velocity
        self.start_time = start_time
        self.inject_particles = inject_particles
        self.stop_inject_time = stop_inject_time

        self.total_shift = None
        self.patch_this_shift = None
        self.num_shifts: int = 0

    def __call__(self, sim: Simulation):
        """Execute moving window operation for current timestep.

        Args:
            sim (Simulation): The simulation object to operate on

        Note:
            - Calculates shift amount based on velocity and timestep
            - Performs patch shifts when accumulated shift exceeds patch size
            - Updates particle distributions in new regions
            - Maintains proper boundary conditions
        """
        patch_Lx = sim.nx_per_patch * sim.dx

        if self.start_time is None:
            self.start_time = sim.Lx/c

        if self.total_shift is None:
            self.total_shift = patch_Lx

        if self.patch_this_shift is None:
            self.patch_this_shift = patch_Lx

        if sim.time < self.start_time:
            return
        
        if self.num_shifts == 0:
            if sim.mpi.rank == 0:
                logger.info("MovingWindow starts.")
            
            logger.info(f"Rank {sim.mpi.rank}: removing PMLX")
            for p in sim.patches:
                # clear PMLX
                if p.pml_boundary:
                    p.pml_boundary = [pml for pml in p.pml_boundary if not issubclass(type(pml), PMLX)]
            sim.maxwell.generate_field_lists()
        
        # Calculate current window velocity
        if callable(self.velocity):
            current_velocity = self.velocity(sim.time)
        else:
            current_velocity = self.velocity
            
        # Calculate shift amount based on velocity and time interval
        shift_amount = current_velocity * sim.dt
        self.total_shift += shift_amount
        self.patch_this_shift += shift_amount

        self.num_shifts += 1

        if self.patch_this_shift >= patch_Lx:
            new_patches = self._shift_right(sim)
            self.patch_this_shift -= patch_Lx
        elif self.patch_this_shift <= -patch_Lx:
            new_patches = self._shift_left(sim)
            self.patch_this_shift += patch_Lx
        else:
            return
        
        self._update_patch_info(sim)
        self._fill_particles(sim, new_patches)

        for p in new_patches:
            for attr in p.fields.attrs:
                getattr(p.fields, attr).fill(0.0)


    def _shift_right(self, sim: Simulation) -> Sequence[Union[Patch3D, Patch2D]]:
        """Shift simulation window right by one patch.

        Args:
            sim (Simulation): The simulation object

        Returns:
            Sequence[Union[Patch3D, Patch2D]]: List of patches shifted from left to right boundary

        Note:
            - Rightmost patches wrap around to left side
            - Updates patch coordinates and field arrays
        """
        new_patches = []
        for p in sim.patches:
            if p.ipatch_x == 0:
                p.ipatch_x = sim.npatch_x - 1
                p.x0 += sim.Lx
                p.xaxis += sim.Lx

                p.fields.x0 += sim.Lx
                p.fields.xaxis += sim.Lx
                new_patches.append(p)
            else:
                p.ipatch_x -= 1

        return new_patches
                
    def _shift_left(self, sim: Simulation) -> Sequence[Union[Patch3D, Patch2D]]:
        """Shift simulation window left by one patch.

        Args:
            sim (Simulation): The simulation object

        Returns:
            Sequence[Union[Patch3D, Patch2D]]: List of patches shifted from right to left boundary

        Note:
            - Leftmost patches wrap around to right side
            - Updates patch coordinates and field arrays
        """
        new_patches = []
        for p in sim.patches:
            if p.ipatch_x == sim.npatch_x - 1:
                p.ipatch_x = 0
                p.x0 -= sim.Lx
                p.xaxis -= sim.Lx

                p.fields.x0 -= sim.Lx
                p.fields.xaxis -= sim.Lx
                new_patches.append(p)
            else:
                p.ipatch_x += 1
        
        return new_patches
                
    def _update_patch_info(self, sim: Simulation):
        """Update patch neighbor relationships after shift.

        Args:
            sim (Simulation): The simulation object

        Note:
            - Gathers patch information from all MPI ranks
            - Rebuilds neighbor mappings
            - Maintains proper patch connectivity
        """
        # Gather updated patch information from all ranks
        comm = sim.mpi.comm
        rank = comm.Get_rank()
        
        # Prepare local patch info
        local_info = []
        for p in sim.patches:
            local_info.append((p.ipatch_x, p.ipatch_y, p.index, p.rank))
            
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

    def _fill_particles(self, sim: Simulation, new_patches: Sequence[Union[Patch2D, Patch3D]]):
        """Fill particles in newly entered regions of moving window.

        Args:
            sim (Simulation): The simulation object
            new_patches (Sequence[Union[Patch2D, Patch3D]]): Patches that entered the simulation domain

        Note:
            - Handles both 2D and 3D cases
            - Respects particle injection settings
            - Maintains proper density distributions
        """
        if not new_patches:
            return
        
        if not self.inject_particles:
            return
        
        if self.stop_inject_time is not None and sim.time >= self.stop_inject_time:
            return
        
        if sim.dimension == 2:
            self._fill_particles_2d(sim, new_patches)
        elif sim.dimension == 3:
            self._fill_particles_3d(sim, new_patches)
        else:
            raise NotImplementedError("Only 2D and 3D simulations are supported")
        sim.update_lists()

        
    def _fill_particles_2d(self, sim: Simulation, patches: Sequence[Patch2D]):
        """Fill particles in 2D patches for moving window.

        Args:
            sim (Simulation): The 2D simulation object
            patches (Sequence[Patch2D]): 2D patches needing particle initialization

        Note:
            - Uses species density profiles
            - Maintains proper particle weighting
            - Handles MPI communication
        """
        for ispec, s in enumerate(sim.species):
            if s.density is None:
                continue
                
            xaxis = typed.List([p.xaxis for p in patches])
            yaxis = typed.List([p.yaxis for p in patches])
            x_list = typed.List([p.particles[ispec].x for p in patches])
            y_list = typed.List([p.particles[ispec].y for p in patches])
            w_list = typed.List([p.particles[ispec].w for p in patches])

            num_macro_particles = get_num_macro_particles_2d(
                s.density_jit,
                xaxis, 
                yaxis, 
                len(patches), 
                s.density_min, 
                s.ppc,
            )

            for ipatch, p in enumerate(patches):
                p.particles[ispec].initialize(num_macro_particles[ipatch])

                x_list[ipatch] = p.particles[ispec].x
                y_list[ipatch] = p.particles[ispec].y
                w_list[ipatch] = p.particles[ispec].w
            
            fill_particles_2d(
                s.density_jit,
                xaxis, 
                yaxis, 
                len(patches), 
                s.density_min, 
                s.ppc,
                x_list, y_list, w_list
            )

    def _fill_particles_3d(self, sim: Simulation, patches: Sequence[Patch3D]):
        pass