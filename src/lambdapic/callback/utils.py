from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import h5py
import numpy as np
from numba import typed
from numpy.typing import NDArray
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi

from lambdapic.callback.hdf5 import SaveSpeciesDensityToHDF5
from lambdapic.core.boundary.cpml import PMLX

from ..core.patch.cpu import (
    fill_particles_2d,
    fill_particles_3d,
    get_num_macro_particles_2d,
    get_num_macro_particles_3d,
)
from ..core.patch.patch import Patch, Patch2D, Patch3D
from ..core.species import Species
from ..core.utils.logger import logger
from ..simulation import Simulation, Simulation3D
from .callback import Callback


def get_fields(sim: Simulation|Simulation3D, fields: Sequence[str], slice_at: Optional[float] = None) -> Sequence[np.ndarray]:
    """
    Get fields from all patches.
    If 3D simulation, fields are sliced at `slice_at` (default: Lz/2).

    Only rank 0 will gather the data, other ranks will get None.
    
    Parameters:
        sim (Simulation): Simulation instance.
        fields (Sequence[str]): List of fields to get.
        slice_at (Optional[float]): z position to slice at. defaults to Lz/2.
        
    Returns:
        Sequence[np.ndarray]: List of fields named as field. 2-dimensional array.
    """

    if isinstance(sim, Simulation3D):
        return get_fields_3d(sim, fields, slice_at)
    elif isinstance(sim, Simulation):
        return get_fields_2d(sim, fields)
    else:
        raise ValueError(f"Unsupported simulation type: {type(sim)}")


def get_fields_2d(sim: Simulation, fields: Sequence[str]) -> Sequence[np.ndarray]:
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


def get_fields_3d(sim: Simulation3D, fields: Sequence[str], slice_at: Optional[float] = None) -> Sequence[np.ndarray]:
    """
    Get 3D fields from all patches at a specific z-slice.
    
    Only rank 0 will gather the data, other ranks will get None.
    This function first extracts the z-slice from each patch, then communicates
    the sliced data to rank 0 for assembly.
    
    Args:
        sim (Simulation3D): 3D Simulation instance.
        fields (Sequence[str]): List of fields to get.
        slice_at (Optional[float]): Z position to slice at. If None, defaults to Lz/2.
        
    Returns:
        Sequence[np.ndarray]: List of 2D fields (nx, ny) representing the z-slice.
    """
    ret = []
    patches = sim.patches
    nx_per_patch = sim.nx_per_patch
    ny_per_patch = sim.ny_per_patch
    nz_per_patch = sim.nz_per_patch
    npatch_x = sim.npatch_x
    npatch_y = sim.npatch_y
    npatch_z = sim.npatch_z
    nx = sim.nx
    ny = sim.ny
    nz = sim.nz
    ng = sim.n_guard
    
    assert sim.dimension == 3, "Only 3D simulation is supported"
    
    if not fields:
        return ret
    
    # Set default slice position to Lz/2 if not provided
    if slice_at is None:
        slice_at = sim.Lz / 2
    
    if slice_at < 0 or slice_at > sim.Lz:
        raise ValueError(f"Slice position {slice_at} is outside the simulation domain [0, {sim.Lz}]")
    
    # Create mapping from patch coordinates to patch index, only for patches containing the slice
    local_index_info = {}
    for ipatch, p in enumerate(patches):
        # Check if this patch contains the slice_at position
        if p.zmin <= slice_at <= p.zmax:
            local_index_info[(p.ipatch_x, p.ipatch_y, p.ipatch_z)] = p.index
    
    index_info = sim.mpi.comm.gather(local_index_info)
    
    # Calculate the z-index for the slice
    z_global = slice_at + sim.dz / 2  # Convert position to grid index
    iz_global = int(z_global / sim.dz)

    iz_local = iz_global - iz_global // nz_per_patch * nz_per_patch
    
    if sim.mpi.rank == 0:
        patch_index_map = {k: v for d in index_info for k, v in d.items()} if index_info else {}
        local_patches = {p.index: ipatch for ipatch, p in enumerate(patches)}
    
    for field in fields:
        if sim.mpi.rank == 0:
            field_ = np.zeros((nx, ny))
            buf = np.zeros((nx_per_patch + 2*ng, ny_per_patch + 2*ng))
            
            # Only process patches that contain the z-slice
            for ipatch_x in range(npatch_x):
                for ipatch_y in range(npatch_y):
                    for ipatch_z in range(npatch_z):
                        # Check if this patch contains the slice
                        if (ipatch_x, ipatch_y, ipatch_z) in patch_index_map:
                            s = np.s_[ipatch_x*nx_per_patch:ipatch_x*nx_per_patch+nx_per_patch,
                                      ipatch_y*ny_per_patch:ipatch_y*ny_per_patch+ny_per_patch]
                            
                            index = patch_index_map[(ipatch_x, ipatch_y, ipatch_z)]
                            if index in local_patches:
                                # Local patch: extract the slice directly
                                p = patches[local_patches[index]]
                                field_data = getattr(p.fields, field)
                                # Extract z-slice, excluding guard cells
                                field_[s] = field_data[:-2*ng, :-2*ng, iz_local]
                            else:
                                # Remote patch: receive the slice data
                                sim.mpi.comm.Recv(buf, tag=index)
                                field_[s] = buf[:-2*ng, :-2*ng]
            
            ret.append(field_)
        else:
            # Other ranks: send slice data for patches they own
            req = []
            for p in patches:
                # Only send data if this patch contains the z-slice
                if p.zmin <= slice_at <= p.zmax:
                    # Calculate local z-index within this patch
                    iz_local = iz_global - p.ipatch_z * nz_per_patch + ng  # Add guard cells
                    
                    # Extract the z-slice from the 3D field data
                    field_data = getattr(p.fields, field)
                    slice_data = field_data[:, :, iz_local].copy()
                    
                    req_ = sim.mpi.comm.Isend(slice_data, dest=0, tag=p.index)
                    req.append(req_)
            
            # Wait for all send operations to complete
            for req_ in req:
                req_.wait()
            
            ret.append(None)
        
        # Synchronize all processes
        sim.mpi.comm.Barrier()
    
    return ret


class ExtractSpeciesDensity(SaveSpeciesDensityToHDF5):
    """Callback to extract species density from all patches.
    
    Only rank 0 will gather the data, other ranks will get zeros.
    
    Args:
        sim (Simulation): Simulation instance.
        species (Species): Species instance to extract density from.
        interval (Union[int, float, Callable], optional): Number of timesteps between saves, or a function(sim) -> bool that determines when to save. Defaults to 100.

    Example:

        >>> ne_ele = ExtractSpeciesDensity(sim, ele, interval=100)
        use in PlotFields:
        >>> sim.run(1000, callbacks[
                ne_ele,
                PlotFields(
                    [dict(field=ne_ele.density, scale=1/nc, cmap='Grays', vmin=0, vmax=20), 
                    dict(field='ey',  scale=e/(m_e*c*omega0), cmap='bwr_alpha', vmin=-laser.a0, vmax=laser.a0) ],
                    prefix='laser-target'),
            ])
    """

    stage = "current deposition"
    def __init__(self, sim: Simulation, species: Species, interval: Union[int, float, Callable] = 100) -> None:
        self.species = species
        self.interval = interval
        self.prev_rho = None
        self.species = species
        if sim.dimension == 2:
            self.density: NDArray[np.float64] = np.zeros((sim.nx, sim.ny))
        else:
            self.density: NDArray[np.float64] = np.zeros((sim.nx, sim.ny, sim.nz))
        

        self.prefix = Path('') # for compatibility

    def _write_2d(self, sim, density_per_patch, filename):
        comm = sim.mpi.comm
        rank = sim.mpi.rank
        
        nx_per_patch = sim.nx_per_patch
        ny_per_patch = sim.ny_per_patch
        ng = sim.n_guard
        npatch_x = sim.npatch_x
        npatch_y = sim.npatch_y

        index_info = {(p.ipatch_x, p.ipatch_y): p.index for ipatch, p in enumerate(sim.patches)}
        index_info: list[dict[tuple[int, int], int]] = sim.mpi.comm.gather(index_info)
        if rank == 0:
            patch_index_map = {k: v for d in index_info for k, v in d.items()}
            local_patches = {p.index: ipatch for ipatch, p in enumerate(sim.patches)}
            
        if rank == 0:
            # local
            for ip, p in enumerate(sim.patches):
                s = np.s_[p.ipatch_x*nx_per_patch:p.ipatch_x*nx_per_patch+nx_per_patch,\
                          p.ipatch_y*ny_per_patch:p.ipatch_y*ny_per_patch+ny_per_patch]
                self.density[s] = density_per_patch[ip][:-2*ng, :-2*ng]

            buf = np.zeros((nx_per_patch+2*ng, ny_per_patch+2*ng))
            for ipatch_x in range(npatch_x):
                for ipatch_y in range(npatch_y):
                    s = np.s_[ipatch_x*nx_per_patch:ipatch_x*nx_per_patch+nx_per_patch,\
                              ipatch_y*ny_per_patch:ipatch_y*ny_per_patch+ny_per_patch]
                    # local
                    index = patch_index_map[(ipatch_x, ipatch_y)]
                    if index in local_patches:
                        continue
                    #remote
                    else:
                        sim.mpi.comm.Recv(buf, tag=index)
                        self.density[s] = buf[:-2*ng, :-2*ng]
        
        else:
            req = []
            for ip, p in enumerate(sim.patches):
                req_ = sim.mpi.comm.Isend(density_per_patch[ip], dest=0, tag=p.index)
                req.append(req_)
            for req_ in req:
                req_.wait()

        comm.Barrier()

    def _write_3d(self, sim, density_per_patch, filename):
        comm = sim.mpi.comm
        rank = sim.mpi.rank
        
        nx_per_patch = sim.nx_per_patch
        ny_per_patch = sim.ny_per_patch
        nz_per_patch = sim.nz_per_patch
        ng = sim.n_guard
        npatch_x = sim.npatch_x
        npatch_y = sim.npatch_y
        npatch_z = sim.npatch_z

        index_info = {(p.ipatch_x, p.ipatch_y, p.ipatch_z): p.index for ipatch, p in enumerate(sim.patches)}
        index_info: list[dict[tuple[int, int, int], int]] = sim.mpi.comm.gather(index_info)
        if rank == 0:
            patch_index_map = {k: v for d in index_info for k, v in d.items()}
            local_patches = {p.index: ipatch for ipatch, p in enumerate(sim.patches)}
            
        if rank == 0:
            # local
            for ip, p in enumerate(sim.patches):
                s = np.s_[p.ipatch_x*nx_per_patch:p.ipatch_x*nx_per_patch+nx_per_patch,\
                          p.ipatch_y*ny_per_patch:p.ipatch_y*ny_per_patch+ny_per_patch,\
                          p.ipatch_z*nz_per_patch:p.ipatch_z*nz_per_patch+nz_per_patch]
                self.density[s] = density_per_patch[ip][:-2*ng, :-2*ng, :-2*ng]

            buf = np.zeros((nx_per_patch+2*ng, ny_per_patch+2*ng, nz_per_patch+2*ng))
            for ipatch_x in range(npatch_x):
                for ipatch_y in range(npatch_y):
                    for ipatch_z in range(npatch_z):
                        s = np.s_[ipatch_x*nx_per_patch:ipatch_x*nx_per_patch+nx_per_patch,\
                                  ipatch_y*ny_per_patch:ipatch_y*ny_per_patch+ny_per_patch,\
                                  ipatch_z*nz_per_patch:ipatch_z*nz_per_patch+nz_per_patch]
                        # local
                        index = patch_index_map[(ipatch_x, ipatch_y, ipatch_z)]
                        if index in local_patches:
                            continue
                        #remote
                        else:
                            sim.mpi.comm.Recv(buf, tag=index)
                            self.density[s] = buf[:-2*ng, :-2*ng, :-2*ng]
        
        else:
            req = []
            for ip, p in enumerate(sim.patches):
                req_ = sim.mpi.comm.Isend(density_per_patch[ip], dest=0, tag=p.index)
                req.append(req_)
            for req_ in req:
                req_.wait()

        comm.Barrier()

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
        start_time: Optional[float] = None,
        inject_particles: bool = True,
        stop_inject_time: Optional[float] = None,
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

        for sorter in sim.sorter:
            sorter.generate_field_lists()
            sorter.generate_particle_lists()


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
                
    def _update_patch_info(self, sim: Simulation|Simulation3D):
        """Update patch neighbor relationships after shift.

        Args:
            sim (Simulation): The simulation object

        Note:
            - Gathers patch information from all MPI ranks
            - Rebuilds neighbor mappings
            - Maintains proper patch connectivity
        """
        if isinstance(sim, Simulation3D):
            self._update_patch_info_3d(sim)
        else:
            self._update_patch_info_2d(sim)
                
    def _update_patch_info_2d(self, sim: Simulation):
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
        sim.patches.init_rect_neighbor_index_2d(sim.npatch_x, sim.npatch_y, boundary_conditions=sim.boundary_conditions, patch_index_map=patch_index_map)
        sim.patches.init_neighbor_ipatch_2d()
        sim.patches.init_neighbor_rank_2d(patch_rank_map)
                
    def _update_patch_info_3d(self, sim: Simulation3D):
        # Gather updated patch information from all ranks
        comm = sim.mpi.comm
        rank = comm.Get_rank()
        
        # Prepare local patch info
        local_info = []
        for p in sim.patches:
            local_info.append((p.ipatch_x, p.ipatch_y, p.ipatch_z, p.index, p.rank))
            
        # Gather all patch info
        all_info = comm.allgather(local_info)
        
        # Build global mappings
        patch_index_map = {}
        patch_rank_map = {}
        for rank_info in all_info:
            for (ipx, ipy, ipz, idx, r) in rank_info:
                patch_index_map[(ipx, ipy, ipz)] = idx
                patch_rank_map[idx] = r
                
        # Reinitialize neighbor relationships
        sim.patches.init_rect_neighbor_index_3d(sim.npatch_x, sim.npatch_y, sim.npatch_z, boundary_conditions=sim.boundary_conditions, patch_index_map=patch_index_map)
        sim.patches.init_neighbor_ipatch_3d()
        sim.patches.init_neighbor_rank_3d(patch_rank_map)

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
            gens = typed.List(sim.rand_gen.spawn(len(patches)))

            num_macro_particles = get_num_macro_particles_2d(
                s.density_jit,
                xaxis, 
                yaxis, 
                len(patches), 
                s.density_min, 
                s.ppc_jit,
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
                s.ppc_jit,
                x_list, y_list, w_list,
                gens
            )

    def _fill_particles_3d(self, sim: Simulation, patches: Sequence[Patch3D]):
        for ispec, s in enumerate(sim.species):
            if s.density is None:
                continue
                
            xaxis = typed.List([p.xaxis for p in patches])
            yaxis = typed.List([p.yaxis for p in patches])
            zaxis = typed.List([p.zaxis for p in patches])
            x_list = typed.List([p.particles[ispec].x for p in patches])
            y_list = typed.List([p.particles[ispec].y for p in patches])
            z_list = typed.List([p.particles[ispec].z for p in patches])
            w_list = typed.List([p.particles[ispec].w for p in patches])
            gens = typed.List(sim.rand_gen.spawn(len(patches)))

            num_macro_particles = get_num_macro_particles_3d(
                s.density_jit,
                xaxis,
                yaxis,
                zaxis,
                len(patches), 
                s.density_min, 
                s.ppc_jit,
            )

            for ipatch, p in enumerate(patches):
                p.particles[ispec].initialize(num_macro_particles[ipatch])

                x_list[ipatch] = p.particles[ispec].x
                y_list[ipatch] = p.particles[ispec].y
                z_list[ipatch] = p.particles[ispec].z
                w_list[ipatch] = p.particles[ispec].w
            
            fill_particles_3d(
                s.density_jit,
                xaxis, 
                yaxis,
                zaxis,
                len(patches), 
                s.density_min, 
                s.ppc_jit,
                x_list, y_list, z_list, w_list,
                gens
            )

class SetTemperature(Callback):
    """
    Callback to set the particle momenta (ux, uy, uz) for a species to a relativistic Maxwell-Jüttner distribution
    with the specified temperature (in units of eV).

    Args:
        species (Species): The target species whose temperature is to be set.
        temperature (float): Temperature in units of eV.
        interval (int or callable): Frequency (in timesteps) or callable(sim) for when to apply, defaults to run at the first timestep only once.
    """
    stage: str = "start"
    def __init__(self, species: Species, temperature: float|int|List[float|int], interval: Union[int, float, Callable]|None = None) -> None:
        self.species = species

        if isinstance(temperature, (float, int)):
            self.temperature = [temperature] * 3
        else:
            self.temperature = temperature

        if interval is None:
            self.interval = lambda sim: sim.itime == 0
        else:
            self.interval = interval

    def _call(self, sim: Simulation) -> None:
        ispec = self.species.ispec
        for p in sim.patches:
            part = p.particles[ispec]
            alive = part.is_alive
            n: int = alive.sum()
            if n == 0:
                continue
            
            thetax = self.temperature[0]*e / (self.species.m * c**2)
            ux, uy, uz = self.sample_maxwell_juttner(n, thetax)
            # stretch to simulate temperature anisotropy
            part.ux[alive] = ux
            part.uy[alive] = uy * self.temperature[1]/self.temperature[0]
            part.uz[alive] = uz * self.temperature[2]/self.temperature[0]
            part.inv_gamma[alive] = 1 / np.sqrt(1 + part.ux[alive]**2 + part.uy[alive]**2 + part.uz[alive]**2)

    @staticmethod
    def maxwell_juttner_pdf(gamma: np.ndarray[float], theta: float) -> np.ndarray[float]:
        """
        Probability density function of Maxwell-Jüttner distribution
        """
        from scipy.special import kn
        beta = np.sqrt(1 - 1/(gamma**2))
        x = 1/theta

        # valid for x < 100
        k2 = kn(2, x)
        
        return (gamma**2 * beta) / (theta * k2) * np.exp(-gamma/theta)
    
    @staticmethod
    def sample_maxwell_juttner(size: int, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate samples from Maxwell-Jüttner distribution
        Using a mixed strategy to adapt to different temperature ranges:
        - θ ≤ 0.01: Non-relativistic approximation (Gamma distribution)
        - 0.01 < θ ≤ 0.5: Uniform distribution + rejection sampling
        - θ > 0.5: Gamma proposal distribution + β acceptance probability
        """
        import scipy
        
        gamma = np.zeros(size)
        # Non-relativistic approximation (θ ≤ 0.01)
        if theta <= 0.01:
            gamma[:] = scipy.stats.gamma(a=1.5, scale=theta).rvs(size=size) + 1
        
        # Medium temperature range (0.01 < θ ≤ 0.5)
        elif theta <= 0.5:
            gamma_max = 1 + 10 * theta
            # Find PDF maximum
            res = scipy.optimize.minimize_scalar(
                lambda g: -SetTemperature.maxwell_juttner_pdf(g, theta),
                bounds=(1, gamma_max),
                method='bounded'
            )
            f_max = -res.fun
            M = f_max * 1.1 + 1e-10  # Safety factor
            
            count: int = 0
            while count < size:
                gamma_prop = np.random.uniform(1, gamma_max, size-count)
                f_val = SetTemperature.maxwell_juttner_pdf(gamma_prop, theta)
                accept = np.random.uniform(0, M, size-count) < f_val
                gamma_accept = gamma_prop[accept]
                n_accept = len(gamma_accept)
                
                gamma[count:count+n_accept] = gamma_accept
                count += n_accept
        
        # High temperature range (θ > 0.5)
        else:
            gamma_dist = scipy.stats.gamma(a=3, scale=theta)
            count: int = 0
            while count < size:
                gamma_prop = gamma_dist.rvs(size-count)
                beta_val = beta_val = np.sqrt(1 - 1/(np.ma.array(gamma_prop, mask=gamma_prop < 1)**2))
                accept = (np.random.uniform(size=size-count) < beta_val) & (gamma_prop >= 1)
                gamma_accept = gamma_prop[accept]
                n_accept = len(gamma_accept)
                
                gamma[count:count+n_accept] = gamma_accept
                count += n_accept
        
        u = np.sqrt(gamma**2 - 1)
        phi = np.random.uniform(0, 2*np.pi, size)
        costheta = np.random.uniform(-1, 1, size)
        sintheta = np.sqrt(1 - costheta**2)
        ux = u * sintheta * np.cos(phi)
        uy = u * sintheta * np.sin(phi)
        uz = u * costheta
        
        return  ux, uy, uz
    
class LoadParticles(Callback):
    """
    Callback to load particles from HDF5 files with batch processing support.
    
    The file should at least contains:

    - '/x'
    - '/y'
    - '/z', for 3d
    - '/w'

    Other attributes supported by the target particle class will be loaded if present.

    You can load from a file generated by :any:`SaveParticlesToHDF5`.
    
    Parameters:
        species (Species): The target species to load particles into.
        file (str|Path): Path to the HDF5 file containing particle data.
        interval (Union[int, float, Callable], optional): Frequency of execution.
            Can be a number of timesteps, a time interval, or a callable function
            that returns True when the callback should execute. Defaults to running
            only at the first timestep.
            
    Attributes:
        _batch_size (int): Number of particles to process in each batch.
            Controls memory usage during loading. Default is 10,000 particles per batch.
            
    Example:
        >>> # Load electrons from an HDF5 file at simulation start
        >>> load_ele = LoadParticles(ele, "particles.h5")
        >>> sim.run(1000, callbacks=[load_ele])
        
    Note:
        - The HDF5 file should contain datasets for all particle attributes (x, y, z, ux, uy, uz, w, etc.)
        - Particles are automatically distributed to patches based on their spatial coordinates
        - Batch processing reduces memory usage for large particle datasets
    """
    stage: str = "start"
    def __init__(self, species: Species, file: str|Path, interval: Union[int, float, Callable]|None = None) -> None:
        
        self.species = species
        self.file = file

        self._batch_size = 10000  # Default batch size for memory-efficient loading

        if interval is None:
            self.interval = lambda sim: sim.itime == 0
        else:
            self.interval = interval

        
    def _load_from_file(self, sim: Simulation) -> None:
        attrs = self._filter_attributes(sim)
        with h5py.File(self.file, 'r', locking=False) as f:
            total_particles = f['x'].shape[0]
            
            # Process data in batches to reduce memory usage
            for start_idx in range(0, total_particles, self._batch_size):
                end_idx = min(start_idx + self._batch_size, total_particles)
                
                # Read current batch of data
                data = {}
                for attr in attrs:
                    data[attr] = f[attr][start_idx:end_idx]
                
                # Extract coordinate data for spatial filtering
                x = data.get('x', None)
                y = data.get('y', None)
                z = data.get('z', None)
                
                # Process particle distribution for each patch
                for ip, p in enumerate(sim.patches):
                    inbound = (x >= p.xmin - p.dx/2) & (x < p.xmax + p.dx/2)
                    
                    if sim.dimension >= 2:
                        inbound &= (y >= p.ymin - p.dy/2) & (y < p.ymax + p.dy/2)
                    
                    if sim.dimension == 3:
                        inbound &= (z >= p.zmin - p.dz/2) & (z < p.zmax + p.dz/2)
                    
                    npart = np.sum(inbound)
                    if npart == 0:
                        continue
                        
                    # Extend particle arrays and assign data
                    part = p.particles[self.species.ispec]
                    part.extend(npart)
                    
                    # Assign particle attributes for this batch
                    for attr in attrs:
                        attr_data = data[attr][inbound]
                        getattr(part, attr)[-npart:] = attr_data

                    # update inv_gamma
                    if 'ux' in attrs or 'uy' in attrs or 'uz' in attrs:
                        part.inv_gamma[-npart:] = np.sqrt(1 + part.ux[-npart:]**2 + part.uy[-npart:]**2 + part.uz[-npart:]**2)

                    # mark particles as alive
                    part.is_dead[-npart:] = False
        sim.update_lists()

    def _filter_attributes(self, sim: Simulation, ) -> set[str]:
        with h5py.File(self.file, 'r', locking=False) as f:
            attrs_file: set[str] = set(f.keys())

        if sim.dimension >= 1 and 'x' not in attrs_file:
            raise ValueError(f"Attribute 'x' not found in {self.file}")
        if sim.dimension >= 2 and 'y' not in attrs_file:
            raise ValueError(f"Attribute 'y' not found in {self.file}")
        if sim.dimension == 3 and 'z' not in attrs_file:
            raise ValueError(f"Attribute 'z' not found in {self.file}")
        if 'w' not in attrs_file:
            raise ValueError(f"Attribute 'w' not found in {self.file}")

        attrs = attrs_file.intersection(sim.patches[0].particles[0].attrs)
        if '_id' in attrs:
            attrs.remove('_id')
        if 'inv_gamma' in attrs:
            attrs.remove('inv_gamma')

        attrs_discard = attrs_file.difference(attrs)
        if len(attrs_discard) > 0:
            logger.warning(f"Attributes {attrs_discard} in {self.file} discarded: Not attribute of particle {self.species.name} or cannot be manually set")

        return attrs
    
    def _call(self, sim: Simulation) -> None:
        self._load_from_file(sim)