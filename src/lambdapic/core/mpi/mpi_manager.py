from typing import List, Optional, Tuple

import numpy as np
from mpi4py import MPI

from ..patch.patch import Patches


class MPIManager:
    """Handles MPI communication between different Patches instances"""
    
    def __init__(self, patches: Patches, comm: Optional[MPI.Comm]=None):
        """Initialize MPI environment"""
        if comm is None:
            from mpi4py.MPI import COMM_WORLD as comm

        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Dedicated communicators so that messages of different sync kinds can
        # never match each other while overlapping in flight (tags are reused
        # across sync kinds). Guard-field syncs keep the base comm.
        self.comm_particles = self.comm.Dup()
        self.comm_currents = self.comm.Dup()

        self.patches = patches
        self.dimension = patches.dimension
        self.n_guard = patches.n_guard

    @property
    def npatches(self) -> int:
        return self.patches.npatches

    def __getstate__(self):
        # Dup'd communicators cannot be pickled; they are re-created on restore
        state = self.__dict__.copy()
        state.pop('comm_particles', None)
        state.pop('comm_currents', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Comm.Dup is collective; restart load runs it on all ranks together
        self.comm_particles = self.comm.Dup()
        self.comm_currents = self.comm.Dup()

    @staticmethod
    def create(patches: Patches, comm: Optional[MPI.Comm]=None) -> "MPIManager":
        if patches.dimension == 2:
            return MPIManager2D(patches, comm)
        elif patches.dimension == 3:
            return MPIManager3D(patches, comm)
        else:
            raise ValueError(f"Invalid dimension {patches.dimension=}")

    @staticmethod
    def get_default_comm():
        return MPI.COMM_WORLD

    @staticmethod
    def get_default_rank():
        return MPI.COMM_WORLD.Get_rank()
    
    @staticmethod
    def get_defailt_size():
        return MPI.COMM_WORLD.Get_size()

    def sync_particles(self, ispec: int):
        raise NotImplementedError
    
    def sync_guard_fields(self, attrs=['ex', 'ey', 'ez', 'bx', 'by', 'bz']):
        raise NotImplementedError
    
    def sync_currents(self):
        raise NotImplementedError

    def sync_particles_start(self, ispec: int) -> object:
        raise NotImplementedError

    def sync_particles_wait(self, handle: object) -> None:
        raise NotImplementedError

    def sync_guard_fields_start(self, attrs: list[str]) -> object:
        raise NotImplementedError

    def sync_guard_fields_wait(self, handle: object) -> None:
        raise NotImplementedError

    def sync_currents_start(self) -> object:
        raise NotImplementedError

    def sync_currents_wait(self, handle: object) -> None:
        raise NotImplementedError

class MPIManager2D(MPIManager):
    def __init__(self, patches: Patches, comm: Optional[MPI.Comm]=None):
        assert patches.dimension == 2
        super().__init__(patches, comm)
        self.nx_per_patch = patches.nx
        self.ny_per_patch = patches.ny
        self.dx = patches.dx
        self.dy = patches.dy

    def sync_particles(self, ispec: int):
        if self.size == 1:
            return
        h = self.sync_particles_start(ispec)
        self.sync_particles_wait(h)

    def sync_particles_start(self, ispec: int) -> object:
        if self.size == 1:
            return None
        from . import sync_particles_2d
        particles_list = [p.particles[ispec] for p in self.patches]
        patch_list = self.patches.patches

        npart_to_extend, npart_incoming, npart_outgoing = sync_particles_2d.get_npart_to_extend_2d(
            particles_list, 
            patch_list, 
            self.comm_particles,
            self.patches.npatches, 
            self.dx, self.dy,
            ispec, len(self.patches.species)
        )
        for ipatch, p in enumerate(self.patches):
            if npart_to_extend[ipatch] > 0:
                p.particles[ispec].extend(npart_to_extend[ipatch])
                self.patches.update_particle_lists(ipatch)
        
        return sync_particles_2d.fill_particles_from_boundary_2d_start(
            [p.particles[ispec] for p in self.patches], 
            patch_list, 
            npart_incoming, 
            npart_outgoing, 
            self.comm_particles, 
            self.patches.npatches, 
            self.dx, self.dy,
            self.patches.xmin_global, self.patches.xmax_global, self.patches.ymin_global, self.patches.ymax_global,
            self.patches[0].particles[ispec].attrs,
            ispec, len(self.patches.species)
        )

    def sync_particles_wait(self, handle: object) -> None:
        if handle is None:
            return
        from . import sync_particles_2d
        sync_particles_2d.fill_particles_from_boundary_2d_wait(handle)

    def sync_guard_fields(self, attrs=['ex', 'ey', 'ez', 'bx', 'by', 'bz']):
        if self.size == 1:
            return
        h = self.sync_guard_fields_start(attrs)
        self.sync_guard_fields_wait(h)

    def sync_guard_fields_start(self, attrs: list[str]) -> object:
        if self.size == 1:
            return None
        from . import sync_fields2d
        return sync_fields2d.sync_guard_fields_2d_start(
            [p.fields for p in self.patches],
            self.patches.patches,
            self.comm,
            attrs,
            self.npatches, self.nx_per_patch, self.ny_per_patch, self.n_guard
        )

    def sync_guard_fields_wait(self, handle: object) -> None:
        if handle is None:
            return
        from . import sync_fields2d
        sync_fields2d.sync_guard_fields_2d_wait(handle)

    def sync_currents(self):
        if self.size == 1:
            return
        h = self.sync_currents_start()
        self.sync_currents_wait(h)

    def sync_currents_start(self) -> object:
        if self.size == 1:
            return None
        from . import sync_fields2d
        return sync_fields2d.sync_currents_2d_start(
            [p.fields for p in self.patches],
            self.patches.patches,
            self.comm_currents,
            self.npatches, self.nx_per_patch, self.ny_per_patch, self.n_guard,
        )

    def sync_currents_wait(self, handle: object) -> None:
        if handle is None:
            return
        from . import sync_fields2d
        sync_fields2d.sync_currents_2d_wait(handle)

class MPIManager3D(MPIManager):
    def __init__(self, patches: Patches, comm: Optional[MPI.Comm]=None):
        assert patches.dimension == 3
        super().__init__(patches, comm)
        self.nx_per_patch = patches.nx
        self.ny_per_patch = patches.ny
        self.nz_per_patch = patches.nz
        self.dx = patches.dx
        self.dy = patches.dy
        self.dz = patches.dz

    def sync_particles(self, ispec: int):
        if self.size == 1:
            return
        h = self.sync_particles_start(ispec)
        self.sync_particles_wait(h)

    def sync_particles_start(self, ispec: int) -> object:
        if self.size == 1:
            return None
        from . import sync_particles_3d
        particles_list = [p.particles[ispec] for p in self.patches]
        patch_list = self.patches.patches
        
        npart_to_extend, npart_incoming, npart_outgoing = sync_particles_3d.get_npart_to_extend_3d(
            particles_list, 
            patch_list, 
            self.comm_particles,
            self.patches.npatches, 
            self.dx, self.dy, self.dz,
            ispec, len(self.patches.species)
        )
        for ipatch, p in enumerate(self.patches):
            if npart_to_extend[ipatch] > 0:
                p.particles[ispec].extend(npart_to_extend[ipatch])
                self.patches.update_particle_lists(ipatch)
        
        return sync_particles_3d.fill_particles_from_boundary_3d_start(
            [p.particles[ispec] for p in self.patches], 
            patch_list, 
            npart_incoming, 
            npart_outgoing, 
            self.comm_particles, 
            self.patches.npatches, 
            self.dx, self.dy, self.dz,
            self.patches.xmin_global, self.patches.xmax_global, self.patches.ymin_global, self.patches.ymax_global, self.patches.zmin_global, self.patches.zmax_global,
            self.patches[0].particles[ispec].attrs,
            ispec, len(self.patches.species)
        )

    def sync_particles_wait(self, handle: object) -> None:
        if handle is None:
            return
        from . import sync_particles_3d
        sync_particles_3d.fill_particles_from_boundary_3d_wait(handle)

    def sync_guard_fields(self, attrs=['ex', 'ey', 'ez', 'bx', 'by', 'bz']):
        if self.size == 1:
            return
        h = self.sync_guard_fields_start(attrs)
        self.sync_guard_fields_wait(h)

    def sync_guard_fields_start(self, attrs: list[str]) -> object:
        if self.size == 1:
            return None
        from . import sync_fields3d
        return sync_fields3d.sync_guard_fields_3d_start(
            [p.fields for p in self.patches],
            self.patches.patches,
            self.comm,
            attrs,
            self.npatches, self.nx_per_patch, self.ny_per_patch, self.nz_per_patch, self.n_guard,
        )

    def sync_guard_fields_wait(self, handle: object) -> None:
        if handle is None:
            return
        from . import sync_fields3d
        sync_fields3d.sync_guard_fields_3d_wait(handle)

    def sync_currents(self):
        if self.size == 1:
            return
        h = self.sync_currents_start()
        self.sync_currents_wait(h)

    def sync_currents_start(self) -> object:
        if self.size == 1:
            return None
        from . import sync_fields3d
        return sync_fields3d.sync_currents_3d_start(
            [p.fields for p in self.patches],
            self.patches.patches,
            self.comm_currents,
            self.npatches, self.nx_per_patch, self.ny_per_patch, self.nz_per_patch, self.n_guard,
        )

    def sync_currents_wait(self, handle: object) -> None:
        if handle is None:
            return
        from . import sync_fields3d
        sync_fields3d.sync_currents_3d_wait(handle)
