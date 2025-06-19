import numpy as np
from mpi4py import MPI
from numba import prange
from typing import List, Tuple

from ..particles import ParticlesBase
from ..patch.patch import Patch2D, Patches, Boundary2D
from ..fields import Fields2D

from . import sync_particles_2d, sync_fields2d, sync_fields3d, sync_particles_3d

class MPIManager:
    """Handles MPI communication between different Patches instances"""
    
    def __init__(self, patches: Patches):
        """Initialize MPI environment"""
        from mpi4py.MPI import COMM_WORLD as comm

        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.patches = patches
        self.dimension = patches.dimension
        self.npatches = patches.npatches

        self.n_guard = patches.n_guard

    @classmethod
    def create(cls, patches: Patches) -> "MPIManager":
        if patches.dimension == 2:
            return MPIManager2D(patches)
        elif patches.dimension == 3:
            return MPIManager3D(patches)
        else:
            raise ValueError(f"Invalid dimension {patches.dimension=}")

    @classmethod
    def get_comm(cls):
        return MPI.COMM_WORLD

    @classmethod
    def get_rank(cls):
        return MPI.COMM_WORLD.Get_rank()
    
    @classmethod
    def get_size(cls):
        return MPI.COMM_WORLD.Get_size()

    def sync_particles(self, ispec: int):
        raise NotImplementedError
    
    def sync_guard_fields(self, attrs=['ex', 'ey', 'ez', 'bx', 'by', 'bz']):
        raise NotImplementedError
    
    def sync_currents(self):
        raise NotImplementedError

class MPIManager2D(MPIManager):
    def __init__(self, patches: Patches):
        assert patches.dimension == 2
        super().__init__(patches)
        self.nx_per_patch = patches.nx
        self.ny_per_patch = patches.ny
        self.dx = patches.dx
        self.dy = patches.dy

    def sync_particles(self, ispec: int):
        if self.size == 1:
            return
        particles_list = [p.particles[ispec] for p in self.patches]
        patch_list = self.patches.patches

        npart_to_extend, npart_incoming, npart_outgoing = sync_particles_2d.get_npart_to_extend_2d(
            particles_list, 
            patch_list, 
            self.comm,
            self.patches.npatches, 
            self.dx, self.dy
        )
        for ipatch, p in enumerate(self.patches):
            p.particles[ispec].extend(npart_to_extend[ipatch])
            p.particles[ispec].extended = True
            self.patches.update_particle_lists(ipatch)
        
        sync_particles_2d.fill_particles_from_boundary_2d(
            [p.particles[ispec] for p in self.patches], 
            patch_list, 
            npart_incoming, 
            npart_outgoing, 
            self.comm, 
            self.patches.npatches, 
            self.dx, self.dy,
            self.patches[0].particles[ispec].attrs
        )

    def sync_guard_fields(self, attrs=['ex', 'ey', 'ez', 'bx', 'by', 'bz']):
        if self.size == 1:
            return
        sync_fields2d.sync_guard_fields_2d(
            [p.fields for p in self.patches],
            self.patches.patches,
            self.comm,
            attrs,
            self.npatches, self.nx_per_patch, self.ny_per_patch, self.n_guard
        )

    def sync_currents(self):
        if self.size == 1:
            return
        sync_fields2d.sync_currents_2d(
            [p.fields for p in self.patches],
            self.patches.patches,
            self.comm,
            self.npatches, self.nx_per_patch, self.ny_per_patch, self.n_guard,
        )

class MPIManager3D(MPIManager):
    def __init__(self, patches: Patches):
        assert patches.dimension == 3
        super().__init__(patches)
        self.nx_per_patch = patches.nx
        self.ny_per_patch = patches.ny
        self.nz_per_patch = patches.nz
        self.dx = patches.dx
        self.dy = patches.dy
        self.dz = patches.dz

    def sync_particles(self, ispec: int):
        if self.size == 1:
            return
        particles_list = [p.particles[ispec] for p in self.patches]
        patch_list = self.patches.patches
        
        npart_to_extend, npart_incoming, npart_outgoing = sync_particles_3d.get_npart_to_extend_3d(
            particles_list, 
            patch_list, 
            self.comm,
            self.patches.npatches, 
            self.dx, self.dy, self.dz
        )
        for ipatch, p in enumerate(self.patches):
            p.particles[ispec].extend(npart_to_extend[ipatch])
            p.particles[ispec].extended = True
            self.patches.update_particle_lists(ipatch)
        
        sync_particles_3d.fill_particles_from_boundary_3d(
            [p.particles[ispec] for p in self.patches], 
            patch_list, 
            npart_incoming, 
            npart_outgoing, 
            self.comm, 
            self.patches.npatches, 
            self.dx, self.dy, self.dz,
            self.patches[0].particles[ispec].attrs
        )

    def sync_guard_fields(self, attrs=['ex', 'ey', 'ez', 'bx', 'by', 'bz']):
        if self.size == 1:
            return
       
        sync_fields3d.sync_guard_fields_3d(
            [p.fields for p in self.patches],
            self.patches.patches,
            self.comm,
            attrs,
            self.npatches, self.nx_per_patch, self.ny_per_patch, self.nz_per_patch, self.n_guard,
        )

    def sync_currents(self):
        if self.size == 1:
            return
            
        sync_fields3d.sync_currents_3d(
            [p.fields for p in self.patches],
            self.patches.patches,
            self.comm,
            self.npatches, self.nx_per_patch, self.ny_per_patch, self.nz_per_patch, self.n_guard,
        )