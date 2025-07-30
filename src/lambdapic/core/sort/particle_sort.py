from typing import List
import numpy as np

from lambdapic.core.species import Species
from ..patch import Patches

from .cpu2d import sort_particles_patches_2d
from .cpu3d import sort_particles_patches_3d

class ParticleSort2D:
    """
    sort after particle sync: no particles in guard cells
    """
    def __init__(
            self, 
            patches: Patches, 
            species: Species,
            nx_buckets: int|None=None,
            ny_buckets: int|None=None,
            dx_buckets: float|None=None,
            dy_buckets: float|None=None,
            attrs: List[str]|None=None,
        ) -> None:
        """
        Construct from patches.

        Parameters
        ----------
        patches : Patches
            Patches to be sorted.
        ispec : int
            Species index.
        """
        self.dimension = patches.dimension
        self.patches = patches
        self.npatches: int = patches.npatches
        self.ispec = species.ispec

        self.attrs = attrs or patches[0].particles[self.ispec].attrs
        self.nattrs = len(self.attrs)
        
        self.dx_buckets = dx_buckets or patches.dx
        self.dy_buckets = dy_buckets or patches.dy
        self.nx_buckets = nx_buckets or patches.nx
        self.ny_buckets = ny_buckets or patches.ny

        self.generate_particle_lists()
        self.generate_field_lists()

    def generate_particle_lists(self) -> None:
        """
        Generate typed.List of particle data of all species in all patches.

        Parameters
        ----------
        particle_list : list of Particles
            List of particles of all patches. 
        """

        ispec = self.ispec
        
        self.x_list = [p.particles[ispec].x for p in self.patches]
        if self.dimension >= 2:
            self.y_list = [p.particles[ispec].y for p in self.patches]
        if self.dimension == 3:
            self.z_list = [p.particles[ispec].z for p in self.patches]

        self.attrs_list = [getattr(p.particles[ispec], attr) for p in self.patches for attr in self.attrs ]

        self.is_dead_list = [p.particles[ispec].is_dead for p in self.patches]
        
        self.particle_index_list = [np.full(p.particles[ispec].is_dead.size, -1, dtype=int) for p in self.patches]
        self.particle_index_ref_list = [np.full(p.particles[ispec].is_dead.size, -1, dtype=int) for p in self.patches]
        self.particle_index_target_list = [np.full(p.particles[ispec].is_dead.size, -1, dtype=int) for p in self.patches]
        self.buf_list = [np.full(p.particles[ispec].is_dead.size, 0, dtype=float) for p in self.patches]

    def update_particle_lists(self, ipatch: int) -> None:
        """
        Update particle lists of a species in a patch.

        Parameters
        ----------
        ipatch : int
            Patch index.
        ispec : int
            Species index.
        """

        particles = self.patches[ipatch].particles[self.ispec]

        self.x_list[ipatch] = particles.x
        if self.dimension >= 2:
            self.y_list[ipatch] = particles.y
        if self.dimension == 3:
            self.z_list[ipatch] = particles.z

        for iattr, attr in enumerate(self.attrs):
            self.attrs_list[iattr+self.nattrs*ipatch] = getattr(particles, attr)
                
        self.is_dead_list[ipatch] = particles.is_dead
        
        self.particle_index_list[ipatch] = np.full(particles.is_dead.size, -1, dtype=int)
        self.particle_index_ref_list[ipatch] = np.full(particles.is_dead.size, -1, dtype=int)
        self.particle_index_target_list[ipatch] = np.full(particles.is_dead.size, -1, dtype=int)
        self.buf_list[ipatch] = np.full(particles.is_dead.size, 0, dtype=float)

    def _update_particle_lists(self) -> None:
        """
        Update particle lists of a species in a patch.

        Parameters
        ----------
        ipatch : int
            Patch index.
        ispec : int
            Species index.
        """

        ispec = self.ispec
        
        self.x_list = [p.particles[ispec].x for p in self.patches]
        if self.dimension >= 2:
            self.y_list = [p.particles[ispec].y for p in self.patches]
        if self.dimension == 3:
            self.z_list = [p.particles[ispec].z for p in self.patches]

        self.attrs_list = [getattr(p.particles[ispec], attr) for p in self.patches for attr in self.attrs ]

        self.is_dead_list = [p.particles[ispec].is_dead for p in self.patches]
        
        for ipatch, p in enumerate(self.patches):
            if self.particle_index_list[ipatch].size < p.particles[ispec].npart:
                self.particle_index_list[ipatch].resize(p.particles[ispec].npart, refcheck=False)
                self.particle_index_ref_list[ipatch].resize(p.particles[ispec].npart, refcheck=False)
                self.particle_index_target_list[ipatch].resize(p.particles[ispec].npart, refcheck=False)
                self.buf_list[ipatch].resize(p.particles[ispec].npart, refcheck=False)
        
    def generate_field_lists(self) -> None:
        """
        Update field lists of all patches.

        Parameters
        ----------
        fields : list of Fields2D
            List of fields of all patches.
        """
        self.bucket_count_list = [np.full((self.nx_buckets, self.ny_buckets), 0, dtype=int) for _ in range(self.npatches)]
        self.bucket_bound_min_list = [np.full((self.nx_buckets, self.ny_buckets), 0, dtype=int) for _ in range(self.npatches)]
        self.bucket_bound_max_list = [np.full((self.nx_buckets, self.ny_buckets), 0, dtype=int) for _ in range(self.npatches)]
        self.bucket_count_not_list = [np.full((self.nx_buckets, self.ny_buckets), 0, dtype=int) for _ in range(self.npatches)]
        self.bucket_start_counter_list = [np.full((self.nx_buckets, self.ny_buckets), 0, dtype=int) for _ in range(self.npatches)]

        self.x0s = [p.x0 - p.dx/2 for p in self.patches]
        self.y0s = [p.y0 - p.dy/2 for p in self.patches]
        
    def __call__(self) -> None:
        self._update_particle_lists()
        sort_particles_patches_2d(
            self.x_list, self.y_list, self.is_dead_list, self.attrs_list,
            self.x0s, self.y0s, 
            self.nx_buckets, self.ny_buckets, self.dx_buckets, self.dy_buckets, 
            self.npatches, 
            self.bucket_count_list, self.bucket_bound_min_list, self.bucket_bound_max_list, self.bucket_count_not_list, self.bucket_start_counter_list, 
            self.particle_index_list, self.particle_index_ref_list, 
            self.particle_index_target_list, self.buf_list
        )


class ParticleSort3D(ParticleSort2D):
    """
    Sort particles in 3D after particle sync: no particles in guard cells
    """
    def __init__(
            self, 
            patches: Patches, 
            species: Species,
            nx_buckets: int|None=None,
            ny_buckets: int|None=None,
            nz_buckets: int|None=None,
            dx_buckets: float|None=None,
            dy_buckets: float|None=None,
            dz_buckets: float|None=None,
            attrs: List[str]|None=None,
        ) -> None:
        """
        Construct from patches.

        Parameters
        ----------
        patches : Patches
            Patches to be sorted.
        ispec : int
            Species index.
        """
        self.dimension = patches.dimension
        self.patches = patches
        self.npatches: int = patches.npatches
        self.ispec = species.ispec

        self.attrs = attrs or patches[0].particles[self.ispec].attrs
        self.nattrs = len(self.attrs)
        
        self.dx_buckets = dx_buckets or patches.dx
        self.dy_buckets = dy_buckets or patches.dy
        self.dz_buckets = dz_buckets or patches.dz
        self.nx_buckets = nx_buckets or patches.nx
        self.ny_buckets = ny_buckets or patches.ny
        self.nz_buckets = nz_buckets or patches.nz

        self.generate_particle_lists()
        self.generate_field_lists()

    def generate_particle_lists(self) -> None:
        """
        Generate typed.List of particle data of all species in all patches.

        Parameters
        ----------
        particle_list : list of Particles
            List of particles of all patches. 
        """
        ispec = self.ispec
        
        self.x_list = [p.particles[ispec].x for p in self.patches]
        self.y_list = [p.particles[ispec].y for p in self.patches]
        self.z_list = [p.particles[ispec].z for p in self.patches]

        self.attrs_list = [getattr(p.particles[ispec], attr) for p in self.patches for attr in self.attrs]

        self.is_dead_list = [p.particles[ispec].is_dead for p in self.patches]
        
        self.particle_index_list = [np.full(p.particles[ispec].is_dead.size, -1, dtype=int) for p in self.patches]
        self.particle_index_ref_list = [np.full(p.particles[ispec].is_dead.size, -1, dtype=int) for p in self.patches]
        self.particle_index_target_list = [np.full(p.particles[ispec].is_dead.size, -1, dtype=int) for p in self.patches]
        self.buf_list = [np.full(p.particles[ispec].is_dead.size, 0, dtype=float) for p in self.patches]

    def update_particle_lists(self, ipatch: int) -> None:
        """
        Update particle lists of a species in a patch.

        Parameters
        ----------
        ipatch : int
            Patch index.
        """
        particles = self.patches[ipatch].particles[self.ispec]

        self.x_list[ipatch] = particles.x
        self.y_list[ipatch] = particles.y
        self.z_list[ipatch] = particles.z

        for iattr, attr in enumerate(self.attrs):
            self.attrs_list[iattr+self.nattrs*ipatch] = getattr(particles, attr)
                
        self.is_dead_list[ipatch] = particles.is_dead
        
        self.particle_index_list[ipatch] = np.full(particles.is_dead.size, -1, dtype=int)
        self.particle_index_ref_list[ipatch] = np.full(particles.is_dead.size, -1, dtype=int)
        self.particle_index_target_list[ipatch] = np.full(particles.is_dead.size, -1, dtype=int)
        self.buf_list[ipatch] = np.full(particles.is_dead.size, 0, dtype=float)
        
    def generate_field_lists(self) -> None:
        """
        Update field lists of all patches.

        Parameters
        ----------
        fields : list of Fields3D
            List of fields of all patches.
        """
        self.bucket_count_list = [np.full((self.nx_buckets, self.ny_buckets, self.nz_buckets), 0, dtype=int) for _ in range(self.npatches)]
        self.bucket_bound_min_list = [np.full((self.nx_buckets, self.ny_buckets, self.nz_buckets), 0, dtype=int) for _ in range(self.npatches)]
        self.bucket_bound_max_list = [np.full((self.nx_buckets, self.ny_buckets, self.nz_buckets), 0, dtype=int) for _ in range(self.npatches)]
        self.bucket_count_not_list = [np.full((self.nx_buckets, self.ny_buckets, self.nz_buckets), 0, dtype=int) for _ in range(self.npatches)]
        self.bucket_start_counter_list = [np.full((self.nx_buckets, self.ny_buckets, self.nz_buckets), 0, dtype=int) for _ in range(self.npatches)]

        self.x0s = [p.x0 - p.dx/2 for p in self.patches]
        self.y0s = [p.y0 - p.dy/2 for p in self.patches]
        self.z0s = [p.z0 - p.dz/2 for p in self.patches]
        
    def __call__(self) -> None:
        self._update_particle_lists()
        sort_particles_patches_3d(
            self.x_list, self.y_list, self.z_list, self.is_dead_list, self.attrs_list,
            self.x0s, self.y0s, self.z0s, 
            self.nx_buckets, self.ny_buckets, self.nz_buckets, self.dx_buckets, self.dy_buckets, self.dz_buckets, 
            self.npatches, 
            self.bucket_count_list, self.bucket_bound_min_list, self.bucket_bound_max_list, self.bucket_count_not_list, self.bucket_start_counter_list, 
            self.particle_index_list, self.particle_index_ref_list, 
            self.particle_index_target_list, self.buf_list
        )
