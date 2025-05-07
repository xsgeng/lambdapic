import numpy as np
from ..patch import Patches

from .cpu2d import sort_particles_patches_2d
from .cpu3d import sort_particles_patches_3d

class ParticleSort2D:
    """
    sort after particle sync: no particles in guard cells
    """
    def __init__(self, patches: Patches, ispec: int) -> None:
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
        self.ispec = ispec
        self.attrs = patches[0].particles[ispec].attrs
        self.nattrs = len(self.attrs)
        
        self.dx: float = patches.dx
        self.dy: float = patches.dy
        self.nx: int = patches.nx
        self.ny: int = patches.ny

        self.n_guard: int = patches.n_guard

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
        
        self.particle_cell_indices_list = [np.full(p.particles[ispec].is_dead.size, -1, dtype=int) for p in self.patches]
        self.sorted_indices_list = [np.full(p.particles[ispec].is_dead.size, -1, dtype=int) for p in self.patches]

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
        
        self.particle_cell_indices_list[ipatch] = np.full(particles.is_dead.size, -1, dtype=int)
        self.sorted_indices_list[ipatch] = np.full(particles.is_dead.size, -1, dtype=int)
        
    def generate_field_lists(self) -> None:
        """
        Update field lists of all patches.

        Parameters
        ----------
        fields : list of Fields2D
            List of fields of all patches.
        """
        self.grid_cell_count_list = [np.full((self.nx, self.ny), 0, dtype=int) for _ in range(self.npatches)]
        self.cell_bound_min_list = [np.full((self.nx, self.ny), -1, dtype=int) for _ in range(self.npatches)]
        self.cell_bound_max_list = [np.full((self.nx, self.ny), -1, dtype=int) for _ in range(self.npatches)]

        self.x0s = [p.x0 - self.dx/2 for p in self.patches]
        self.y0s = [p.y0 - self.dy/2 for p in self.patches]
        
    def __call__(self) -> None:
        sort_particles_patches_2d(
            self.grid_cell_count_list, self.cell_bound_min_list, self.cell_bound_max_list, self.x0s, self.y0s,
            self.nx, self.ny, self.dx, self.dy, 
            self.npatches, 
            self.particle_cell_indices_list, self.sorted_indices_list, self.x_list, self.y_list, self.is_dead_list,
            self.attrs_list
        )


class ParticleSort3D:
    """
    Sort particles in 3D after particle sync: no particles in guard cells
    """
    def __init__(self, patches: Patches, ispec: int) -> None:
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
        self.ispec = ispec
        self.attrs = patches[0].particles[ispec].attrs
        self.nattrs = len(self.attrs)
        
        self.dx: float = patches.dx
        self.dy: float = patches.dy
        self.dz: float = patches.dz
        self.nx: int = patches.nx
        self.ny: int = patches.ny
        self.nz: int = patches.nz

        self.n_guard: int = patches.n_guard

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
        
        self.particle_cell_indices_list = [np.full(p.particles[ispec].is_dead.size, -1, dtype=int) for p in self.patches]
        self.sorted_indices_list = [np.full(p.particles[ispec].is_dead.size, -1, dtype=int) for p in self.patches]

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
        
        self.particle_cell_indices_list[ipatch] = np.full(particles.is_dead.size, -1, dtype=int)
        self.sorted_indices_list[ipatch] = np.full(particles.is_dead.size, -1, dtype=int)
        
    def generate_field_lists(self) -> None:
        """
        Update field lists of all patches.

        Parameters
        ----------
        fields : list of Fields3D
            List of fields of all patches.
        """
        self.grid_cell_count_list = [np.full((self.nx, self.ny, self.nz), 0, dtype=int) for _ in range(self.npatches)]
        self.cell_bound_min_list = [np.full((self.nx, self.ny, self.nz), -1, dtype=int) for _ in range(self.npatches)]
        self.cell_bound_max_list = [np.full((self.nx, self.ny, self.nz), -1, dtype=int) for _ in range(self.npatches)]

        self.x0s = [p.x0 - self.dx/2 for p in self.patches]
        self.y0s = [p.y0 - self.dy/2 for p in self.patches]
        self.z0s = [p.z0 - self.dz/2 for p in self.patches]
        
    def __call__(self) -> None:
        sort_particles_patches_3d(
            self.grid_cell_count_list, self.cell_bound_min_list, self.cell_bound_max_list, 
            self.x0s, self.y0s, self.z0s,
            self.nx, self.ny, self.nz, self.dx, self.dy, self.dz, 
            self.npatches, 
            self.particle_cell_indices_list, self.sorted_indices_list, 
            self.x_list, self.y_list, self.z_list, self.is_dead_list,
            self.attrs_list
        )
