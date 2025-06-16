from typing import Any

import numpy as np
from numba import njit, prange, typed, types
from scipy.constants import c, e, epsilon_0, mu_0

from lambdapic.core.particles import SpinParticles

from ..patch import Patches

from .cpu import boris_push_patches, push_position_patches_2d, photon_push_patches
from .unified.unified_pusher_2d import unified_boris_pusher_cpu_2d
from .unified.unified_pusher_3d import unified_boris_pusher_cpu_3d


class PusherBase:
    def __init__(self, patches: Patches, ispec: int) -> None:
        """
        Construct from patches.

        Parameters
        ----------
        
        """
        self.dimension = patches.dimension
        self.patches = patches
        self.npatches: int = patches.npatches
        self.ispec = ispec

        self.q = patches.species[ispec].q
        self.m = patches.species[ispec].m

        self.generate_particle_lists()

    def generate_particle_lists(self) -> None:
        """
        Generate typed.List of particle data of all species in all patches.

        Parameters
        ----------
        particle_list : list of Particles
            List of particles of all patches. 
        """

        ispec = self.ispec
        self.x_list = typed.List([p.particles[ispec].x for p in self.patches])
        if self.dimension >= 2:
            self.y_list = typed.List([p.particles[ispec].y for p in self.patches])
        if self.dimension == 3:
            self.z_list = typed.List([p.particles[ispec].z for p in self.patches])
        self.ux_list = typed.List([p.particles[ispec].ux for p in self.patches])
        self.uy_list = typed.List([p.particles[ispec].uy for p in self.patches])
        self.uz_list = typed.List([p.particles[ispec].uz for p in self.patches])
        self.inv_gamma_list = typed.List([p.particles[ispec].inv_gamma for p in self.patches])

        self.ex_part_list = typed.List([p.particles[ispec].ex_part for p in self.patches])
        self.ey_part_list = typed.List([p.particles[ispec].ey_part for p in self.patches])
        self.ez_part_list = typed.List([p.particles[ispec].ez_part for p in self.patches])
        self.bx_part_list = typed.List([p.particles[ispec].bx_part for p in self.patches])
        self.by_part_list = typed.List([p.particles[ispec].by_part for p in self.patches])
        self.bz_part_list = typed.List([p.particles[ispec].bz_part for p in self.patches])

        self.is_dead_list = typed.List([p.particles[ispec].is_dead for p in self.patches])


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

        self.ux_list[ipatch] = particles.ux
        self.uy_list[ipatch] = particles.uy
        self.uz_list[ipatch] = particles.uz
        self.inv_gamma_list[ipatch] = particles.inv_gamma

        self.ex_part_list[ipatch] = particles.ex_part
        self.ey_part_list[ipatch] = particles.ey_part
        self.ez_part_list[ipatch] = particles.ez_part
        self.bx_part_list[ipatch] = particles.bx_part
        self.by_part_list[ipatch] = particles.by_part
        self.bz_part_list[ipatch] = particles.bz_part
        self.is_dead_list[ipatch] = particles.is_dead


    def push_position(self, dt: float):
        if self.dimension == 2:
            push_position_patches_2d(
                self.x_list, self.y_list,
                self.ux_list, self.uy_list, self.inv_gamma_list,
                self.is_dead_list, 
                self.npatches, dt,
            )


    def __call__(self, dt: float) -> None:
        raise NotImplementedError
    

class BorisPusher(PusherBase):
    def __call__(self, dt: float, unified: bool=False) -> None:
        if unified:
            if self.patches.dimension == 2:
                unified_boris_pusher_cpu_2d(
                    [p.particles[self.ispec] for p in self.patches],
                    [p.fields for p in self.patches],
                    self.npatches, dt, self.q, self.m
                )
            if self.patches.dimension == 3:
                unified_boris_pusher_cpu_3d(
                    [p.particles[self.ispec] for p in self.patches],
                    [p.fields for p in self.patches],
                    self.npatches, dt, self.q, self.m
                )
        else:
            boris_push_patches(
                self.ux_list, self.uy_list, self.uz_list, self.inv_gamma_list,
                self.ex_part_list, self.ey_part_list, self.ez_part_list,
                self.bx_part_list, self.by_part_list, self.bz_part_list,
                self.is_dead_list,
                self.npatches, self.q, self.m, dt
            )

class UnifiedBorisPusher2D(PusherBase):
    def __call__(self, dt: float) -> None:
        unified_boris_pusher_cpu_2d(
            [p.particles[self.ispec] for p in self.patches],
            [p.fields for p in self.patches],
            self.npatches, dt, self.q, self.m
        )

class UnifiedBorisPusher3D(PusherBase):
    def __call__(self, dt: float) -> None:
        unified_boris_pusher_cpu_3d(
            [p.particles[self.ispec] for p in self.patches],
            [p.fields for p in self.patches],
            self.npatches, dt, self.q, self.m
        )


class PhotonPusher(PusherBase):
    def __call__(self, dt: float):
        ...
        

class BorisTBMTPusher(PusherBase):
    def generate_particle_lists(self) -> None:
        super().generate_particle_lists()
        self.sx_list = typed.List([p.particles[self.ispec].sx for p in self.patches])
        self.sy_list = typed.List([p.particles[self.ispec].sy for p in self.patches])
        self.sz_list = typed.List([p.particles[self.ispec].sz for p in self.patches])


    def update_particle_lists(self, ipatch: int) -> None:
        super().update_particle_lists(ipatch)
        particles = self.patches[ipatch].particles[self.ispec]
        assert isinstance(particles, SpinParticles)
        self.sx_list[ipatch] = particles.sx
        self.sy_list[ipatch] = particles.sy
        self.sz_list[ipatch] = particles.sz


    def __call__(self, dt: float) -> None:
        ...