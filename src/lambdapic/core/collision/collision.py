
from typing import Any, List, Sequence, Tuple
from itertools import combinations

import numpy as np
from numba import typed
from scipy.constants import c, epsilon_0, pi

from ..patch.patch import Patches
from ..sort.particle_sort import ParticleSort2D, ParticleSort3D
from ..species import Species
from .cpu import (
    constrain_debye_length_patches,
    debye_length_patches,
    inter_collision_patches,
    intra_collision_patches,
)
from .utils import ParticleData, pack_particle_data


class Collision:
    """
    
    """
    def __init__(self, collision_groups: Sequence[Sequence[Species]], patches: Patches, sorter: Sequence[ParticleSort2D], gen: np.random.Generator):
        self.collision_groups = collision_groups

        self.all_species = patches.species
        self.patches = patches
        self.sorter = sorter
        self.gen = gen

        self.lnLambda = 2.0

        self.cell_vol = self.patches.dx
        if self.patches.dimension >= 2:
            self.cell_vol *= self.patches.dy
        if self.patches.dimension == 3:
            self.cell_vol *= self.patches.dz


        collisions = set()
        self.collisions: List[Tuple[Species, Species]] = []
        for group in self.collision_groups:
            for s1, s2 in combinations(group, 2):
                if (s1.ispec, s2.ispec) not in collisions:
                    self.collisions.append((s1, s2))
                collisions.add((s1.ispec, s2.ispec))
            
    def generate_particle_lists(self) -> None:
        self.part_lists: List[List[ParticleData]] = []
        for s in self.all_species:
            self.part_lists.append(
                typed.List(# type: ignore
                    pack_particle_data([p.particles[s.ispec] for p in self.patches], s.m, s.q)
                )
            )
    
    def update_particle_lists(self) -> None:
        for ispec, s in enumerate(self.patches.species):
            for ipatch, p in enumerate(self.patches):
                if p.particles[ispec].extended:
                    self.part_lists[ispec][ipatch] = pack_particle_data(p.particles[ispec], s.m, s.q)

    def generate_field_lists(self) -> None:
        self.debye_length_inv_square_list = typed.List(# type: ignore
            [np.zeros((p.nx, p.ny), dtype=np.float64) for p in self.patches]
        )
        self.total_density_list = typed.List(# type: ignore
            [np.zeros((p.nx, p.ny), dtype=np.float64) for p in self.patches]
        )
        self.bucket_bound_min_list = [typed.List(s.bucket_bound_min_list) for s in self.sorter] # type: ignore
        self.bucket_bound_max_list = [typed.List(s.bucket_bound_max_list) for s in self.sorter] # type: ignore
        self.gen_list = typed.List(self.gen.spawn(self.patches.npatches)) # type: ignore

    def calculate_debye_length(self) -> None:
        for s in self.all_species:
            debye_length_patches(
                self.part_lists[s.ispec],
                self.bucket_bound_min_list[s.ispec],
                self.bucket_bound_max_list[s.ispec],
                self.cell_vol,
                self.debye_length_inv_square_list,
                self.total_density_list,
                reset=s.ispec==0
            )
        constrain_debye_length_patches(
            self.debye_length_inv_square_list,
            self.total_density_list
        )

    def __call__(self, dt: float) -> Any:
        for (s1, s2) in self.collisions:
            if s1 is s2:
                intra_collision_patches(
                    self.part_lists[s1.ispec], self.bucket_bound_min_list[s1.ispec], self.bucket_bound_max_list[s1.ispec],
                    self.lnLambda,
                    self.debye_length_inv_square_list,
                    self.cell_vol, dt,
                    self.gen_list
                )
            else:
                inter_collision_patches(
                    self.part_lists[s1.ispec], self.bucket_bound_min_list[s1.ispec], self.bucket_bound_max_list[s1.ispec],
                    self.part_lists[s2.ispec], self.bucket_bound_min_list[s2.ispec], self.bucket_bound_max_list[s2.ispec],
                    self.patches.npatches,
                    self.lnLambda,
                    self.debye_length_inv_square_list,
                    self.cell_vol, dt,
                    self.gen_list
                )