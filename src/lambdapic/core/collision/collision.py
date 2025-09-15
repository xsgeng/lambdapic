
from typing import Any, List, Sequence, Tuple

import numpy as np
from numba import typed
from scipy.constants import c, epsilon_0, pi

from ..patch.patch import Patches
from ..sort.particle_sort import ParticleSort2D, ParticleSort3D
from ..species import Species
from .cpu import debye_length_patches, constrain_debye_length_patches
from .utils import ParticleData, pack_particle_data


class Collission:
    """
    
    """
    def __init__(self, collision_groups: Sequence[Sequence[Species]], patches: Patches, sorter: Sequence[ParticleSort2D]):
        self.collision_groups = collision_groups

        self.all_species = patches.species
        self.patches = patches
        self.sorter = sorter

        self.cell_vol = self.patches.dx
        if self.patches.dimension >= 2:
            self.cell_vol *= self.patches.dy
        if self.patches.dimension == 3:
            self.cell_vol *= self.patches.dz


        collisions = set()
        self.collisions = []
        for group in self.collision_groups:
            for s1 in group:
                for s2 in group:
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

    def generate_field_lists(self) -> None:
        self.debye_length_inv_sqare_list = typed.List(# type: ignore
            [np.zeros((p.nx, p.ny), dtype=np.float64) for p in self.patches]
        )
        self.total_density_list = typed.List(# type: ignore
            [np.zeros((p.nx, p.ny), dtype=np.float64) for p in self.patches]
        )

    def calculate_debye_length(self) -> None:
        for s in self.all_species:
            debye_length_patches(
                self.part_lists[s.ispec],
                self.sorter[s.ispec].bucket_bound_min_list,
                self.sorter[s.ispec].bucket_bound_max_list,
                self.cell_vol,
                self.debye_length_inv_sqare_list,
                self.total_density_list,
                reset=s.ispec==0
            )
        constrain_debye_length_patches(
            self.debye_length_inv_sqare_list,
            self.total_density_list
        )

    def __call__(self) -> Any:
        pass