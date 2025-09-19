
from typing import Any, Dict, List, Sequence, Tuple
from itertools import combinations

import numpy as np
from numba import typed
from scipy.constants import c, epsilon_0, pi

from ..patch.patch import Patches
from ..sort.particle_sort import ParticleSort2D, ParticleSort3D
from ..species import Species
from ..utils.pickle_list import PickleableTypedList
from .cpu import (
    constrain_debye_length_patches,
    debye_length_patches,
    inter_collision_patches,
    intra_collision_patches,
)
from .utils import ParticleData, pack_particle_data


class Collision(PickleableTypedList):
    """
    Particle collision module for PIC simulations.
    
    Handles both intra-species and inter-species collisions using
    a binary collision model with Coulomb scattering.

    Parameters:
        collision_groups (Sequence[Sequence[Species]]): A list of groups of species that can collide with each other. Example: [[s1, s1, s2, s3]].
            s1 will collide with it self, and s1, s2, and s3 will collide with each other. 
        patches (Patches): The patches object containing the simulation domain.
        sorter (Sequence[ParticleSort2D]): The particle sorter object.
        gen (np.random.Generator): The random number generator object.
    """
    def __init__(self, collision_groups: Sequence[Sequence[Species]], patches: Patches, sorter: Sequence[ParticleSort2D], gen: np.random.Generator):
        self.collision_groups = collision_groups

        self.all_species = patches.species
        self.patches = patches
        self.sorter = sorter
        self.gen = gen

        self.lnLambda = 0.0

        self.cell_vol = self.patches.dx
        if self.patches.dimension >= 2:
            self.cell_vol *= self.patches.dy
        if self.patches.dimension == 3:
            self.cell_vol *= self.patches.dz


        collisions = set()
        self.collisions: List[tuple[int, int]] = []
        # Track enabled/disabled state for each collision pair
        self._enabled: Dict[tuple[int, int], bool] = {}
        for group in self.collision_groups:
            for s1, s2 in combinations(group, 2):
                pair = (s1.ispec, s2.ispec) if s1.ispec < s2.ispec else (s2.ispec, s1.ispec)
                if pair not in collisions:
                    self.collisions.append(pair)
                    self._enabled[pair] = True
                collisions.add(pair)
            
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
        for pair in self.collisions:
            if not self._enabled[pair]:
                continue

            ispec1, ispec2 = pair
            if ispec1 == ispec2:
                ispec = ispec1
                intra_collision_patches(
                    self.part_lists[ispec], self.bucket_bound_min_list[ispec], self.bucket_bound_max_list[ispec],
                    self.lnLambda,
                    self.debye_length_inv_square_list,
                    self.cell_vol, dt,
                    self.gen_list
                )
            else:
                inter_collision_patches(
                    self.part_lists[ispec1], self.bucket_bound_min_list[ispec1], self.bucket_bound_max_list[ispec1],
                    self.part_lists[ispec2], self.bucket_bound_min_list[ispec2], self.bucket_bound_max_list[ispec2],
                    self.patches.npatches,
                    self.lnLambda,
                    self.debye_length_inv_square_list,
                    self.cell_vol, dt,
                    self.gen_list
                )