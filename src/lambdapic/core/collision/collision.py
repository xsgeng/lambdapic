
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
        if self.patches.dimension == 2:
            self.debye_length_inv_square_list = typed.List(# type: ignore
                [np.zeros((p.nx, p.ny), dtype=np.float64) for p in self.patches]
            )
            self.total_density_list = typed.List(# type: ignore
                [np.zeros((p.nx, p.ny), dtype=np.float64) for p in self.patches]
            )
        elif self.patches.dimension == 3:
            self.debye_length_inv_square_list = typed.List(# type: ignore
                [np.zeros((p.nx, p.ny, p.nz), dtype=np.float64) for p in self.patches]
            )
            self.total_density_list = typed.List(# type: ignore
                [np.zeros((p.nx, p.ny, p.nz), dtype=np.float64) for p in self.patches]
            )
        self._check_shared_bucket_ordering()
        self.bucket_bound_min_lists = [typed.List(s.bucket_bound_min_list) for s in self.sorter] # type: ignore
        self.bucket_bound_max_lists = [typed.List(s.bucket_bound_max_list) for s in self.sorter] # type: ignore
        self.gen_list = typed.List(self.gen.spawn(self.patches.npatches)) # type: ignore

    def _check_shared_bucket_ordering(self) -> None:
        """Verify that sorters in collision groups share the same bucket
        ordering.

        Inter-species collisions iterate both species' ``bucket_bound_*``
        arrays in lockstep, so the sorters must use the same ``reverse_x``
        setting. The per-species ``_decide_reverse_x`` auto-decision (1D-bucket
        path only) can produce mismatched settings when a ``Collision`` is
        constructed after ``sim.initialize()`` without ``add_collision`` — the
        sorters were initialized as 1D-bucket and independently decided their
        x numbering. A mismatch makes every bucket pair against an empty one
        and the collision step silently becomes a no-op.

        Raises:
            ValueError: if two collision-group sorters have been sorted with
                different ``reverse_x`` values.
        """
        collision_ispecs = set()
        for group in self.collision_groups:
            for s in group:
                collision_ispecs.add(s.ispec)
        reverse_by_ispec: Dict[int, bool] = {}
        for ispec in collision_ispecs:
            sorter = self.sorter[ispec]
            if not hasattr(sorter, "reverse_x"):
                continue
            if sorter._reverse_decided:
                reverse_by_ispec[ispec] = bool(sorter.reverse_x)
        if len(set(reverse_by_ispec.values())) > 1:
            raise ValueError(
                f"Collision group sorters have inconsistent 'reverse_x' "
                f"settings ({reverse_by_ispec}); inter-species collisions "
                f"require shared bucket numbering. Register collision groups "
                f"via Simulation.add_collision() before initialize() so "
                f"sorters use 2D buckets (which force reverse_x=False), or "
                f"pass reverse_x=False explicitly when constructing "
                f"ParticleSort2D/3D."
            )

    def __getstate__(self) -> Dict:
        state = super().__getstate__()

        # these cannot be pickled
        del state['part_lists']
        del state['bucket_bound_min_lists']
        del state['bucket_bound_max_lists']
        return state
    
    def __setstate__(self, state: Dict) -> None:
        super().__setstate__(state)
        self.generate_particle_lists()
        self.generate_field_lists()

    def calculate_debye_length(self) -> None:
        for s in self.all_species:
            debye_length_patches(
                self.part_lists[s.ispec],
                self.bucket_bound_min_lists[s.ispec],
                self.bucket_bound_max_lists[s.ispec],
                self.cell_vol,
                self.debye_length_inv_square_list,
                self.total_density_list,
                reset=s.ispec==0
            )
        constrain_debye_length_patches(
            self.debye_length_inv_square_list,
            self.total_density_list
        )

    def enable(self, collision: Tuple[Species, Species]) -> None:
        s1, s2 = collision
        pair = (s1.ispec, s2.ispec) if s1.ispec < s2.ispec else (s2.ispec, s1.ispec)
        if pair not in self.collisions:
            raise ValueError(f"Collision pair {collision} not found")
        self._enabled[pair] = True

    def disable(self, collision: Tuple[Species, Species]) -> None:
        s1, s2 = collision
        pair = (s1.ispec, s2.ispec) if s1.ispec < s2.ispec else (s2.ispec, s1.ispec)
        if pair not in self.collisions:
            raise ValueError(f"Collision pair {collision} not found")
        self._enabled[pair] = False
    
    def __call__(self, dt: float) -> Any:
        for pair in self.collisions:
            if not self._enabled[pair]:
                continue

            ispec1, ispec2 = pair
            if ispec1 == ispec2:
                ispec = ispec1
                intra_collision_patches(
                    self.part_lists[ispec], self.bucket_bound_min_lists[ispec], self.bucket_bound_max_lists[ispec],
                    self.lnLambda,
                    self.debye_length_inv_square_list,
                    self.cell_vol, dt,
                    self.gen_list
                )
            else:
                inter_collision_patches(
                    self.part_lists[ispec1], self.bucket_bound_min_lists[ispec1], self.bucket_bound_max_lists[ispec1],
                    self.part_lists[ispec2], self.bucket_bound_min_lists[ispec2], self.bucket_bound_max_lists[ispec2],
                    self.patches.npatches,
                    self.lnLambda,
                    self.debye_length_inv_square_list,
                    self.cell_vol, dt,
                    self.gen_list
                )