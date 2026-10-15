
from typing import Any, Dict, List, Sequence, Tuple
from itertools import combinations

import numpy as np
from scipy.constants import c, epsilon_0, pi

from ..patch.patch import Patches
from ..sort.particle_sort import ParticleSort2D, ParticleSort3D
from ..species import Species
from .cpu import (
    constrain_debye_length_patches,
    debye_length_patches,
    inter_collision_patches,
    intra_collision_patches,
    seed_rng_states,
)


class Collision:
    """
    Particle collision module for PIC simulations.

    Handles both intra-species and inter-species collisions using
    a binary collision model with Coulomb scattering.

    Parameters:
        collision_groups (Sequence[Sequence[Species]]): A list of groups of species that can collide with each other. Example: [[s1, s1, s2, s3]].
            s1 will collide with it self, and s1, s2, and s3 will collide with each other.
        patches (Patches): The patches object containing the simulation domain.
        sorter (Sequence[ParticleSort2D]): The particle sorter object.
        gen (np.random.Generator): The random number generator object. Used to derive the per-patch PCG32 seeds of rng_state_list (re-derived whenever generate_field_lists() runs: rebalance/restart).
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
        # no-op: particle object lists are constructed at the call sites
        # (see calculate_debye_length / __call__), so they can never go stale
        pass

    def generate_field_lists(self) -> None:
        if self.patches.dimension == 2:
            self.debye_length_inv_square_list = [
                np.zeros((p.nx, p.ny), dtype=np.float64) for p in self.patches
            ]
            self.total_density_list = [
                np.zeros((p.nx, p.ny), dtype=np.float64) for p in self.patches
            ]
        elif self.patches.dimension == 3:
            self.debye_length_inv_square_list = [
                np.zeros((p.nx, p.ny, p.nz), dtype=np.float64) for p in self.patches
            ]
            self.total_density_list = [
                np.zeros((p.nx, p.ny, p.nz), dtype=np.float64) for p in self.patches
            ]
        self._check_shared_bucket_ordering()
        # bucket bounds are NOT copied here: the sorters replace those arrays
        # wholesale in their own generate_field_lists(), so a generate-time
        # copy would go stale whenever the sorters regenerate after this
        # module (ordering-dependent). calculate_debye_length/__call__ fetch
        # the live lists from the sorters at call time instead.
        # PCG32 states, one stream per patch; deterministic given gen's seed
        seeds = self.gen.integers(0, 2**62, size=self.patches.npatches, dtype=np.int64)
        self.rng_state_list: np.ndarray = seed_rng_states(seeds)
        assert self.rng_state_list.shape == (self.patches.npatches, 2)

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
        # plain containers pickle natively; the sorter-derived bucket bounds
        # are fetched at call time (calculate_debye_length / __call__), so
        # there is nothing sorter-derived left to regenerate
        return self.__dict__.copy()

    def __setstate__(self, state: Dict) -> None:
        # drop bucket-bound copies from pre-call-time checkpoints (unused now)
        for k in [k for k in state if k.startswith("bucket_bound_")]:
            del state[k]
        self.__dict__.update(state)
        self.generate_field_lists()

    def calculate_debye_length(self) -> None:
        for s in self.all_species:
            sorter = self.sorter[s.ispec]
            debye_length_patches(
                [p.particles[s.ispec] for p in self.patches],
                s.m, s.q,
                sorter.bucket_bound_min_list, sorter.bucket_bound_max_list,
                self.cell_vol,
                self.debye_length_inv_square_list, self.total_density_list,
                int(s.ispec == 0)
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
                sorter = self.sorter[ispec]
                intra_collision_patches(
                    [p.particles[ispec] for p in self.patches],
                    self.all_species[ispec].m, self.all_species[ispec].q,
                    sorter.bucket_bound_min_list, sorter.bucket_bound_max_list,
                    self.lnLambda,
                    self.debye_length_inv_square_list,
                    self.cell_vol, dt,
                    self.rng_state_list
                )
            else:
                s1 = self.all_species[ispec1]
                s2 = self.all_species[ispec2]
                sorter1 = self.sorter[ispec1]
                sorter2 = self.sorter[ispec2]
                inter_collision_patches(
                    [p.particles[ispec1] for p in self.patches],
                    s1.m, s1.q,
                    sorter1.bucket_bound_min_list, sorter1.bucket_bound_max_list,
                    [p.particles[ispec2] for p in self.patches],
                    s2.m, s2.q,
                    sorter2.bucket_bound_min_list, sorter2.bucket_bound_max_list,
                    self.lnLambda,
                    self.debye_length_inv_square_list,
                    self.cell_vol, dt,
                    self.rng_state_list
                )
