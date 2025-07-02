
from typing import Any, Sequence

from scipy.constants import c, epsilon_0, pi

from ..patch.patch import Patches
from ..sort.particle_sort import ParticleSort2D
from ..species import Species


class Collission:
    def __init__(
        self,
        patches: Patches,
        species: Sequence[Species],
        sorter: Sequence[ParticleSort2D]
    ):
        self.patches = patches
        self.ispec1 = patches.species.index(species[0])
        self.ispec2 = patches.species.index(species[1])
        self.sorter = sorter

        self.q1 = species[0].q
        self.m1 = species[0].m
        self.q2 = species[1].q
        self.m2 = species[1].m

    def __call__(self) -> Any:
        pass