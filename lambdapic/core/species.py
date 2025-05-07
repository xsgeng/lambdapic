import inspect
from functools import cached_property
from typing import Callable, Literal

from numba import njit
from numba.extending import is_jitted
from pydantic import BaseModel, computed_field, Field
from scipy.constants import e, m_e, m_p

from .particles import (ParticlesBase, QEDParticles, SpinParticles,
                        SpinQEDParticles)


class Species(BaseModel):
    name: str
    charge: int
    mass: float

    density: Callable = None
    density_min: float = 0
    ppc: int = 0

    momentum: tuple[Callable, Callable, Callable] = (None, None, None)
    polarization: tuple[float, float, float] = None

    pusher: Literal["boris", "photon", "boris+tbmt"] = "boris"

    ispec: int = None

    @computed_field
    @cached_property
    def q(self) -> float:
        return self.charge * e

    @computed_field
    @cached_property
    def m(self) -> float:
        return self.mass * m_e

    @computed_field
    @cached_property
    def density_jit(self) -> Callable | None:
        if is_jitted(self.density):
            self.density.enable_caching()
            return self.density
        elif inspect.isfunction(self.density):
            return njit(self.density)
        elif self.density is None:
            return None
        

    def create_particles(self, ipatch: int=None, rank: int=None) -> ParticlesBase:
        """ 
        Create Particles from the species.

        Particles class holds the particle data.

        Called by patch. 

        Then particles are created within the patch.
        """
        return ParticlesBase(ipatch, rank)

class Electron(Species):
    name: str = 'electron'
    charge: int = -1
    mass: float = 1
    radiation: Literal["ll", "photons"] = None
    photon: Species = None

    def set_photon(self, photon: Species):
        assert self.radiation == "photons"
        assert isinstance(photon, Species)
        self.photon = photon

    def create_particles(self, ipatch: int=None, rank: int=None) -> ParticlesBase:
        if self.photon:
            if self.polarization is None:
                return QEDParticles(ipatch, rank)
            else:
                return SpinQEDParticles(ipatch, rank)
        elif self.polarization is not None:
            return SpinParticles(ipatch, rank)

        return super().create_particles(ipatch, rank)


class Positron(Electron):
    name: str = 'positron'
    charge: int = 1


class Proton(Species):
    name: str = 'proton'
    charge: int = 1
    mass: float = m_p/m_e


class Photon(Species):
    name: str = 'photon'
    charge: int = 0
    mass: float = 0

    pusher: str = "photon"

    electron: Species = None
    positron: Species = None

    def set_bw_pair(self, *, electron: Species, positron: Species):
        assert isinstance(electron, Species)
        assert isinstance(positron, Species)
        self.electron = electron
        self.positron = positron

    def create_particles(self, ipatch: int=None, rank: int=None) -> ParticlesBase:
        if self.electron is not None:
            return QEDParticles(ipatch, rank)
        # else:
        #     return SpinQEDParticles(ipatch, rank)

        return super().create_particles(ipatch, rank)
