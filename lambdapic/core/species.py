import inspect
from functools import cached_property
from typing import Callable, Literal

from numba import njit
from numba.extending import is_jitted
from pydantic import BaseModel, computed_field, Field, model_validator
from scipy.constants import e, m_e, m_p

from .particles import (ParticlesBase, QEDParticles, SpinParticles,
                        SpinQEDParticles)


class Species(BaseModel):
    name: str = Field(..., description="Name of the particle species")
    charge: int = Field(..., description="Charge number (e.g. -1 for electron, +1 for proton)")
    mass: float = Field(..., description="Mass in units of electron mass")

    density: Callable|None = Field(None, description="Function defining particle density distribution")
    density_min: float = Field(0, description="Minimum density threshold")
    ppc: int = Field(0, description="Particles per cell")

    momentum: tuple[Callable|None, Callable|None, Callable|None] = Field(
        (None, None, None), 
        description="Tuple of functions defining momentum distribution in x,y,z directions"
    )
    polarization: tuple[float, float, float]|None = Field(
        None, 
        description="Polarization vector (x,y,z components) for spin particles"
    )

    pusher: Literal["boris", "photon", "boris+tbmt"] = Field(
        "boris", 
        description="Particle pusher algorithm to use"
    )

    ispec: int|None = Field(None, description="Internal species index")

    @computed_field
    @cached_property
    def q(self) -> float:
        """charge in SI units"""
        return self.charge * e

    @computed_field
    @cached_property
    def m(self) -> float:
        """mass in SI units"""
        return self.mass * m_e

    @computed_field
    @cached_property
    def density_jit(self) -> Callable | None:
        """density function in JIT compiled form"""
        if is_jitted(self.density):
            self.density.enable_caching()
            return self.density
        elif inspect.isfunction(self.density):
            return njit(self.density)
        elif self.density is None:
            return None
        

    def create_particles(self, ipatch: int|None=None, rank: int|None=None) -> ParticlesBase:
        """ 
        Create Particles from the species.

        Particles class holds the particle data.

        Called by patch. 

        Then particles are created within the patch.
        """
        return ParticlesBase(ipatch, rank)

class Electron(Species):
    name: str = Field('electron', description="Electron particle species")
    charge: int = Field(-1, description="Electron charge (-1)")
    mass: float = Field(1, description="Electron mass (1 in units of electron mass)")
    radiation: Literal["ll", "photons"]|None = Field(
        None, 
        description="Radiation model ('ll' for Landau-Lifshitz, 'photons' for QED)"
    )
    photon: Species|None = Field(None, description="Photon species for QED radiation")

    def set_photon(self, photon: Species):
        assert self.radiation == "photons"
        assert isinstance(photon, Species)
        self.photon = photon

    def create_particles(self, ipatch: int|None=None, rank: int|None=None) -> ParticlesBase:
        if self.photon:
            if self.polarization is None:
                return QEDParticles(ipatch, rank)
            else:
                return SpinQEDParticles(ipatch, rank)
        elif self.polarization is not None:
            return SpinParticles(ipatch, rank)

        return super().create_particles(ipatch, rank)


class Positron(Electron):
    name: str = Field('positron', description="Positron particle species")
    charge: int = Field(1, description="Positron charge (+1)")


class Proton(Species):
    name: str = Field('proton', description="Proton particle species")
    charge: int = Field(1, description="Proton charge (+1)")
    mass: float = Field(m_p/m_e, description="Proton mass in units of electron mass")


class Photon(Species):
    name: str = Field('photon', description="Photon particle species")
    charge: int = Field(0, description="Photon charge (0)")
    mass: float = Field(0, description="Photon mass (0)")

    pusher: Literal["boris", "photon", "boris+tbmt"] = Field(
        "photon", 
        description="Photon pusher algorithm (must be 'photon')"
    )

    electron: Species|None = Field(None, description="Electron species for Breit-Wheeler pair production")
    positron: Species|None = Field(None, description="Positron species for Breit-Wheeler pair production")

    def set_bw_pair(self, *, electron: Species, positron: Species):
        assert isinstance(electron, Species)
        assert isinstance(positron, Species)
        self.electron = electron
        self.positron = positron

    def create_particles(self, ipatch: int|None=None, rank: int|None=None) -> ParticlesBase:
        if self.electron is not None:
            return QEDParticles(ipatch, rank)
        # else:
        #     return SpinQEDParticles(ipatch, rank)

        return super().create_particles(ipatch, rank)
