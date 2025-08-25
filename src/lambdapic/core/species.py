import inspect
from functools import cached_property
from types import FunctionType
from typing import Callable, Literal, Optional, Union
from dataclasses import dataclass, field

from numba import njit
from numba.core.dispatcher import Dispatcher
from numba.extending import is_jitted
from pydantic import BaseModel, Field, computed_field, model_validator
from scipy.constants import e, m_e, m_p

from .particles import ParticlesBase, QEDParticles, SpinParticles, SpinQEDParticles

Profile = Callable[[float, float, float], float] | Callable[[float, float], float]

class SpeciesConfig(BaseModel):
    name: str = Field(..., description="Name of the particle species")
    charge: int = Field(..., description="Charge number (e.g. -1 for electron, +1 for proton)")
    mass: float = Field(..., description="Mass in units of electron mass")

    density: Callable | None = Field(None, description="Function defining particle density distribution")
    density_min: float = Field(0, description="Minimum density threshold")
    ppc: Union[int, Callable] = Field(0, description="Particles per cell (constant int or coordinate-based function)")

    momentum: tuple[Profile | None, Profile | None, Profile | None] | None = Field(
        (None, None, None), 
        description="Tuple of functions defining momentum distribution in x,y,z directions"
    )
    polarization: tuple[float, float, float] | None = Field(
        None, 
        description="Polarization vector (x,y,z components) for spin particles"
    )

    pusher: Literal["boris", "photon", "boris+tbmt"] = Field(
        "boris", 
        description="Particle pusher algorithm to use"
    )

    # ispec: int | None = Field(None, description="Internal species index")
    
    # density_jit: Callable | None = Field(None, description="JIT-compiled density function")
    # ppc_jit: Callable | None = Field(None, description="JIT-compiled ppc function")


@dataclass(kw_only=True)    
class Species:
    """Base Species class
        
    Parameters:
        name (str): Particle species name
        charge (int): Particle charge
        mass (float): Particle mass in units of electron mass
        density (Callable, optional): Density function
        density_min (float): Minimum density threshold"
        ppc (int or Callable): Particles per cell (constant or function)
        momentum (tuple): Momentum distribution functions
        polarization (tuple, optional): Spin polarization vector
        pusher (str): Particle pusher algorithm
    """
    name: str
    charge: int
    mass: float
    
    density: Callable | None = field(default=None)
    density_min: float = field(default=0)
    ppc: int|Callable = field(default=0)
    momentum: tuple[Profile | None, Profile | None, Profile | None] | None = field(default=(None, None, None))
    polarization: tuple[float, float, float] | None = field(default=None)
    
    pusher: Literal["boris", "photon", "boris+tbmt"] = field(default="boris")
    
    def __post_init__(self):
        
        validated = SpeciesConfig(
            name=self.name,
            charge=self.charge,
            mass=self.mass,
            density=self.density,
            density_min=self.density_min,
            ppc=self.ppc,
            momentum=self.momentum,
            polarization=self.polarization,
            pusher=self.pusher
        )
        
        self.name = validated.name
        self.charge = validated.charge
        self.mass = validated.mass
        self.density = validated.density
        self.density_min = validated.density_min
        self.ppc = validated.ppc
        self.momentum = validated.momentum
        self.polarization = validated.polarization
        self.pusher = validated.pusher
        
        # in SI units
        self.m = self.mass * m_e
        self.q = self.charge * e
        
        # will be post initialized with dimension info
        self.density_jit: Callable | None = None
        self.ppc_jit: Callable | None = None

        self._aux_attrs: list[str] = []
        self._ispec: int | None = None
        
    @property
    def ispec(self) -> int:
        if self._ispec is None:
            raise ValueError("Species index is not set. Maybe not added via Simulation.add_species")
        return self._ispec
    
    @ispec.setter
    def ispec(self, value: int):
        self._ispec = value
        
    @staticmethod
    def compile_jit(func_or_val: Callable|Dispatcher|float|int, dimension: Literal[2, 3]) -> FunctionType:
        if is_jitted(func_or_val):
            assert isinstance(func_or_val, Dispatcher)
            func_or_val.enable_caching()
            return func_or_val
        
        elif inspect.isfunction(func_or_val):
            assert not isinstance(func_or_val, Dispatcher)
            narg = func_or_val.__code__.co_argcount
            if narg != dimension:
                raise ValueError(f"function {func_or_val} must have {dimension} arguments")
            return njit(func_or_val)
        
        elif isinstance(func_or_val, (int, float)):
            if dimension == 2:
                @njit('float64(float64, float64)')
                def jit_func2d(x, y):
                    return func_or_val
                return jit_func2d
            elif dimension == 3:
                @njit('float64(float64, float64, float64)')
                def jit_func3d(x, y, z):
                    return func_or_val
                return jit_func3d
            else:
                raise ValueError("dimension must be 2 or 3")
        
        else:
            raise ValueError(f"Invalid profile {func_or_val}. Must be a function, int or float.")
            

    def create_particles(self, ipatch: int | None=None, rank: int | None=None) -> ParticlesBase:
        """ 
        Create Particles from the species.

        Particles class holds the particle data.

        Called by patch. 

        Then particles are created within the patch.
        """
        return ParticlesBase(ipatch, rank)

@dataclass(kw_only=True)
class Electron(Species):
    name: str = field(default='electron', init=True)
    radiation: Literal["ll", "photons"] | None = field(default=None, init=True)
    
    charge: int = field(default=-1, init=False)
    mass: float = field(default=1, init=False)
    photon: Species | None = field(default=None, init=False)

    def set_photon(self, photon: Species):
        if self.radiation != "photons":
            raise ValueError("radiation must be 'photons'")
        assert isinstance(photon, Species)
        self.photon = photon

    def create_particles(self, ipatch: int | None=None, rank: int | None=None) -> ParticlesBase:
        if self.photon:
            if self.polarization is None:
                return QEDParticles(ipatch, rank)
            else:
                return SpinQEDParticles(ipatch, rank)
        elif self.polarization is not None:
            return SpinParticles(ipatch, rank)

        return super().create_particles(ipatch, rank)

@dataclass(kw_only=True)
class Positron(Electron):
    name: str = field(default='positron', init=True)
    charge: int = field(default=1, init=False)

@dataclass(kw_only=True)
class Proton(Species):
    name: str = field(default='proton', init=True)
    charge: int = field(default=1, init=False)
    mass: float = field(default=m_p/m_e, init=False)


@dataclass(kw_only=True)    
class Photon(Species):
    name: str = field(default='photon', init=True)
    charge: int = field(default=0, init=False)
    mass: float = field(default=0, init=False)

    pusher: Literal["boris", "photon", "boris+tbmt"] = field(default="photon", init=False)

    electron: Species | None = field(default=None, init=False)
    positron: Species | None = field(default=None, init=False)

    def set_bw_pair(self, *, electron: Species, positron: Species):
        assert isinstance(electron, Species)
        assert isinstance(positron, Species)
        self.electron = electron
        self.positron = positron

    def create_particles(self, ipatch: int | None=None, rank: int | None=None) -> ParticlesBase:
        if self.electron is not None:
            return QEDParticles(ipatch, rank)
        # else:
        #     return SpinQEDParticles(ipatch, rank)

        return super().create_particles(ipatch, rank)
