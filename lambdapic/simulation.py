from collections.abc import Callable, Sequence
from typing import Optional

from libpic.boundary.cpml import PMLXmax, PMLXmin, PMLYmax, PMLYmin
from libpic.current.deposition import CurrentDeposition2D
from libpic.fields import Fields2D
from libpic.interpolation.field_interpolation import FieldInterpolation2D
from libpic.maxwell.solver import MaxwellSolver2d
from libpic.patch.patch import Patch2D, Patches
from libpic.pusher.pusher import BorisPusher, PhotonPusher, PusherBase
from libpic.qed.radiation import NonlinearComptonLCFA, RadiationBase
from libpic.sort.particle_sort import ParticleSort2D
from libpic.species import Species
from libpic.utils.timer import Timer
from pydantic import BaseModel, Field, validator
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi
from tqdm.auto import tqdm, trange

from .callback.callback import SimulationStage, callback


class SimulationConfig(BaseModel):
    nx: int = Field(..., gt=0, description="Number of cells in x direction")
    ny: int = Field(..., gt=0, description="Number of cells in y direction")
    dx: float = Field(..., gt=0, description="Cell size in x direction")
    dy: float = Field(..., gt=0, description="Cell size in y direction")
    npatch_x: int = Field(..., gt=0, description="Number of patches in x direction")
    npatch_y: int = Field(..., gt=0, description="Number of patches in y direction")
    dt_cfl: float = Field(0.95, gt=0, le=1, description="CFL condition factor")
    n_guard: int = Field(3, gt=0, description="Number of guard cells")
    cpml_thickness: int = Field(6, gt=0, description="CPML boundary thickness")

    @validator('nx')
    def validate_nx_divisible(cls, v, values):
        if 'npatch_x' in values and v % values['npatch_x'] != 0:
            raise ValueError(f'nx ({v}) must be divisible by npatch_x ({values["npatch_x"]})')
        return v

    @validator('ny')
    def validate_ny_divisible(cls, v, values):
        if 'npatch_y' in values and v % values['npatch_y'] != 0:
            raise ValueError(f'ny ({v}) must be divisible by npatch_y ({values["npatch_y"]})')
        return v


class Simulation:
    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        npatch_x: int,
        npatch_y: int,
        dt_cfl: float = 0.95,
        n_guard: int = 3,
        cpml_thickness: int = 6,
    ) -> None:
        config = SimulationConfig(
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            npatch_x=npatch_x,
            npatch_y=npatch_y,
            dt_cfl=dt_cfl,
            n_guard=n_guard,
            cpml_thickness=cpml_thickness,
        )
        
        self.nx = config.nx
        self.ny = config.ny
        self.dx = config.dx
        self.dy = config.dy
        self.npatch_x = config.npatch_x
        self.npatch_y = config.npatch_y
        self.dt = config.dt_cfl * (dx**-2 + dy**-2)**-0.5 / c
        self.n_guard = config.n_guard
        self.cpml_thickness = config.cpml_thickness

        self.Lx = self.nx * self.dx
        self.Ly = self.ny * self.dy

        self.nx_per_patch = self.nx // self.npatch_x
        self.ny_per_patch = self.ny // self.npatch_y

        self.create_patches()
        self.species = []
        
        self.maxwell = MaxwellSolver2d(self.patches)
        self.interpolator = None
        self.current_depositor = None
        
        self.itime = 0
        
    @property
    def time(self) -> float:
        return self.itime * self.dt
    
    
    def create_patches(self):
        self.patches = Patches(dimension=2)
        for j in range(self.npatch_y):
            for i in range(self.npatch_x):
                index = i + j * self.npatch_x
                p = Patch2D(
                    rank=0, 
                    index=index, 
                    ipatch_x=i, 
                    ipatch_y=j, 
                    x0=i*self.Lx/self.npatch_x, 
                    y0=j*self.Ly/self.npatch_y,
                    nx=self.nx_per_patch, 
                    ny=self.ny_per_patch, 
                    dx=self.dx,
                    dy=self.dy,
                )
                f = Fields2D(
                    nx=self.nx_per_patch, 
                    ny=self.ny_per_patch, 
                    dx=self.dx,
                    dy=self.dy, 
                    x0=i*self.Lx/self.npatch_x, 
                    y0=j*self.Ly/self.npatch_y, 
                    n_guard=self.n_guard
                )
                
                p.set_fields(f)

                if i > 0:
                    p.set_neighbor_index(xmin=(i - 1) + j * self.npatch_x)
                if i < self.npatch_x - 1:
                    p.set_neighbor_index(xmax=(i + 1) + j * self.npatch_x)
                if j > 0:
                    p.set_neighbor_index(ymin=i + (j - 1) * self.npatch_x)
                if j < self.npatch_y - 1:
                    p.set_neighbor_index(ymax=i + (j + 1) * self.npatch_x)

                if i == 0:
                    p.add_pml_boundary(PMLXmin(f, thickness=self.cpml_thickness))
                if i == self.npatch_x - 1:
                    p.add_pml_boundary(PMLXmax(f, thickness=self.cpml_thickness))
                if j == 0:
                    p.add_pml_boundary(PMLYmin(f, thickness=self.cpml_thickness))
                if j == self.npatch_y - 1:
                    p.add_pml_boundary(PMLYmax(f, thickness=self.cpml_thickness))

                self.patches.append(p)

        self.patches.update_lists()


    def add_species(self, species: Sequence[Species]):
        self.species.extend(species)
        for s in species:
            if isinstance(s, Species):
                self.patches.add_species(s)
            else:
                raise TypeError("`species` must be a sequence of Species objects")
        self.patches.fill_particles()
        self.patches.update_lists()

        self.interpolator = FieldInterpolation2D(self.patches)
        self.current_depositor = CurrentDeposition2D(self.patches)

        self.pusher: list[PusherBase] = []
        for ispec, s in enumerate(self.patches.species):
            if s.pusher == "boris":
                self.pusher.append(BorisPusher(self.patches, ispec))
            elif s.pusher == "photon":
                self.pusher.append(PhotonPusher(self.patches, ispec))

        self.radiation: list[RadiationBase] = []
        for ispec, s in enumerate(self.patches.species):
            if not hasattr(s, "radiation"):
                self.radiation.append(None)
                continue
            if s.radiation == "photons":
                self.radiation.append(NonlinearComptonLCFA(self.patches, ispec))
            elif s.radiation is None:
                self.radiation.append(None)
            else:
                raise ValueError(f"Unknown radiation model: {s.radiation}")

    def maxwell_stage(self):
        self.maxwell.update_efield(0.5*self.dt)
        self.patches.sync_guard_fields()
        self.maxwell.update_bfield(0.5*self.dt)
        self.patches.sync_guard_fields()

    def generate_lists(self):
        self.patches.update_lists()
        for p in self.pusher:
            p.generate_particle_lists()
        for r in self.radiation:
            if r is not None:
                r.generate_particle_lists()
        self.interpolator.generate_particle_lists()
        self.current_depositor.generate_particle_lists()

    def update_lists(self):
        for ispec, s in enumerate(self.patches.species):
            for ipatch, p in enumerate(self.patches):
                if p.particles[ispec].extended:
                    self.current_depositor.update_particle_lists(ipatch, ispec)
                    self.interpolator.update_particle_lists(ipatch, ispec)
                    self.pusher[ispec].update_particle_lists(ipatch)
        for r in self.radiation:
            if r is None:
                continue
            for ispec in range(len(self.patches.species)):
                if ispec not in [r.ispec, r.photon_ispec]:
                    continue
                for ipatch, p in enumerate(self.patches):
                    if p.particles[ispec].extended:
                        r.update_particle_lists(ipatch)

        for ispec, s in enumerate(self.patches.species):
            for ipatch, p in enumerate(self.patches):
                p.particles[ispec].extended = False
                    
                    

    def run(self, nsteps: int, callbacks: Sequence[Callable] = None):
        stage_callbacks = SimulationCallbacks(callbacks, self)

        for self.istep in trange(nsteps):
            
            # start of simulation stages
            stage_callbacks.run('start')
            
            # EM from t to t+0.5dt
            with Timer('Maxwell'):
                self.maxwell_stage()
            stage_callbacks.run('maxwell first')
                

            self.current_depositor.reset()
            for ispec, s in enumerate(self.patches.species):
                self.ispec = ispec
                # position from t to t+0.5dt
                with Timer('push_position'):
                    self.pusher[ispec].push_position(0.5*self.dt)
                
                stage_callbacks.run('push position first')

                if self.interpolator:
                    with Timer(f'Interpolation for {ispec} species'):
                        self.interpolator(ispec)
                    stage_callbacks.run('interpolator')

                if self.radiation[ispec] is not None:
                    with Timer(f'chi for {self.species[ispec].name} species'):
                        self.radiation[ispec].update_chi()
                        self.radiation[ispec].event(dt=self.dt)
                        self.radiation[ispec].reaction()
                stage_callbacks.run('qed')
                
                # momentum from t to t+dt
                with Timer(f"Pushing {ispec} species"):
                    self.pusher[ispec](self.dt)
                stage_callbacks.run('push momentum')
                
                # position from t+0.5t to t+dt, using new momentum
                with Timer('push_position'):
                    self.pusher[ispec].push_position(0.5*self.dt)
                stage_callbacks.run('push position second')
                    
                if self.current_depositor:
                    with Timer(f"Current deposition for {ispec} species"):
                        self.current_depositor(ispec, self.dt)
                    
                    with Timer("sync_currents"):
                        self.patches.sync_currents()
                    
                    stage_callbacks.run('current deposition')

            # set ispec to None out of species loop
            self.ispec = None
            
            # create photons after particle loop
            # TODO: the position and momentum of ele are pushed
            # before photons are created, so the photons are created
            # at t+dt. It will be fixed later.
            for ispec, s in enumerate(self.patches.species):
                with Timer(f'create photons for {ispec} species'):
                    if self.radiation[ispec] is not None:
                        # creating photons involves two species
                        # be careful updating the lists
                        self.radiation[ispec].create_particles()

            with Timer("sync_particles"):
                self.patches.sync_particles()

            with Timer("Updating lists"):
                self.update_lists()

            stage_callbacks.run("qed create particles")

            with Timer('Maxwell'):
                # EM from t to t+0.5dt
                self.maxwell_stage()
            stage_callbacks.run('maxwell second')
        
            self.itime += 1


class SimulationCallbacks:
    """Manages the execution of callbacks at different simulation stages."""
    
    def __init__(self, callbacks: list[Callable], simulation):
        """
        Initialize the callback manager.
        
        Args:
            callbacks: List of callback objects
            simulation: The simulation instance to pass to callbacks
        """
        self.simulation = simulation
        stage_callbacks = {stage: [] for stage in SimulationStage.all_stages()}
        
        if callbacks:
            for cb in callbacks:
                if hasattr(cb, 'stage'):
                    stage_callbacks[cb.stage].append(cb)
                else:
                    # Wrap plain functions as callbacks with default stage
                    wrapped = callback()(cb)
                    stage_callbacks[wrapped.stage].append(wrapped)

        self.stage_callbacks = stage_callbacks

    def run(self, stage: str):
        """
        Execute all callbacks registered for a given stage.
        
        Args:
            stage: The simulation stage to run callbacks for
        """
        for cb in self.stage_callbacks[stage]:
            cb(self.simulation)
