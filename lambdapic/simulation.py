from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi
from tqdm.auto import tqdm, trange

from .callback.callback import SimulationStage, callback
from .core.boundary.cpml import PMLXmax, PMLXmin, PMLYmax, PMLYmin, PMLZmax, PMLZmin
from .core.current.deposition import (
    CurrentDeposition,
    CurrentDeposition2D,
    CurrentDeposition3D,
)
from .core.fields import Fields2D, Fields3D
from .core.interpolation.field_interpolation import (
    FieldInterpolation,
    FieldInterpolation2D,
    FieldInterpolation3D,
)
from .core.maxwell.solver import MaxwellSolver, MaxwellSolver2D, MaxwellSolver3D
from .core.mpi.mpi_manager import MPIManager
from .core.patch.metis import compute_rank
from .core.patch.patch import Patch2D, Patch3D, Patches
from .core.pusher.pusher import BorisPusher, PhotonPusher, PusherBase
from .core.qed.pair_production import NonlinearPairProductionLCFA, PairProductionBase
from .core.qed.radiation import NonlinearComptonLCFA, RadiationBase
from .core.sort.particle_sort import ParticleSort2D
from .core.species import Electron, Photon, Species
from .core.utils.logger import logger, configure_logger
from .core.utils.timer import Timer


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
    log_file: Optional[str] = Field(
        None, 
        description="Log file name (default: auto-generated based on timestamp)"
    )
    truncate_log: bool = Field(
        True, 
        description="Truncate existing log file"
    )

    @model_validator(mode='after')
    def validate_nx_divisible(self):
        if self.nx % self.npatch_x != 0:
            raise ValueError(f'nx ({self.nx}) must be divisible by npatch_x ({self.npatch_x})')
        return self

    @model_validator(mode='after')
    def validate_ny_divisible(self):
        if self.ny % self.npatch_y != 0:
            raise ValueError(f'ny ({self.ny}) must be divisible by npatch_y ({self.npatch_y})')
        return self

class Simulation3DConfig(SimulationConfig):
    nz: int = Field(..., gt=0, description="Number of cells in z direction")
    dz: float = Field(..., gt=0, description="Cell size in z direction")
    npatch_z: int = Field(..., gt=0, description="Number of patches in z direction")

    @model_validator(mode='after')
    def validate_nz_divisible(self):
        if self.nz % self.npatch_z != 0:
            raise ValueError(f'nz ({self.nz}) must be divisible by npatch_z ({self.npatch_z})')
        return self


class Simulation:
    """Main simulation class for 2D Particle-In-Cell (PIC) simulations.

    Attributes:
        dimension (int): Simulation dimensionality (2 for 2D)
        nx (int): Number of cells in x direction
        ny (int): Number of cells in y direction
        dx (float): Cell size in x direction [m]
        dy (float): Cell size in y direction [m]
        dt (float): Time step [s]
        n_guard (int): Number of guard cells
        cpml_thickness (int): CPML boundary thickness in cells
        species (list[Species]): List of particle species
        patches (Patches): Collection of simulation patches
        mpi (MPIManager): MPI communication handler
    """
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
        log_file: Optional[str] = None,
        truncate_log: bool = True
    ) -> None:
        """Initialize a 2D PIC simulation.

        Args:
            nx: Number of cells in x direction
            ny: Number of cells in y direction
            dx: Cell size in x direction (meters)
            dy: Cell size in y direction (meters)
            npatch_x: Number of patches in x direction
            npatch_y: Number of patches in y direction
            dt_cfl: CFL condition factor (default: 0.95)
            n_guard: Number of guard cells (default: 3)
            cpml_thickness: CPML boundary thickness in cells (default: 6)
            log_file: Log file name (default: auto-generated based on timestamp)
            truncate_log: Whether to truncate existing log file (default: True)
        """
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
            log_file=log_file,
            truncate_log=truncate_log
        )
        self.dimension = 2
        
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

        self.species: list[Species] = []
        
        self.itime = 0
        
        # Configure logger
        configure_logger(
            sink=config.log_file,
            truncate_existing=config.truncate_log
        )
        
        logger.info("Simulation instance created")
        self.initialized = False
        
    @property
    def time(self) -> float:
        """Get the current simulation time in seconds.
        
        Returns:
            Current simulation time (itime * dt)
        """
        return self.itime * self.dt
    
    def initialize(self):
        """Initialize the simulation components.
        
        This method:
        1. Creates and distributes patches across MPI ranks
        2. Initializes fields, boundaries, and MPI communication
        3. Adds species and particles
        4. Sets up solvers, interpolators, and pushers
        5. Configures QED modules if needed
        
        Note:
            Must be called before running the simulation. Performs collective
            MPI operations and should be called on all ranks.
        """
        comm = MPIManager.get_comm()
        rank = MPIManager.get_rank()
        comm_size = MPIManager.get_size()
        
        if rank == 0:
            logger.info(f"Starting simulation initialization on {comm_size} MPI ranks")
            logger.info(f"Domain size: {self.Lx:.2e} x {self.Ly:.2e} m")
            logger.info(f"Grid: {self.nx} x {self.ny} cells")
            logger.info(f"Patches: {self.npatch_x} x {self.npatch_y}")
            logger.info(f"Patch size: {self.nx_per_patch} x {self.ny_per_patch} cells")
            logger.info(f"Time step: {self.dt:.2e} s")
        
        patches_list = [Patches(self.dimension) for _ in range(comm_size)]
        if rank == 0:
            logger.info("Creating patches on rank 0")
            patches = self.create_patches()

            logger.info("Calculating patch loads")
            patches_npart: NDArray[np.int64] = np.zeros(patches.npatches, dtype='int64')
            for ispec, s in enumerate(self.species):
                npart_ = patches.calculate_npart(s)
                logger.info(f"Species {s.name} has {npart_.sum():,} macro particles")
                patches_npart += npart_

            logger.info("Computing rank assignments")
            patches_load = patches_npart + self.nx_per_patch * self.ny_per_patch / 2
            rank_load = np.zeros(comm_size)
            ranks, npatch_per_rank = compute_rank(patches, comm_size, patches_load)

            for ipatch, (p, r) in enumerate(zip(patches.patches, ranks)):
                p.rank = r
                rank_load[r] += patches_load[ipatch]

            rank_load /= rank_load.sum()

            message = ", ".join([f"Rank {r}: {load*100:.2f}%" for r, load in enumerate(rank_load)])
            logger.info("Loads: " + message)
            
                
            logger.info("Initializing neighbor ranks")
            if self.dimension == 2:
                patches.init_neighbor_ipatch_2d()
                patches.init_neighbor_rank_2d()
            elif self.dimension == 3:
                patches.init_neighbor_ipatch_3d()
                patches.init_neighbor_rank_3d()
            
            for p in patches:
                assert p.rank is not None
                patches_list[p.rank].append(p)

        comm.Barrier()
        logger.info(f"Rank {rank}: Receiving patch info")
        self.patches: Patches = comm.scatter(patches_list, root=0)
        
        logger.info(f"Rank {rank}: Initializing neighbor indices")
        if self.dimension == 2:
            self.patches.init_neighbor_ipatch_2d()
        elif self.dimension == 3:
            self.patches.init_neighbor_ipatch_3d()

        logger.info(f"Rank {rank}: Initializing fields")
        self._init_fields()
        
        logger.info(f"Rank {rank}: Initializing MPI manager")
        self.mpi = MPIManager.create(self.patches)

        logger.info(f"Rank {rank}: Adding PML boundaries")
        self._init_pml()
        
        for s in self.species:
            npart = self.patches.add_species(s)
            logger.info(f"Rank {rank}: Adding {npart:,} macro particles to {s.name}")
            
        logger.info(f"Rank {rank}: Filling particles")
        self.patches.fill_particles()

        logger.info(f"Rank {rank}: Initializing Maxwell solver")
        self._init_maxwell_solver()
        
        logger.info(f"Rank {rank}: Initializing field interpolator")
        self._init_interpolator()
        
        logger.info(f"Rank {rank}: Initializing current depositor")
        self._init_current_depositor()
        
        logger.info(f"Rank {rank}: Initializing pushers")
        self._init_pushers()

        logger.info(f"Rank {rank}: Initializing QED modules")
        self._init_qed()
        
        self.initialized = True
        logger.success(f"Rank {rank}: Initialization complete")

        comm.Barrier()

    def _init_fields(self):
        """Initialize field arrays for each patch.
        
        Creates a Fields2D object for each patch with the correct dimensions
        and guard cells, then assigns it to the patch.
        """
        for p in self.patches:
            f = Fields2D(
                nx=self.nx_per_patch, 
                ny=self.ny_per_patch, 
                dx=self.dx,
                dy=self.dy, 
                x0=p.x0, 
                y0=p.y0, 
                n_guard=self.n_guard
            )
            p.set_fields(f)
    
    def _init_pml(self):
        """Initialize CPML boundary conditions for patches at domain edges.
        
        Adds PML boundaries to patches that are located at the simulation domain
        edges (xmin, xmax, ymin, ymax). The thickness is determined by cpml_thickness.
        """
        for p in self.patches:
            if p.ipatch_x == 0:
                p.add_pml_boundary(PMLXmin(p.fields, thickness=self.cpml_thickness))
            if p.ipatch_x == self.npatch_x - 1:
                p.add_pml_boundary(PMLXmax(p.fields, thickness=self.cpml_thickness))
            if p.ipatch_y == 0:
                p.add_pml_boundary(PMLYmin(p.fields, thickness=self.cpml_thickness))
            if p.ipatch_y == self.npatch_y - 1:
                p.add_pml_boundary(PMLYmax(p.fields, thickness=self.cpml_thickness))

        
    def create_patches(self) -> Patches:
        """Create and initialize all patches for the simulation domain.
        
        Returns:
            Collection of all patches with initialized neighbor relationships
            
        Note:
            - Creates a 2D grid of patches based on npatch_x and npatch_y
            - Initializes neighbor indices for patch communication
            - Only called on rank 0 during initialization
        """
        patches = Patches(dimension=2)
        for j in range(self.npatch_y):
            for i in range(self.npatch_x):
                index = i + j * self.npatch_x
                x0 = i * self.Lx / self.npatch_x
                y0 = j * self.Ly / self.npatch_y
                p = Patch2D(
                    rank=None, 
                    index=index, 
                    ipatch_x=i, 
                    ipatch_y=j, 
                    x0=x0, 
                    y0=y0,
                    nx=self.nx_per_patch, 
                    ny=self.ny_per_patch, 
                    dx=self.dx,
                    dy=self.dy,
                )

                
                patches.append(p)
        patches.init_rect_neighbor_index_2d(npatch_x=self.npatch_x, npatch_y=self.npatch_y)
        patches.update_lists()

        return patches
    
    def _init_maxwell_solver(self):
        """Initialize the Maxwell field solver.
        
        Creates a MaxwellSolver2D instance configured for the current patches.
        The solver handles electromagnetic field updates using the FDTD method.
        """
        self.maxwell = MaxwellSolver2D(self.patches)

    def _init_interpolator(self):
        """Initialize the field interpolator.
        
        Creates a FieldInterpolation2D instance configured for the current patches.
        The interpolator handles field interpolation from grid to particle positions.
        """
        self.interpolator = FieldInterpolation2D(self.patches)

    def _init_current_depositor(self):
        """Initialize the current deposition module.
        
        Creates a CurrentDeposition2D instance configured for the current patches.
        Handles deposition of particle currents onto the grid.
        """
        self.current_depositor = CurrentDeposition2D(self.patches)

    def add_species(self, species: Sequence[Species]):
        """Add particle species to the simulation.
        
        Args:
            species: One or more species to add to the simulation
            
        Note:
            - Automatically ensures unique species names by renaming: 
              `electron` -> `electron.1`
            - Assigns ispec indices to each species
            - Species must be added before initialization
        """
        from .utils import uniquify_species_names
        
        # Convert to list for modification
        species_list = list(species)
        
        # Directly modify the names of original species
        uniquify_species_names(self.species, species_list)
        
        # Add type checking
        for s in species_list:
            if not isinstance(s, Species):
                raise TypeError("`species` must be a sequence of Species objects")
        
        self.species.extend(species_list)

        # Assign ispec
        for ispec, s in enumerate(self.species):
            s.ispec = ispec
    
    def _init_pushers(self):
        """Initialize particle pushers for each species.
        
        Creates pusher instances based on each species' configuration:
        - "boris": Boris pusher for charged particles
        - "photon": Photon pusher for massless particles
        
        Note:
            The pushers handle particle position and momentum updates.
        """
        self.pusher: list[PusherBase] = []
        for ispec, s in enumerate(self.patches.species):
            if s.pusher == "boris":
                logger.info(f"Using Boris pusher for {s.name}")
                self.pusher.append(BorisPusher(self.patches, ispec))
            elif s.pusher == "photon":
                logger.info(f"Using Photon pusher for {s.name}")
                self.pusher.append(PhotonPusher(self.patches, ispec))

    def _init_qed(self):
        """Initialize Quantum Electrodynamics (QED) modules.
        
        Sets up radiation and pair production modules based on species configuration:
        - Nonlinear Compton radiation for high-energy particles
        - Nonlinear pair production for strong fields
        
        Note:
            Only initializes modules for species that have QED effects enabled.
        """
        self.radiation: list[RadiationBase|None] = []
        for ispec, s in enumerate(self.patches.species):
            if not hasattr(s, "radiation"):
                self.radiation.append(None)
                continue
            if hasattr(s, "radiation"):
                if s.radiation == "photons":
                    logger.info(f"Using nonlinear Compton LCFA for {s.name}")
                    self.radiation.append(NonlinearComptonLCFA(self.patches, ispec))
                elif s.radiation is None:
                    self.radiation.append(None)
                else:
                    raise ValueError(f"Unknown radiation model: {s.radiation}")
            
        self.pairproduction: list[PairProductionBase|None] = []
        for ispec, s in enumerate(self.patches.species):
            if hasattr(s, "electron") and hasattr(s, "positron"):
                if s.electron is not None and s.positron is not None:
                    logger.info(f"Using nonlinear pair production LCFA for {s.name}")
                    self.pairproduction.append(NonlinearPairProductionLCFA(self.patches, ispec))
                    continue
            self.pairproduction.append(None)

    def maxwell_stage(self):
        """Perform a single Maxwell solver stage (half time step).
        
        Updates electromagnetic fields using the FDTD method:
        1. Updates E field by 0.5*dt
        2. Synchronizes E field across patches
        3. Updates B field by 0.5*dt
        4. Synchronizes B field across patches
        """
        with Timer('update E field'):
            self.maxwell.update_efield(0.5*self.dt)
        with Timer('sync E field'):
            self.patches.sync_guard_fields(['ex', 'ey', 'ez'])
            self.mpi.sync_guard_fields(['ex', 'ey', 'ez'])
        with Timer('update B field'):
            self.maxwell.update_bfield(0.5*self.dt)
        with Timer('sync B field'):
            self.patches.sync_guard_fields(['bx', 'by', 'bz'])
            self.mpi.sync_guard_fields(['bx', 'by', 'bz'])

    def generate_lists(self):
        """Generate particle lists for all modules.
        
        Creates particle lists needed by:
        - Pushers
        - Radiation modules
        - Field interpolator
        - Current depositor
        """
        self.patches.update_lists()
        for p in self.pusher:
            p.generate_particle_lists()
        for r in self.radiation:
            if r is not None:
                r.generate_particle_lists()
        self.interpolator.generate_particle_lists()
        self.current_depositor.generate_particle_lists()

    def update_lists(self):
        """Update particle lists after particle creation/destruction.
        
        Updates particle lists for all modules when particles have been:
        - Created (e.g., through QED processes)
        - Destroyed
        - Moved between patches
        
        Also resets the 'extended' flag for all particles.
        """
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
        
        for pp in self.pairproduction:
            if pp is None:
                continue
            for ispec in range(len(self.patches.species)):
                if ispec not in [pp.ispec, pp.electron_ispec, pp.positron_ispec]:
                    continue
                for ipatch, p in enumerate(self.patches):
                    if p.particles[ispec].extended:
                        pp.update_particle_lists(ipatch)

        for ispec, s in enumerate(self.patches.species):
            for ipatch, p in enumerate(self.patches):
                p.particles[ispec].extended = False
                    
                    

    def run(self, nsteps: int, callbacks: Sequence[Callable[['Simulation'], None]]|None = None):
        """Run the simulation for a specified number of steps.
        
        Args:
            nsteps: Number of time steps to run
            callbacks: Callbacks to execute at different simulation stages
            
        Note:
            The main simulation loop performs:
            1. Field updates (Maxwell solver)
            2. Particle pushing (position and momentum)
            3. Current deposition
            4. QED processes (radiation and pair production)
            5. Particle synchronization between patches
            6. Callback execution at defined stages
        """
        if not self.initialized:
            self.initialize()
            
        stage_callbacks = SimulationCallbacks(callbacks, self)
        # check unified pusher
        stages_in_pusher = {
            "push position first",
            "interpolator",  
            "qed",     
            "push momentum", 
            "push position second",
        }
        use_unified_pusher = [False] * len(self.patches.species)
        for ispec, pusher in enumerate(self.pusher):
            if isinstance(pusher, BorisPusher) and \
                not self.radiation[ispec] and \
                not self.pairproduction[ispec] and \
                not stages_in_pusher.intersection([stage for stage, cb in stage_callbacks.stage_callbacks.items() if cb]):
                use_unified_pusher[ispec] = True
                logger.info(f"Rank {self.mpi.rank}: No callbacks in pusher stages, switching to unified pusher for {self.species[ispec].name}")

        if self.mpi.rank == 0:
            logger.info("Starting simulation")
        for self.istep in trange(nsteps, disable=self.mpi.rank>0):
            
            # start of simulation stages
            with Timer('callback start'):
                stage_callbacks.run('start')
            
            # EM from t to t+0.5dt
            with Timer('update E field'):
                self.maxwell.update_efield(0.5*self.dt)
            with Timer('sync E field'):
                self.patches.sync_guard_fields(['ex', 'ey', 'ez'])
                self.mpi.sync_guard_fields(['ex', 'ey', 'ez'])
            with Timer('update B field'):
                self.maxwell.update_bfield(0.5*self.dt)
            with Timer('sync B field'):
                self.patches.sync_guard_fields(['bx', 'by', 'bz'])
                self.mpi.sync_guard_fields(['bx', 'by', 'bz'])
                
            with Timer("maxwell first"):
                stage_callbacks.run('maxwell first')
                

            if self.current_depositor:
                self.current_depositor.reset()
            for ispec, s in enumerate(self.patches.species):
                self.ispec = ispec

                if use_unified_pusher[ispec]:
                    with Timer(f"unified pusher for {self.species[ispec].name}"):
                        self.pusher[ispec](self.dt, unified=True)
                else:
                    # position from t to t+0.5dt
                    with Timer('push_position'):
                        self.pusher[ispec].push_position(0.5*self.dt)
                        
                    with Timer("callback push_position first"):
                        stage_callbacks.run('push position first')

                    if self.interpolator:
                        with Timer(f'Interpolation for {self.species[ispec].name}'):
                            self.interpolator(ispec)
                            
                        with Timer("callback interpolator"):
                            stage_callbacks.run('interpolator')

                    with Timer(f'radiation for {self.species[ispec].name}'):
                        if self.radiation[ispec] is not None:
                            self.radiation[ispec].update_chi()
                            self.radiation[ispec].event(dt=self.dt)
                            

                    if self.pairproduction[ispec] is not None:
                        with Timer(f'pairproduction for {self.species[ispec].name}'):
                            self.pairproduction[ispec].update_chi()
                            self.pairproduction[ispec].event(dt=self.dt)
                            
                    stage_callbacks.run('qed')
                    
                    # momentum from t to t+dt
                    with Timer(f"Pushing {self.species[ispec].name}"):
                        self.pusher[ispec](self.dt)
                        
                    with Timer("callback push momentum"):
                        stage_callbacks.run('push momentum')
                    
                    # position from t+0.5t to t+dt, using new momentum
                    with Timer('push_position'):
                        self.pusher[ispec].push_position(0.5*self.dt)
                        
                    with Timer("callback push_position second"):
                        stage_callbacks.run('push position second')
                        
                    if self.current_depositor:
                        with Timer(f"Current deposition for {self.species[ispec].name}"):
                            self.current_depositor(ispec, self.dt)
                        
                with Timer("sync_currents"):
                    self.patches.sync_currents()
                    self.mpi.sync_currents()
                    
                with Timer("callback current deposition"):
                    stage_callbacks.run('current deposition')

            # set ispec to None out of species loop
            self.ispec = None
            
            # create photons after particle loop
            # TODO: the position and momentum of ele are pushed
            # before photons are created, so the photons are created
            # at t+dt. It will be fixed later.
            for ispec, s in enumerate(self.patches.species):
                with Timer(f'create photons for {self.species[ispec].name}'):
                    if self.radiation[ispec] is not None:
                        # creating photons involves two species
                        # be careful updating the lists
                        self.radiation[ispec]._update_particle_lists()
                        self.radiation[ispec].create_particles()
                        self.radiation[ispec].reaction()
                    if self.pairproduction[ispec] is not None:
                        self.pairproduction[ispec]._update_particle_lists()
                        self.pairproduction[ispec].create_particles()
                        self.pairproduction[ispec].reaction()

            for ispec, s in enumerate(self.patches.species):
                self.mpi.sync_particles(ispec)

            with Timer("sync_particles"):
                self.patches.sync_particles()

            with Timer("Updating lists"):
                self.update_lists()

            with Timer("callback qed create particles"):
                stage_callbacks.run("qed create particles")

            # EM from t to t+0.5dt
            with Timer('update B field'):
                self.maxwell.update_bfield(0.5*self.dt)
                
            with Timer("laser"):
                stage_callbacks.run('_laser')
                
            with Timer('sync B field'):
                self.patches.sync_guard_fields(['bx', 'by', 'bz'])
                self.mpi.sync_guard_fields(['bx', 'by', 'bz'])
                

            with Timer('update E field'):
                self.maxwell.update_efield(0.5*self.dt)
            with Timer('sync E field'):
                self.patches.sync_guard_fields(['ex', 'ey', 'ez'])
                self.mpi.sync_guard_fields(['ex', 'ey', 'ez'])

            with Timer("callback maxwell second"):
                stage_callbacks.run('maxwell second')
        
            self.itime += 1
        
        self.mpi.comm.Barrier()


class Simulation3D(Simulation):
    def __init__(
        self,
        nx: int, ny: int, nz: int,
        dx: float, dy: float, dz: float,
        npatch_x: int, npatch_y: int, npatch_z: int,
        dt_cfl: float = 0.95,
        n_guard: int = 3,
        cpml_thickness: int = 6,
    ) -> None:
        """Initialize a 3D PIC simulation.

        Args:
            nx: Number of cells in x direction
            ny: Number of cells in y direction
            nz: Number of cells in z direction
            dx: Cell size in x direction (meters)
            dy: Cell size in y direction (meters)
            dz: Cell size in z direction (meters)
            npatch_x: Number of patches in x direction
            npatch_y: Number of patches in y direction
            npatch_z: Number of patches in z direction
            dt_cfl: CFL condition factor (default: 0.95)
            n_guard: Number of guard cells (default: 3)
            cpml_thickness: CPML boundary thickness in cells (default: 6)
        """
        config = Simulation3DConfig(
            nx=nx, ny=ny, nz=nz,
            dx=dx, dy=dy, dz=dz,
            npatch_x=npatch_x, npatch_y=npatch_y, npatch_z=npatch_z,
            dt_cfl=dt_cfl,
            n_guard=n_guard,
            cpml_thickness=cpml_thickness,
        )
        self.dimension = 3
        
        self.nx = config.nx
        self.ny = config.ny
        self.nz = config.nz
        self.dx = config.dx
        self.dy = config.dy
        self.dz = config.dz

        self.npatch_x = config.npatch_x
        self.npatch_y = config.npatch_y
        self.npatch_z = config.npatch_z

        self.dt = config.dt_cfl * (dx**-2 + dy**-2 + dz**-2)**-0.5 / c
        self.n_guard = config.n_guard
        self.cpml_thickness = config.cpml_thickness

        self.Lx = self.nx * self.dx
        self.Ly = self.ny * self.dy
        self.Lz = self.nz * self.dz

        self.nx_per_patch = self.nx // self.npatch_x
        self.ny_per_patch = self.ny // self.npatch_y
        self.nz_per_patch = self.nz // self.npatch_z

        self.species: list[Species] = []
        
        self.maxwell = None
        self.interpolator = None
        self.current_depositor = None
        
        self.itime = 0

        self.initialized = False

    def _init_fields(self):
        """Initialize 3D field arrays for each patch.
        
        Creates a Fields3D object for each patch with the correct dimensions
        and guard cells, then assigns it to the patch.
        """
        for p in self.patches:
            f = Fields3D(
                nx=self.nx_per_patch,
                ny=self.ny_per_patch,
                nz=self.nz_per_patch,
                dx=self.dx,
                dy=self.dy,
                dz=self.dz,
                x0=p.x0,
                y0=p.y0,
                z0=p.z0,
                n_guard=self.n_guard
            )
            p.set_fields(f)
    
    def _init_pml(self):
        """Initialize 3D CPML boundary conditions.
        
        Adds PML boundaries to patches at all domain edges (xmin, xmax, ymin, ymax, zmin, zmax).
        The thickness is determined by cpml_thickness.
        """
        for p in self.patches:
            if p.ipatch_x == 0:
                p.add_pml_boundary(PMLXmin(p.fields, thickness=self.cpml_thickness))
            if p.ipatch_x == self.npatch_x - 1:
                p.add_pml_boundary(PMLXmax(p.fields, thickness=self.cpml_thickness))
            if p.ipatch_y == 0:
                p.add_pml_boundary(PMLYmin(p.fields, thickness=self.cpml_thickness))
            if p.ipatch_y == self.npatch_y - 1:
                p.add_pml_boundary(PMLYmax(p.fields, thickness=self.cpml_thickness))
            if p.ipatch_z == 0:
                p.add_pml_boundary(PMLZmin(p.fields, thickness=self.cpml_thickness))
            if p.ipatch_z == self.npatch_z - 1:
                p.add_pml_boundary(PMLZmax(p.fields, thickness=self.cpml_thickness))

    def create_patches(self):
        """Create and initialize all 3D patches for the simulation domain.
        
        Returns:
            Collection of all 3D patches with initialized neighbor relationships
            
        Note:
            - Creates a 3D grid of patches based on npatch_x, npatch_y, and npatch_z
            - Initializes neighbor indices for patch communication
            - Only called on rank 0 during initialization
        """
        patches = Patches(dimension=3)
        for k in range(self.npatch_z):
            for j in range(self.npatch_y):
                for i in range(self.npatch_x):
                    index = i + j * self.npatch_x + k*self.npatch_x*self.npatch_y
                    p = Patch3D(
                        rank=None, index=index, 
                        ipatch_x=i, ipatch_y=j, ipatch_z=k,
                        x0=i*self.Lx/self.npatch_x, y0=j*self.Ly/self.npatch_y, z0=k*self.Lz/self.npatch_z,
                        nx=self.nx_per_patch, ny=self.ny_per_patch, nz=self.nz_per_patch,
                        dx=self.dx, dy=self.dy, dz=self.dz
                    )

                    patches.append(p)

        patches.init_rect_neighbor_index_3d(self.npatch_x, self.npatch_y, self.npatch_z)
        patches.update_lists()
        return patches

    def _init_maxwell_solver(self):
        """Initialize the 3D Maxwell field solver.
        
        Creates a MaxwellSolver3D instance configured for the current patches.
        The solver handles electromagnetic field updates using the FDTD method.
        """
        self.maxwell = MaxwellSolver3D(self.patches)

    def _init_interpolator(self):
        """Initialize the 3D field interpolator.
        
        Creates a FieldInterpolation3D instance configured for the current patches.
        The interpolator handles field interpolation from grid to particle positions.
        """
        self.interpolator = FieldInterpolation3D(self.patches)

    def _init_current_depositor(self):
        """Initialize the 3D current deposition module.
        
        Creates a CurrentDeposition3D instance configured for the current patches.
        Handles deposition of particle currents onto the grid.
        """
        self.current_depositor = CurrentDeposition3D(self.patches)
        
class SimulationCallbacks:
    """Manages the execution of callbacks at different simulation stages."""
    
    def __init__(self, callbacks: Sequence[Callable[[Simulation], None]]|None, simulation):
        """Initialize the callback manager.
        
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
        """Execute all callbacks registered for a given simulation stage.
        
        Args:
            stage: The simulation stage to run callbacks for (e.g., 'start', 'maxwell first')
            
        Note:
            Calls each callback function in sequence, passing the simulation instance.
        """
        for cb in self.stage_callbacks[stage]:
            cb(self.simulation)

    def non_empty_stages(self):
        """Get stages that have registered callbacks.
        
        Returns:
            List of simulation stages that have at least one callback registered
            
        Note:
            Useful for checking which stages will trigger callback execution.
        """
        return [stage for stage, callbacks in self.stage_callbacks.items() if callbacks]
