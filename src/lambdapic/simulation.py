import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Sequence

import mpi4py
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi
from tqdm.auto import tqdm, trange
from yaspin import yaspin

from .core.boundary.cpml import PMLXmax, PMLXmin, PMLYmax, PMLYmin, PMLZmax, PMLZmin
from .core.collision.collision import Collision
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
from .core.sort.particle_sort import ParticleSort2D, ParticleSort3D
from .core.species import Electron, Photon, Species
from .core.utils.logger import configure_logger, logger, rank_log
from .core.utils.timer import Timer
from .utils import (
    auto_patch_2d,
    auto_patch_3d,
    check_newer_version_on_pypi,
    get_num_threads,
    is_version_outdated,
)


class SimulationConfig(BaseModel):
    nx: int = Field(..., gt=0, description="Number of cells in x direction")
    ny: int = Field(..., gt=0, description="Number of cells in y direction")
    dx: float = Field(..., gt=0, description="Cell size in x direction")
    dy: float = Field(..., gt=0, description="Cell size in y direction")
    npatch_x: int = Field(..., gt=0, description="Number of patches in x direction")
    npatch_y: int = Field(..., gt=0, description="Number of patches in y direction")
    nsteps: int | None = Field(None, gt=0, description="Number of simulation steps")
    sim_time: float | None = Field(None, gt=0, description="Simulation time in seconds")
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
    boundary_conditions: Dict[Literal['xmin', 'xmax', 'ymin', 'ymax'], Literal['pml', 'periodic']] = Field(
        {'xmin': 'pml', 'xmax': 'pml', 'ymin': 'pml', 'ymax': 'pml'}, 
        description="Boundary conditions for each side of the domain. Supported values: 'pml', 'periodic'"
    )
    random_seed: Optional[int] = Field(
        None,
        description="Random seed for reproducible particle initialization"
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

    @model_validator(mode='after')
    def validate_nsteps_sim_time_mutual_exclusion(self):
        """Validate that nsteps and sim_time are not both provided."""
        if self.nsteps is not None and self.sim_time is not None:
            raise ValueError("Cannot specify both nsteps and sim_time. Use only one.")
        return self


class Simulation3DConfig(SimulationConfig):
    nz: int = Field(..., gt=0, description="Number of cells in z direction")
    dz: float = Field(..., gt=0, description="Cell size in z direction")
    npatch_z: int = Field(..., gt=0, description="Number of patches in z direction")
    boundary_conditions: Dict[Literal['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'], Literal['pml', 'periodic']] = Field(
        {'xmin': 'pml', 'xmax': 'pml', 'ymin': 'pml', 'ymax': 'pml', 'zmin': 'pml', 'zmax': 'pml'}, 
        description="Boundary conditions for each side of the domain. Supported values: 'pml', 'periodic'"
    )

    @model_validator(mode='after')
    def validate_nz_divisible(self):
        if self.nz % self.npatch_z != 0:
            raise ValueError(f'nz ({self.nz}) must be divisible by npatch_z ({self.npatch_z})')
        return self

@dataclass
class Simulation:
    """Main simulation class for 2D Particle-In-Cell (PIC) simulations.

    Parameters:
        nx (int): Number of grid cells in x direction. Must be divisible by npatch_x.
        ny (int): Number of grid cells in y direction. Must be divisible by npatch_y.
        dx (float): Grid cell size in x direction (meters).
        dy (float): Grid cell size in y direction (meters).
        npatch_x (int): Number of patches to divide the domain into along x direction. Default 0 for auto patch number.
        npatch_y (int): Number of patches to divide the domain into along y direction. Default 0 for auto patch number.
        nsteps (int, optional): Number of simulation steps. Mutually exclusive with sim_time.
        sim_time (float, optional): Total simulation time in seconds. Mutually exclusive with nsteps.
        dt_cfl (float, optional): CFL (Courant-Friedrichs-Lewy) stability factor. Must be ≤ 1.0.
            The actual time step is calculated as dt = dt_cfl / (c * sqrt(1/dx² + 1/dy²)). Defaults to 0.95.
        n_guard (int, optional): Number of guard cells used for field synchronization between patches. Defaults to 3.
        boundary_conditions (Dict[Literal['xmin', 'xmax', 'ymin', 'ymax'], Literal['pml', 'periodic']], optional):
            Dictionary mapping boundary names to their conditions. Supported boundaries: 'xmin', 'xmax', 'ymin', 'ymax'.
            Supported conditions: 'pml' (Perfectly Matched Layer) or 'periodic'. Defaults to all boundaries set to 'pml'.
        cpml_thickness (int, optional): Thickness of CPML (Convolutional PML) absorbing boundary layers in grid cells. Defaults to 6.
        log_file (str, optional): Path to log file. If None, generates timestamp-based filename. Defaults to None.
        truncate_log (bool, optional): Whether to truncate existing log file or append to it. Defaults to True.
        random_seed (int, optional): Random seed for reproducible particle initialization (default: None)
        comm (mpi4py.MPI.Comm, optional): MPI communicator. If None, uses MPI.COMM_WORLD. Defaults to None.
    """
    nx: int
    ny: int
    nz: int = field(init=False)
    dx: float
    dy: float
    dz: float = field(init=False)
    npatch_x: int = field(default=0)
    npatch_y: int = field(default=0)
    npatch_z: int = field(init=False)
    nsteps: int | None = field(default=None)
    sim_time: float | None = field(default=None)
    dt_cfl: float  = field(default=0.95)
    n_guard: int = field(default=3)
    boundary_conditions: Dict[Literal['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'], Literal['pml', 'periodic']] = field(default_factory=lambda:{
        'xmin': 'pml',
        'xmax': 'pml',
        'ymin': 'pml',
        'ymax': 'pml',
    })
    cpml_thickness: int = field(default=6)
    log_file: Optional[str] = field(default=None)
    truncate_log: bool = field(default=True)
    random_seed: Optional[int] = field(default=None)
    comm: Optional[mpi4py.MPI.Comm] = field(default=None)

    _auto_patch_factor: int = field(default=4, init=False)

    def _validate(self):
        self.dimension = 2

        config = SimulationConfig(
            nx=self.nx,
            ny=self.ny,
            dx=self.dx,
            dy=self.dy,
            npatch_x=self.npatch_x,
            npatch_y=self.npatch_y,
            nsteps=self.nsteps,
            sim_time=self.sim_time if hasattr(self, 'sim_time') else None,
            dt_cfl=self.dt_cfl,
            n_guard=self.n_guard,
            boundary_conditions=self.boundary_conditions,
            cpml_thickness=self.cpml_thickness,
            log_file=self.log_file,
            truncate_log=self.truncate_log,
            random_seed=self.random_seed
        )
        
        self.nx = config.nx
        self.ny = config.ny
        self.dx = config.dx
        self.dy = config.dy
        self.npatch_x = config.npatch_x
        self.npatch_y = config.npatch_y
        self.dt = config.dt_cfl * (self.dx**-2 + self.dy**-2)**-0.5 / c

        self.Lx = self.nx * self.dx
        self.Ly = self.ny * self.dy

        self.nx_per_patch = self.nx // self.npatch_x
        self.ny_per_patch = self.ny // self.npatch_y
    
        return config
    
    def __post_init__(self,) -> None:
        self._auto_patch()

        config = self._validate()

        self.nsteps = config.nsteps
        self.sim_time = config.sim_time
        self.n_guard = config.n_guard
        self.boundary_conditions = config.boundary_conditions
        self.cpml_thickness = config.cpml_thickness

        self.species: list[Species] = []
        
        self.itime = 0
        self.random_seed = config.random_seed
        self.rand_gen: Optional[np.random.Generator] = None # will be initialized after mpi initialization

        # Configure logger
        configure_logger(
            sink=config.log_file,
            truncate_existing=config.truncate_log
        )
        
        logger.info("Simulation instance created")
        self.initialized = False
        # Collision system placeholders; configured via add_collision()
        self._collision_groups: Optional[Sequence[Sequence[Species]]] = None
        self.collision: Optional[Collision] = None
        
    def _auto_patch(self):
        if self.npatch_x == 0 or self.npatch_y == 0:
            num_threades = get_num_threads()
            if self.comm is None:
                comm = MPIManager.get_default_comm()
                rank = MPIManager.get_default_rank()
            else:
                comm = self.comm
                rank = comm.Get_rank()
            
            num_threades = comm.reduce(num_threades)
            if rank == 0:
                assert isinstance(num_threades, int)
                npatch_x, npatch_y = auto_patch_2d(self.nx, self.ny, self.n_guard, self.cpml_thickness, num_threades*self._auto_patch_factor)
            else:
                npatch_x, npatch_y = None, None

            self.npatch_x, self.npatch_y = comm.bcast((npatch_x, npatch_y))

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
        if self.comm is None:
            comm = MPIManager.get_default_comm()
            rank = MPIManager.get_default_rank()
            comm_size = MPIManager.get_defailt_size()
        else:
            comm = self.comm
            rank = comm.Get_rank()
            comm_size = comm.Get_size()
        
        if rank == 0:
            logger.info(f"Starting simulation initialization on {comm_size} MPI ranks")
            logger.info(f"Domain size: {self.Lx:.2e} x {self.Ly:.2e}" + (f" x {self.Lz:.2e}" if self.dimension == 3 else "") + " m")
            logger.info(f"Grid: {self.nx} x {self.ny}" + (f" x {self.nz}" if self.dimension == 3 else ""))
            logger.info(f"Patches: {self.npatch_x} x {self.npatch_y}" + (f" x {self.npatch_z}" if self.dimension == 3 else ""))
            logger.info(f"Patch size: {self.nx_per_patch} x {self.ny_per_patch}" + (f" x {self.nz_per_patch}" if self.dimension == 3 else "") + " cells")
            logger.info(f"Time step: {self.dt:.2e} s")
            logger.info(f"Guard cells: {self.n_guard}")
            logger.info(f"Boundary conditions: {self.boundary_conditions}")
            logger.info(f"CPML thickness: {self.cpml_thickness}")
        
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
        rank_log("Receiving patch info", comm)
        self.patches: Patches = comm.scatter(patches_list, root=0)

        self._set_global_domain_bounds()
        
        rank_log("Initializing neighbor indices", comm)
        if self.dimension == 2:
            self.patches.init_neighbor_ipatch_2d()
        elif self.dimension == 3:
            self.patches.init_neighbor_ipatch_3d()

        rank_log("Initializing fields", comm)
        self._init_fields()
        
        rank_log("Initializing MPI manager", comm)
        self.mpi = MPIManager.create(self.patches, self.comm)

        rank_log("Adding PML boundaries", comm)
        self._init_pml()
        
        for s in self.species:
            npart = self.patches.add_species(s, aux_attrs=s._aux_attrs)
            logger.info(f"Rank {rank}: Adding {npart:,} macro particles to {s.name}")
            
                
        rank_log("Creating random generators", comm)
        self._init_random_generator()
        
        rank_log("Filling particles", comm)
        self.patches.fill_particles(self.rand_gen)

        rank_log("Initializing Maxwell solver", comm)
        self._init_maxwell_solver()
        
        rank_log("Initializing field interpolator", comm)
        self._init_interpolator()
        
        rank_log("Initializing current depositor", comm)
        self._init_current_depositor()
        
        rank_log("Initializing pushers", comm)
        self._init_pushers()

        rank_log("Initializing QED modules", comm)
        self._init_qed()

        rank_log("Initializing particle sorter", comm)
        self._init_sorter()
        
        # Initialize collision module if groups were registered
        rank_log("Initializing collision module", comm)
        self._init_collision()

        self.initialized = True
        logger.success(f"Rank {rank}: Initialization complete")

        comm.Barrier()

    def _set_global_domain_bounds(self):
        """Set global domain bounds on patches."""
        self.patches.xmin_global = -self.dx/2
        self.patches.xmax_global = self.Lx - self.dx/2  
        self.patches.ymin_global = -self.dy/2
        self.patches.ymax_global = self.Ly - self.dy/2

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
            if p.ipatch_x == 0 and self.boundary_conditions['xmin'] == 'pml':
                p.add_pml_boundary(PMLXmin(p.fields, thickness=self.cpml_thickness))
            if p.ipatch_x == self.npatch_x - 1 and self.boundary_conditions['xmax'] == 'pml':
                p.add_pml_boundary(PMLXmax(p.fields, thickness=self.cpml_thickness))
            if p.ipatch_y == 0 and self.boundary_conditions['ymin'] == 'pml':
                p.add_pml_boundary(PMLYmin(p.fields, thickness=self.cpml_thickness))
            if p.ipatch_y == self.npatch_y - 1 and self.boundary_conditions['ymax'] == 'pml':
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
        patches.init_rect_neighbor_index_2d(npatch_x=self.npatch_x, npatch_y=self.npatch_y, boundary_conditions=self.boundary_conditions)
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

    def add_collision(self, collision_groups: Sequence[Sequence[Species]]):
        """
        Register particle collision groups for the simulation.

        Parameters:
            collision_groups: A sequence of groups, where each group is a sequence of Species. 
                All unique pairs within a group (including intra-species) are considered for collisions. 

        Example:
            - [[e1, e1, e2, e3], [ion, ion]] will perform e1<->e1, e1<->e2, e1<->e3, e2<->e3 collisions, and ion<->ion collisions.
            - [[e1, e1], [e1, ion], [e2, ion], [ion, ion]]. This is manually specifying all collision pairs.
            
        Notes:
            - Species in the groups must already be added to the simulation via
              add_species().
            - If called after the simulation has already been initialized, this
              will immediately construct the Collision object.
        """
        # Basic validation
        if not isinstance(collision_groups, Sequence):
            raise TypeError("collision_groups must be a sequence of species groups")

        # Flatten to validate items and ensure species are known to the simulation
        flat: list[Species] = []
        for group in collision_groups:
            for s in group:
                if not isinstance(s, Species):
                    raise TypeError("Collision groups must contain Species instances")
                flat.append(s)

        # Ensure all species exist in the simulation registry
        for s in flat:
            if s not in self.species:
                raise ValueError(f"Species {getattr(s, 'name', str(s))} not added to simulation")

        # Store groups; actual Collision is created in _init_collision
        self._collision_groups = collision_groups

        # If already initialized, construct the Collision module now
        if self.initialized:
            self._init_collision()
        return None

    def _init_collision(self) -> None:
        """Create the Collision module after initialization.

        Requires that patches, sorter, and random generators are ready. This is
        called during initialize() after sorter setup. If no collision groups were
        registered, this is a no-op.
        """
        # No collision groups configured; nothing to do
        if not self._collision_groups:
            self.collision = None
            return

        # Construct collision module and prepare its lists
        assert self.rand_gen
        self.collision = Collision(self._collision_groups, self.patches, self.sorter, self.rand_gen)
        self.collision.generate_particle_lists()
        self.collision.generate_field_lists()
    
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
                rank_log(f"Using Boris pusher for {s.name}", self.mpi.comm)
                self.pusher.append(BorisPusher(self.patches, ispec))
            elif s.pusher == "photon":
                rank_log(f"Using Photon pusher for {s.name}", self.mpi.comm)
                self.pusher.append(PhotonPusher(self.patches, ispec))

    def _init_qed(self):
        """Initialize Quantum Electrodynamics (QED) modules.
        
        Sets up radiation and pair production modules based on species configuration:
        - Nonlinear Compton radiation for high-energy particles
        - Nonlinear pair production for strong fields
        
        Note:
            Only initializes modules for species that have QED effects enabled.
        """
        self.radiation: List[RadiationBase|None] = []
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
            
        self.pairproduction: List[PairProductionBase|None] = []
        for ispec, s in enumerate(self.patches.species):
            if hasattr(s, "electron") and hasattr(s, "positron"):
                if s.electron is not None and s.positron is not None:
                    logger.info(f"Using nonlinear pair production LCFA for {s.name}")
                    self.pairproduction.append(NonlinearPairProductionLCFA(self.patches, ispec))
                    continue
            self.pairproduction.append(None)

    def _init_sorter(self):
        self.sorter: list[ParticleSort2D] = []
        if self._collision_groups:
            for s in self.patches.species:
                self.sorter.append(ParticleSort2D(self.patches, s))
        else:
            for s in self.patches.species:
                self.sorter.append(ParticleSort2D(self.patches, s, ny_buckets=1, dy_buckets=self.Ly))

    def _init_random_generator(self) -> None:
        """Create MPI-level generators for each rank.
        
        Returns:
            List of generators for each MPI rank
        """
        if self.random_seed is None:
            self.rand_gen = np.random.default_rng()
            return
        
        if self.mpi.rank == 0:
            master_gen = np.random.default_rng(self.random_seed)
            gens = master_gen.spawn(self.mpi.size)
        else:
            gens = None
            
        self.rand_gen = self.mpi.comm.scatter(gens, root=0)


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
                    self.sorter[ispec].update_particle_lists(ipatch)

        if self.collision is not None:
            self.collision.update_particle_lists()

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
                    
                    

    def run(self, nsteps: int|None = None, sim_time: float|None = None, callbacks: Optional[Sequence[Callable[['Simulation'], None]]] = None,
            stop_callback: Callable[..., bool] = lambda: False,):
        """Run the simulation for a specified number of steps or time duration.
        
        Args:
            nsteps: Number of time steps to run (mutually exclusive with sim_time)
            sim_time: Total simulation time in seconds (mutually exclusive with nsteps)
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
        
        if callbacks is None:
            callbacks = []
        stage_callbacks = SimulationCallbacks(callbacks, self)

        # check restart
        restart_cb = None
        for cb in callbacks:
            if hasattr(cb, "__class__") and cb.__class__.__name__ == "RestartDump":
                restart_cb = cb
                break

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
                rank_log(f"No callbacks in pusher stages, switching to unified pusher for {self.species[ispec].name}", self.mpi.comm)

        if self.mpi.rank == 0 and os.environ.get("LAMBDAPIC_CHECK_UPDATE", "1") == "1":
            with yaspin(text="Checking for newer version on PyPI. Disable with LAMBDAPIC_CHECK_UPDATE=0"):
                current_version, latest_version = check_newer_version_on_pypi()
            if current_version and latest_version and is_version_outdated(current_version, latest_version):
                logger.info(f"New version available: {current_version} -> {latest_version}. Upgrade with `pip install --upgrade --upgrade-strategy=only-if-needed lambdapic`")
        
        # handle simulation steps
        nsteps_total = self._handle_nsteps(nsteps, sim_time)
            
        self.mpi.comm.Barrier()
        for self.istep in trange(nsteps_total-self.itime, total=nsteps_total, initial=self.itime, 
                                 disable=self.mpi.rank>0, position=1):
            
            # start of simulation stages
            with Timer('Callbacks: start stage'):
                stage_callbacks.run('start')
            
            # EM from t to t+0.5dt
            with Timer('update E field'):
                self.maxwell.update_efield(0.5*self.dt)
            with Timer('mpi sync E field'):
                self.mpi.sync_guard_fields(['ex', 'ey', 'ez'])
            with Timer('sync E field'):
                self.patches.sync_guard_fields(['ex', 'ey', 'ez'])
            with Timer('update B field'):
                self.maxwell.update_bfield(0.5*self.dt)
            with Timer('mpi sync B field'):
                self.mpi.sync_guard_fields(['bx', 'by', 'bz'])
            with Timer('sync B field'):
                self.patches.sync_guard_fields(['bx', 'by', 'bz'])
                
            with Timer("maxwell first"):
                stage_callbacks.run('maxwell first')
                
            for ispec, s in enumerate(self.patches.species):
                self.ispec = ispec

                with Timer(f"Sorting {self.species[ispec].name}"):
                    self.sorter[ispec]()

            if self.collision is not None:
                with Timer("collision.calculate_debye_length"):
                    self.collision.calculate_debye_length()
                with Timer("collision"):
                    self.collision(self.dt)

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
                        
                    with Timer("Callbacks: push position first stage"):
                        stage_callbacks.run('push position first')

                    if self.interpolator:
                        with Timer(f'Interpolation for {self.species[ispec].name}'):
                            self.interpolator(ispec)
                            
                        with Timer("Callbacks: interpolator stage"):
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
                        
                    with Timer("Callbacks: push momentum stage"):
                        stage_callbacks.run('push momentum')
                    
                    # position from t+0.5t to t+dt, using new momentum
                    with Timer('push_position'):
                        self.pusher[ispec].push_position(0.5*self.dt)
                        
                    with Timer("Callbacks: push position second stage"):
                        stage_callbacks.run('push position second')
                        
                    if self.current_depositor:
                        with Timer(f"Current deposition for {self.species[ispec].name}"):
                            self.current_depositor(ispec, self.dt)
                        
                with Timer("sync_currents"):
                    self.patches.sync_currents()
                with Timer("mpi.sync_currents"):
                    self.mpi.sync_currents()
                    
                with Timer("Callbacks: current deposition stage"):
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

            with Timer("mpi.sync_particles"):
                for ispec, s in enumerate(self.patches.species):
                    self.mpi.sync_particles(ispec)

            with Timer("sync_particles"):
                self.patches.sync_particles()

            with Timer("Updating lists"):
                self.update_lists()

            with Timer("Callbacks: qed create particles stage"):
                stage_callbacks.run("qed create particles")

            # EM from t to t+0.5dt
            with Timer('update B field'):
                self.maxwell.update_bfield(0.5*self.dt)
                
            with Timer("laser"):
                stage_callbacks.run('_laser')
                
            with Timer('mpi sync B field'):
                self.mpi.sync_guard_fields(['bx', 'by', 'bz'])
            with Timer('sync B field'):
                self.patches.sync_guard_fields(['bx', 'by', 'bz'])
                

            with Timer('update E field'):
                self.maxwell.update_efield(0.5*self.dt)
            with Timer('mpi sync E field'):
                self.mpi.sync_guard_fields(['ex', 'ey', 'ez'])
            with Timer('sync E field'):
                self.patches.sync_guard_fields(['ex', 'ey', 'ez'])

            with Timer("Callbacks: maxwell second stage"):
                stage_callbacks.run('maxwell second')
        
            if restart_cb and restart_cb._dump_requested:
                restart_cb._call(self)
                return
            
            self.itime += 1


            if stop_callback():
                return "stop by callback"
        
        self.mpi.comm.Barrier()

    def _handle_nsteps(self, nsteps, sim_time):
        if nsteps is not None and sim_time is not None:
            raise ValueError("Cannot specify both nsteps and sim_time in run() method")
        # Determine number of steps to run based on nsteps or sim_time
        elif nsteps is None and sim_time is None:
            if self.nsteps is not None:
                nsteps_total = self.nsteps
            elif self.sim_time is not None:
                # Calculate nsteps from simulation time
                nsteps_total = int(self.sim_time / self.dt)
            else:
                raise ValueError("Must provide either nsteps or sim_time, either in Simulation or as an argument to run()")
        elif sim_time is not None and nsteps is None:
            # Calculate nsteps from simulation time parameter
            nsteps_total = int(sim_time / self.dt)
        elif nsteps is not None and sim_time is None:
            nsteps_total = nsteps + self.itime
        else:
            raise RuntimeError("This should never be reached")
        return nsteps_total

Simulation2D = Simulation

@dataclass
class Simulation3D(Simulation):
    """Main class for 3D PIC simulation.
    
    Parameters:
        nx (int): Number of grid cells in x direction. Must be divisible by npatch_x.
        ny (int): Number of grid cells in y direction. Must be divisible by npatch_y.
        nz (int): Number of grid cells in z direction. Must be divisible by npatch_z.
        dx (float): Grid cell size in x direction (meters).
        dy (float): Grid cell size in y direction (meters).
        dz (float): Grid cell size in z direction (meters).
        npatch_x (int): Number of patches to divide the domain into along x direction. Default 0 for auto patch number.
        npatch_y (int): Number of patches to divide the domain into along y direction. Default 0 for auto patch number.
        npatch_z (int): Number of patches to divide the domain into along z direction. Default 0 for auto patch number.
        nsteps (int, optional): Number of simulation steps. Mutually exclusive with sim_time.
        sim_time (float, optional): Total simulation time in seconds. Mutually exclusive with nsteps.
        dt_cfl (float, optional): CFL (Courant-Friedrichs-Lewy) stability factor. Must be ≤ 1.0.
            The actual time step is calculated as dt = dt_cfl / (c * sqrt(1/dx² + 1/dy² + 1/dz²)). Defaults to 0.95.
        n_guard (int, optional): Number of guard cells used for field synchronization between patches. Defaults to 3.
        boundary_conditions (Dict[Literal['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'], Literal['pml', 'periodic']], optional): 
            Dictionary mapping boundary names to their conditions. Supported boundaries: 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'.
            Supported conditions: 'pml' (Perfectly Matched Layer) or 'periodic'. Defaults to all boundaries set to 'pml'.
        cpml_thickness (int, optional): Thickness of CPML (Convolutional PML) absorbing boundary layers in grid cells. Defaults to 6.
        log_file (Optional[str], optional): Path to log file. If None, generates timestamp-based filename. Defaults to None.
        truncate_log (bool, optional): Whether to truncate existing log file or append to it. Defaults to True.
        random_seed (int, optional): Random seed for reproducible particle initialization (default: None)
        comm (Optional[mpi4py.MPI.Comm], optional): MPI communicator. If None, uses MPI.COMM_WORLD. Defaults to None.
    """
    nz: int = field(init=True)
    dz: float = field(init=True)
    npatch_z: int = field(default=0, init=True)
    boundary_conditions: Dict[Literal['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'], Literal['pml', 'periodic']] = field(default_factory=lambda:{
        'xmin': 'pml',
        'xmax': 'pml',
        'ymin': 'pml',
        'ymax': 'pml',
        'zmin': 'pml',
        'zmax': 'pml',
    })

    def _validate(self):
        config = Simulation3DConfig(
            nx=self.nx, ny=self.ny, nz=self.nz,
            dx=self.dx, dy=self.dy, dz=self.dz,
            npatch_x=self.npatch_x, npatch_y=self.npatch_y, npatch_z=self.npatch_z,
            nsteps=self.nsteps,
            sim_time=self.sim_time,
            dt_cfl=self.dt_cfl,
            n_guard=self.n_guard,
            cpml_thickness=self.cpml_thickness,
            boundary_conditions=self.boundary_conditions,
            log_file=self.log_file,
            truncate_log=self.truncate_log,
            random_seed=self.random_seed
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

        self.dt = config.dt_cfl * (self.dx**-2 + self.dy**-2 + self.dz**-2)**-0.5 / c

        self.Lx = self.nx * self.dx
        self.Ly = self.ny * self.dy
        self.Lz = self.nz * self.dz

        self.nx_per_patch = self.nx // self.npatch_x
        self.ny_per_patch = self.ny // self.npatch_y
        self.nz_per_patch = self.nz // self.npatch_z

        return config
    
    def _auto_patch(self):
        if self.npatch_x == 0 or self.npatch_y == 0 or self.npatch_z == 0:
            num_threades = get_num_threads()
            if self.comm is None:
                comm = MPIManager.get_default_comm()
                rank = MPIManager.get_default_rank()
            else:
                comm = self.comm
                rank = comm.Get_rank()
            
            num_threades = comm.reduce(num_threades)
            if rank == 0:
                assert isinstance(num_threades, int)
                npatch_x, npatch_y, npatch_z = auto_patch_3d(self.nx, self.ny, self.nz, self.n_guard, self.cpml_thickness, num_threades*self._auto_patch_factor)
            else:
                npatch_x, npatch_y, npatch_z = None, None, None

            self.npatch_x, self.npatch_y, self.npatch_z = comm.bcast((npatch_x, npatch_y, npatch_z))

    def _set_global_domain_bounds(self):
        """Set global domain bounds on patches."""
        self.patches.xmin_global = -self.dx/2
        self.patches.xmax_global = self.Lx - self.dx/2
        self.patches.ymin_global = -self.dy/2
        self.patches.ymax_global = self.Ly - self.dy/2
        self.patches.zmin_global = -self.dz/2
        self.patches.zmax_global = self.Lz - self.dz/2

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
            if p.ipatch_x == 0 and self.boundary_conditions['xmin'] == 'pml':
                p.add_pml_boundary(PMLXmin(p.fields, thickness=self.cpml_thickness))
            if p.ipatch_x == self.npatch_x - 1 and self.boundary_conditions['xmax'] == 'pml':
                p.add_pml_boundary(PMLXmax(p.fields, thickness=self.cpml_thickness))
            if p.ipatch_y == 0 and self.boundary_conditions['ymin'] == 'pml':
                p.add_pml_boundary(PMLYmin(p.fields, thickness=self.cpml_thickness))
            if p.ipatch_y == self.npatch_y - 1 and self.boundary_conditions['ymax'] == 'pml':
                p.add_pml_boundary(PMLYmax(p.fields, thickness=self.cpml_thickness))
            if p.ipatch_z == 0 and self.boundary_conditions['zmin'] == 'pml':
                p.add_pml_boundary(PMLZmin(p.fields, thickness=self.cpml_thickness))
            if p.ipatch_z == self.npatch_z - 1 and self.boundary_conditions['zmax'] == 'pml':
                p.add_pml_boundary(PMLZmax(p.fields, thickness=self.cpml_thickness))

    def _init_sorter(self):
        self.sorter: list[ParticleSort3D] = []
        if self._collision_groups:
            for s in self.patches.species:
                self.sorter.append(ParticleSort3D(self.patches, s))
        else:
            for s in self.patches.species:
                self.sorter.append(ParticleSort3D(self.patches, s, ny_buckets=1, dy_buckets=self.Ly, nz_buckets=1, dz_buckets=self.Lz))

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

        patches.init_rect_neighbor_index_3d(self.npatch_x, self.npatch_y, self.npatch_z, boundary_conditions=self.boundary_conditions)
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
    
    def __init__(self, callbacks: Sequence[Callable[[Simulation], None]], simulation):
        """Initialize the callback manager.
        
        Args:
            callbacks: List of callback objects
            simulation: The simulation instance to pass to callbacks
        """
        from .callback.callback import SimulationStage, callback

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
