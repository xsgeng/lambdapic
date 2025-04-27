from collections.abc import Callable, Sequence

from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi
from tqdm.auto import trange

from .current.deposition import CurrentDeposition2D
from .fields import Fields2D
from .interpolation.field_interpolation import FieldInterpolation2D
from .maxwell.solver import MaxwellSolver2D
from .patch.patch import Patch2D, Patches
from .pusher.pusher import BorisPusher, PhotonPusher, PusherBase
from .sort.particle_sort import ParticleSort2D
from .species import Species
from .utils.timer import Timer


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
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.npatch_x = npatch_x
        self.npatch_y = npatch_y
        self.dt = dt_cfl * (dx**-2 + dy**-2)**-0.5 / c
        self.n_guard = n_guard

        self.Lx = self.nx * self.dx
        self.Ly = self.ny * self.dy

        self.nx_per_patch = self.nx // self.npatch_x
        self.ny_per_patch = self.ny // self.npatch_y

        self.create_patches()
        
        self.maxwell = MaxwellSolver2D(self.patches)
        

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

                self.patches.append(p)

        self.patches.update_lists()


    def add_species(self, species: Sequence[Species]):
        self.species = species
        for s in species:
            if isinstance(s, Species):
                self.patches.add_species(s)
            else:
                raise TypeError("`species` must be a sequence of Species objects")


        self.pusher: list[PusherBase] = []
        for ispec, s in enumerate(self.patches.species):
            if s.pusher == "boris":
                self.pusher.append(BorisPusher(self.patches, ispec))
            elif s.pusher == "photon":
                self.pusher.append(PhotonPusher(self.patches, ispec))
            
        self.patches.fill_particles()
        self.patches.update_lists()

        self.interpolator = FieldInterpolation2D(self.patches)
        self.current_depositor = CurrentDeposition2D(self.patches)


    def run(self, nsteps: int, callback: Sequence[Callable[[int], None]] = None, callback_species: Sequence[Callable[[int, int], None]] = None):
        for it in trange(nsteps):
            # EM from t to t+0.5dt
            with Timer('Maxwell'):
                self.maxwell.update_efield(0.5*self.dt)
                self.patches.sync_guard_fields()
                self.maxwell.update_bfield(0.5*self.dt)

            self.current_depositor.reset()
            for ispec, s in enumerate(self.patches.species):
                # position from t to t+0.5dt
                with Timer('push_position'):
                    self.pusher[ispec].push_position(0.5*self.dt)

                with Timer(f'Interpolation for {ispec} species'):
                    self.interpolator(ispec)

                # momentum from t to t+dt
                with Timer(f"Pushing {ispec} species"):
                    self.pusher[ispec](self.dt)
                # position from t+0.5t to t+dt, using new momentum
                with Timer('push_position'):
                    self.pusher[ispec].push_position(0.5*self.dt)
                
                with Timer(f"Current deposition for {ispec} species"):
                    self.current_depositor(ispec, self.dt)

                with Timer("sync_currents"):
                    self.patches.sync_currents()

                if callback_species:
                    for cb in callback_species:
                        cb(it, ispec)
            
            with Timer("sync_particles"):
                self.patches.sync_particles()
                
            with Timer("Updating lists"):
                for ispec, s in enumerate(self.patches.species):
                    for ipatch, p in enumerate(self.patches):
                        if p.particles[ispec].extended:
                            self.current_depositor.update_particle_lists(ipatch, ispec)
                            self.interpolator.update_particle_lists(ipatch, ispec)
                            self.pusher[ispec].update_particle_lists(ipatch)
                            p.particles[ispec].extended = False

                
            with Timer('Maxwell'):
                # EM from t to t+0.5dt
                self.maxwell.update_bfield(0.5*self.dt)
                self.patches.sync_guard_fields()
                self.maxwell.update_efield(0.5*self.dt)
                self.patches.sync_guard_fields()
            
            if callback:
                for cb in callback:
                    cb(it)