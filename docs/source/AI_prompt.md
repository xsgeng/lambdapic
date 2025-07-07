# λpic Simulation Reference for AI

## Quick Start Templates

### Basic 2D Simulation
```python
from lambdapic import Simulation, Electron, SaveFieldsToHDF5

# Initialize simulation
sim = Simulation(
    nx=512, ny=512, 
    dx=0.8e-6/20, dy=0.8e-6/20,
    npatch_x=8, npatch_y=8
)

# Add species
electron = Electron(
    density=lambda x,y: 1e18,  # 1e18 cm^-3 uniform plasma
    ppc=8
)
sim.add_species(electron)

# Run with diagnostics
sim.run(1000, callbacks=[
    SaveFieldsToHDF5('fields', interval=50)
])
```

### Common Operations
```python
# Laser initialization
from lambdapic import GaussianLaser2D
laser = GaussianLaser2D(
    a0=5, w0=5e-6, focus_position=20e-6
)

# Custom field modification
@callback('interpolator')
def add_fields(sim):
    for p in sim.patches:
        p.fields.bz[:] += 10.0  # Add 10T Bz field
```

## API reference

### Simulation
```python
class Simulation:
"""Main simulation class for 2D Particle-In-Cell (PIC) simulations."""
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
```

### Species

```python
class Species(BaseModel):
    name: str = Field(..., description="Name of the particle species")
    charge: int = Field(..., description="Charge number (e.g. -1 for electron, +1 for proton)")
    mass: float = Field(..., description="Mass in units of electron mass")

    density: Optional[Callable] = Field(None, description="Function defining particle density distribution")
    density_min: float = Field(0, description="Minimum density threshold")
    ppc: int = Field(0, description="Particles per cell")

    momentum: tuple[Optional[Callable], Optional[Callable], Optional[Callable]] = Field(
        (None, None, None), 
        description="Tuple of functions defining momentum distribution in x,y,z directions"
    )
    polarization: Optional[tuple[float, float, float]] = Field(
        None, 
        description="Polarization vector (x,y,z components) for spin particles"
    )

    pusher: Literal["boris", "photon", "boris+tbmt"] = Field(
        "boris", 
        description="Particle pusher algorithm to use"
    )

    ispec: Optional[int] = Field(None, description="Internal species index")

class Electron(Species):
    name: str = Field('electron', description="Electron particle species")
    charge: int = Field(-1, description="Electron charge (-1)")
    mass: float = Field(1, description="Electron mass (1 in units of electron mass)")
    radiation: Optional[Literal["ll", "photons"]] = Field(
        None, 
        description="Radiation model ('ll' for Landau-Lifshitz, 'photons' for QED)"
    )
    photon: Optional[Species] = Field(None, description="Photon species for QED radiation")

    def set_photon(self, photon: Species):
        """set QED radiation photon species"""

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

    electron: Optional[Species] = Field(None, description="Electron species for Breit-Wheeler pair production")
    positron: Optional[Species] = Field(None, description="Positron species for Breit-Wheeler pair production")

    def set_bw_pair(self, *, electron: Species, positron: Species):
        """set Breit-Wheeler pair production species"""

```

### Tools

```python
class SaveFieldsToHDF5:
    """Callback to save field data to HDF5 files.

    Creates a new HDF5 file for each save with name pattern:
    'prefix_t000100.h5', 'prefix_t000200.h5', etc.

    The data structure in each file:
    - /ex, /ey, /ez (electric fields)
    - /bx, /by, /bz (magnetic fields)
    - /jx, /jy, /jz (currents)
    - /rho (charge density)

    Args:
        prefix (str): Prefix for output filenames. For example, if prefix is 'output', the files will be named 'output/t000100.h5', 'output/t000200.h5', etc.
        interval (Union[int, Callable], optional): Number of timesteps between saves, or a 
            function(sim) -> bool that determines when to save. Defaults to 100.
        components (Optional[List[str]], optional): List of field components to save. 
            Available: ['ex','ey','ez','bx','by','bz','jx','jy','jz','rho']. 
            If None, saves all components.

    Attributes:
        stage (str): The simulation stage when this callback is executed.
        all_components (Set[str]): Set of all available field components.
        components (List[str]): List of components to actually save.
    """

class SaveSpeciesDensityToHDF5:
    """Callback to save species density data to HDF5 files.

    Creates a new HDF5 file for each save with name pattern:
    'prefix_speciesname_t000100.h5', 'prefix_speciesname_t000200.h5', etc.

    The data structure in each file:
    - /density (2D or 3D array)

    Args:
        species (Species): The species whose density will be saved
        prefix (str): Prefix for output filenames. For example, if prefix is 'output', the files will be named 'output/{species.name}_t000100.h5', 'output/{species.name}_t000200.h5', etc.
        interval (Union[int, Callable], optional): Number of timesteps between saves, or a 
            function(sim) -> bool that determines when to save. Defaults to 100.

    Attributes:
        stage (str): The simulation stage when this callback is executed.
        species (Species): The species being tracked.
        prev_rho (Optional[List[np.ndarray]]): Previous charge density values for computation.
    """

class SaveParticlesToHDF5:
    """Callback to save particle data to HDF5 files.

    The data structure in each file:
    - /id
    - /x, y (positions)
    - /w (weights)
    - /... (other specified attributes)

    Args:
        species (Species): The particle species to save
        prefix (str): Prefix for output filenames. For example, if prefix is 'output', the files will be named 'output/{species.name}_particles_0000100.h5'.
        interval (Union[int, Callable], optional): Number of timesteps between saves, or a
            function(sim) -> bool that determines when to save. Defaults to 100.
        attrs (Optional[List[str]], optional): List of particle attributes to save.
            If None, saves all attributes.

    Attributes:
        stage (str): The simulation stage when this callback is executed.
        species (Species): The particle species being tracked.
    """

class SetTemperature(Callback):
    """
    Callback to set the particle momenta (ux, uy, uz) for a species to a relativistic Maxwell-Jüttner distribution
    with the specified temperature (in units of mc^2).

    Args:
        species (Species): The target species whose temperature is to be set.
        temperature (float): Temperature in units of mc^2 (theta = kT/mc^2).
        interval (int or callable): Frequency (in timesteps) or callable(sim) for when to apply, defaults to run at the first timestep only once.
    """

class ExtractSpeciesDensity(SaveSpeciesDensityToHDF5):
    """Callback to extract species density from all patches.
    
    Only rank 0 will gather the data, other ranks will get zeros.
    
    Args:
        sim (Simulation): Simulation instance.
        species (Species): Species instance to extract density from.
        interval (Union[int, Callable], optional): Number of timesteps between saves, or a function(sim) -> bool that determines when to save. Defaults to 100.

    Example:

        >>> ne_ele = ExtractSpeciesDensity(sim, ele, interval=100)
        use in PlotFields:
        >>> sim.run(1000, callbacks[
                ne_ele,
                PlotFields(
                    [dict(field=ne_ele.density, scale=1/nc, cmap='Grays', vmin=0, vmax=20), 
                    dict(field='ey',  scale=e/(m_e*c*omega0), cmap='bwr_alpha', vmin=-laser.a0, vmax=laser.a0) ],
                    prefix='laser-target'),
            ])
    """

class MovingWindow:
    """Callback implementing moving window technique along x-direction.

    The moving window follows the laser or plasma flow by periodically shifting
    the simulation domain while maintaining proper boundary conditions.

    Args:
        velocity (Union[float, Callable[[float], float]]): Window velocity in m/s. 
            Can be constant or function of time (velocity=f(sim.time))
        start_time (Optional[float]): Time at which to start moving window. 
            Defaults to sim.Lx/c.
        inject_particles (bool): Whether to inject particles in new regions. 
            Defaults to True.
        stop_inject_time (Optional[float]): Time to stop particle injection. 
            Defaults to None.

    Note:
        - Handles both forward (positive) and backward (negative) moving windows
        - Maintains proper particle distributions in new regions
        - Updates patch neighbor relationships after shifts
        - Removes PML boundaries when moving starts
    """
```

### Laser

```python
class SimpleLaser(Laser):
    """
    A simple laser pulse implementation with basic spatial and temporal profiles.
    This class provides a straightforward way to inject a laser pulse into the simulation
    from the left boundary with a Gaussian transverse profile and a smooth temporal envelope.

    Use `SimpleLaser2D` or `SimpleLaser3D` for 2D and 3D simulations, respectively.
    DO NOT use this class directly.
    
    Attributes:
        a0 (float): Normalized vector potential amplitude
        l0 (float): Laser wavelength
        omega0 (float): Laser angular frequency (2πc/λ)
        w0 (float): Laser waist size (determines transverse width of the beam)
        ctau (float): Pulse duration (c*tau)
        E0 (float): Peak electric field amplitude, derived from a0
        pol_angle (float): Polarization angle in radians, 0 for z-polarization, π/2 for y-polarization
    
    Note:
        This is a simplified laser implementation suitable for basic simulations.
        For more accurate physics including proper beam evolution, wavefront curvature,
        and Gouy phase, use the GaussianLaser class instead.
    """

class GaussianLaser(Laser):
    """
    Implementation of a proper Gaussian laser beam with full physics including:
    
    - Gaussian temporal and spatial profiles
    - Proper beam waist evolution (:math:`w(z) = w_0\sqrt{1 + (z/z_R)^2}`)
    - Gouy phase (:math:`tan^{-1}(z/z_R)`)
    - Wavefront curvature (:math:`R(z) = z(1 + (z_R/z)^2)`)
    - Correct phase evolution including propagation and curvature terms

    Use GaussianLaser2D or GaussianLaser3D for 2D and 3D simulations, respectively.
    DO NOT use this class directly.
    
    Attributes:
        a0 (float): Normalized vector potential amplitude
        l0 (float): Laser wavelength
        omega0 (float): Laser angular frequency (2πc/λ)
        w0 (float): Laser waist size at focus
        ctau (float): Pulse duration (c*tau)
        x0 (float): Pulse center position (default: 3*ctau)
        tstop (float): Time to stop injection (default: 6*ctau)
        E0 (float): Peak electric field amplitude, derived from a0
        pol_angle (float): Polarization angle in radians, 0 for z-polarization, π/2 for y-polarization
        focus_position (float): Position of laser focus relative to boundary
        zR (float): Rayleigh length (πw₀²/λ)
        side (str): Injection boundary ('xmin' or 'xmax')
    
    Note:
        This implementation provides more accurate physics than SimpleLaser,
        including proper beam evolution and phase effects. Use this for
        realistic simulations where these effects matter.
    """

```

### Data structures

```python
class ParticlesBase:
    """
    The dataclass of particle data.

    This class stores and manages particle attributes including position, momentum,
    electromagnetic fields at particle positions, and particle status (alive/dead).

    Attributes:
        x,y,z (NDArray[float64]): Particle positions in x, y, z coordinates
        w (NDArray[float64]): Particle weights
        ux,uy,uz (NDArray[float64]): Normalized momentum :math:`u_i = \\gamma\\beta_i`
        inv_gamma (NDArray[float64]): Inverse relativistic gamma factor
        ex_part,ey_part,ez_part (NDArray[float64]): Electric fields interpolated at particle positions
        bx_part,by_part,bz_part (NDArray[float64]): Magnetic fields interpolated at particle positions
        is_dead (NDArray[bool_]): Boolean array indicating dead particles
        _id (NDArray[float64]): Unique particle IDs stored as float64
        npart (int): Total number of particles (including dead)

    Property:
        id (NDArray[float64]): Unique particle IDs 
        is_alive (NDArray[bool_]): Boolean mask of alive particles
    """

class Fields:
    """Base class for electromagnetic field data in particle-in-cell simulations.

    Attributes:
        nx,ny,nz (int): Grid dimensions in x, y, z directions
        n_guard (int): Number of guard cells
        dx,dy,dz (float): Grid spacings
        shape (tuple): Full array shape including guard cells
        x0,y0,z0 (float): Grid origin coordinates
        
        ex,ey,ez (NDArray[float64]): Electric field components
        bx,by,bz (NDArray[float64]): Magnetic field components  
        jx,jy,jz (NDArray[float64]): Current density components
        rho (NDArray[float64]): Charge density
        
        xaxis,yaxis,zaxis (NDArray[float64]): Coordinate axes including guard cells
        attrs (list[str]): List of field attribute names

    Note:
        Fields data are stored in [:nx, :ny, :nz] range, and the guard cells are in the [nx:, ny:, nz:] range.
        The guard cells are therefore accessed using [-n_guard:, -n_guard:, -n_guard:] and [nx:nx+n_guard, ny:ny+n_guard, nz:nz+n_guard].
    ```
## Writing callbacks

### Example: plot

```python
@callback(interval=100)
def plot_results(sim: Simulation):
    it = sim.itime
    ex, ey, ez, bx, by, bz, jy, rho = get_fields(sim, ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'jy', 'rho'])
    
    if sim.mpi.rank > 0:
        return
    ey *= e / (m_e * c * omega0)

    # color map for overlapping fields
    bwr_alpha = LinearSegmentedColormap(
        'bwr_alpha', 
        dict( 
            red=[ (0, 0, 0), (0.5, 1, 1), (1, 1, 1) ], 
            green=[ (0, 0.5, 0), (0.5, 1, 1), (1, 0, 0) ], 
            blue=[ (0, 1, 1), (0.5, 1, 1), (1, 0, 0) ], 
            alpha = [ (0, 1, 1), (0.5, 0, 0), (1, 1, 1) ]
        )
    )
    fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")

    h1 = ax.imshow(
        -rho.T/e/nc, 
        extent=[0, Lx, 0, Ly],
        origin='lower',
        cmap='Grays',
        vmax=20,
        vmin=0,
    )
    h2 = ax.imshow(
        ey.T, 
        extent=[0, Lx, 0, Ly],
        origin='lower',
        cmap=bwr_alpha,
        vmax=laser.a0,
        vmin=-laser.a0,
    )
    fig.colorbar(h1)
    fig.colorbar(h2)

    figdir = Path('laser-target')
    if not figdir.exists():
        figdir.mkdir()

    fig.savefig(figdir/f'{it:04d}.png', dpi=300)
    plt.close()
```

### Example: setting external fields

```python
@callback('interpolator')
def set_static_fields(sim: Simulation):
    for p in sim.patches:
        for part in p.particles:
            part.bz_part[:] += 10 # 10T static
            part.ex_part[:] += np.sin(sim.t) # time dependent
            part.ey_part[:] += np.sin(part.x/1e-6) # space dependent
```
or faster with numba
```python
@njit(parallel=True)
def set_static_fields(x, is_dead, t, ex_part):
    for ipart in prange(ex_part.size):
        if is_dead[ipart]:
            continue
        ex_part[ipart] += 10 # 10T static
        ex_part[ipart] += np.sin(t) # time dependent
        ex_part[ipart] += np.sin(x[ipart]/1e-6) # space dependent

@callback('interpolator')
def set_static_fields(sim: Simulation):
    for p in sim.patches:
        part = p.particles[ele.ispec]
        set_static_fields(part.x, part.is_dead, sim.t, part.ex_part)
```

### Example: reduction/summation

Calculate total EM energy and total electron kinetic energy.

```python
@callback('start', interval=100)
def sum_EM_enerty(sim: Simulation):
    Eem = 0.0
    # sum over all patches
    for p in sim.patches:
        f = p.fields
        # NOTE: guard cells are in the [nx_per_patch:, ny_per_patch:] region
        s = np.s_[:sim.nx_per_patch, :sim.ny_per_patch]
        Eem += (0.5*epsilon_0*(f.ex[s]**2+f.ey[s]**2+f.ez[s]**2) + 
                0.5/mu_0     *(f.bx[s]**2+f.by[s]**2+f.bz[s]**2)).sum()

    # sum over all mpi ranks
    Eem = sim.mpi.comm.reduce(Eem)
    if sim.mpi.rank > 0:
        return
    
    # print, or save to some file
    print(f"{Eem=:g}")


@callback('start', interval=100)
def sum_ek(sim: Simulation):
    ek = 0.0

    # sum over all patches
    for p in sim.patches:
        part = p.particles[ele.ispec]
        # select alive particles
        alive = part.is_alive
        ek += ((1/part.inv_gamma[alive] - 1) * ele.m/m_e * part.w[alive]).sum() # mc2

    # sum over all mpi ranks
    ek = sim.mpi.comm.reduce(ek)
    if sim.mpi.rank > 0:
        return
```

## Full example

### Laser incident on a carbon target
```python
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from lambdapic import (
    Electron,
    GaussianLaser2D,
    Proton,
    SaveFieldsToHDF5,
    SaveSpeciesDensityToHDF5,
    Simulation,
    Species,
    c,
    callback,
    e,
    epsilon_0,
    get_fields,
    m_e,
    pi,
)

um = 1e-6
l0 = 0.8 * um
t0 = l0 / c
omega0 = 2 * pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2

nx = 1024
ny = 1024
dx = l0 / 50
dy = l0 / 50

Lx = nx * dx
Ly = ny * dy


def density(n0):
    def _density(x, y):
        ne = 0.0
        if x > Lx/2 and x < Lx/2+1*um:
            ne = n0
        return ne
    return _density

laser = GaussianLaser2D(
    a0=10,
    w0=2e-6,
    l0=0.8e-6,
    ctau=5e-6,
    focus_position=Lx/2,
)

if __name__ == "__main__":
    sim = Simulation(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        npatch_x=16,
        npatch_y=16,
    )

    ele = Electron(density=density(10*nc), ppc=10)
    proton = Proton(density=density(10*nc/8*2), ppc=10)
    carbon = Species(name="C", charge=6, mass=12*1800, density=density(10*nc/8), ppc=10)

    sim.add_species([ele, carbon, proton])

    @callback(interval=100)
    def plot_results(sim: Simulation):
        it = sim.itime
        ex, ey, ez, bx, by, bz, jy, rho = get_fields(sim, ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'jy', 'rho'])
        
        if sim.mpi.rank > 0:
            return
        ey *= e / (m_e * c * omega0)

        bwr_alpha = LinearSegmentedColormap(
            'bwr_alpha', 
            dict( 
                red=[ (0, 0, 0), (0.5, 1, 1), (1, 1, 1) ], 
                green=[ (0, 0.5, 0), (0.5, 1, 1), (1, 0, 0) ], 
                blue=[ (0, 1, 1), (0.5, 1, 1), (1, 0, 0) ], 
                alpha = [ (0, 1, 1), (0.5, 0, 0), (1, 1, 1) ]
            )
        )
        fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")

        h1 = ax.imshow(
            -rho.T/e/nc, 
            extent=[0, Lx, 0, Ly],
            origin='lower',
            cmap='Grays',
            vmax=20,
            vmin=0,
        )
        h2 = ax.imshow(
            ey.T, 
            extent=[0, Lx, 0, Ly],
            origin='lower',
            cmap=bwr_alpha,
            vmax=laser.a0,
            vmin=-laser.a0,
        )
        fig.colorbar(h1)
        fig.colorbar(h2)

        figdir = Path('laser-target')
        if not figdir.exists():
            figdir.mkdir()

        fig.savefig(figdir/f'{it:04d}.png', dpi=300)
        plt.close()

    
    sim.run(2001, callbacks=[
            laser, 
            plot_results,
            SaveFieldsToHDF5('laser-target/fields', 100, ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'rho']),
            SaveSpeciesDensityToHDF5(carbon, 'laser-target/density', 100),
            SaveSpeciesDensityToHDF5(ele, 'laser-target/density', 100),
        ]
    )
```