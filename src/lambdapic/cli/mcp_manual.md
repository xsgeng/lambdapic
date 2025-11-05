# LambdaPIC MCP Manual (AI‑Focused)

This manual is written for AI agents using the LambdaPIC MCP server. It gives a compact map of the simulation and callback model, then defers deeper details to the `get_doc` tool.

Use these tools to explore details:
- `list_simulations()` to discover simulation classes
- `list_callbacks()` to discover callbacks and their stages
- `get_doc(symbol)` for authoritative docstrings of any symbol
- `get_code(symbol)` only if `get_doc` is insufficient
- `read_manual()` to read this manual again

Examples of `get_doc` calls:
- `get_doc('lambdapic.simulation.Simulation')`
- `get_doc('lambdapic.simulation.Simulation3D')`
- `get_doc('lambdapic.callback.callback.callback')`  (decorator)
- `get_doc('lambdapic.callback.callback.Callback')`   (base class)
- `get_doc('lambdapic.callback.utils.get_fields')`
- `get_doc('lambdapic.callback.hdf5.SaveFieldsToHDF5')`


## Simulation Structure

- Core object: `lambdapic.simulation.Simulation` (2D) and `Simulation3D` (3D).
- Domain tiling: the global grid is split into rectangular “patches” distributed over MPI ranks.
- Key attributes on `Simulation`:
  - Timekeeping: `itime` (step), `time` (seconds), `dt` (seconds)
  - Grid: `nx, ny[, nz]`, `dx, dy[, dz]`, `Lx, Ly[, Lz]`
  - Patching: `npatch_x, npatch_y[, npatch_z]`, `nx_per_patch, ny_per_patch[, nz_per_patch]`, `n_guard`
  - Collections: `species` (list), `patches` (iterable of patch objects)
  - MPI: `sim.mpi.rank`, `sim.mpi.size`, collectives on `sim.mpi.comm`
- Per‑patch structure:
  - `p.fields`: grid arrays with guard cells (see Data Structures)
  - `p.particles[ispec]`: particles for each species (see Data Structures)
  - `p.index`, `p.ipatch_x`, `p.ipatch_y[, p.ipatch_z]`, `p.x0`, `p.y0[, p.z0]`
- Execution stages (callbacks hook here; see `Simulation.STAGES` or customize `sim.stages`):
  - `start` → `maxwell_1` → species loop: `_push_position_1` → `_interpolator` → `_qed` → `_push_momentum` → `_push_position_2` → `current_deposition` → post‑species: `qed_create_particles` → `_laser` → `maxwell_2`
- Discovery: use `list_simulations()` then `get_doc('lambdapic.simulation.Simulation')` for full constructor/options.

## Stage Selection Cheat Sheet

- `start`: diagnostics that read fields/particles before updates.
- `maxwell_1`: observe/modify fields after first half‑step.
- `_push_position_1`: per‑species work after positions move by 0.5·dt.
- `_interpolator`: modify per‑particle fields before momentum push.
- `_qed`: radiation/pair production related instrumentation.
- `_push_momentum`: momentum‑space diagnostics or tweaks.
- `_push_position_2`: per‑species work after momentum push.
- `current_deposition`: observe/modify deposited currents; density extractions.
- `qed_create_particles`: react to new particles created.
- `_laser`: laser injection stage.
- `maxwell_2`: diagnostics after second half‑step field update.

## Units

- SI units throughout: meters, seconds, kilograms, coulombs.
- Densities are in m^-3; fields are SI electric/magnetic fields.
- `sim.time` and `dt` are in seconds; CFL determines `dt` from cell sizes.

## Writing Callbacks

Two styles are supported and discovered by `list_callbacks()`:

- Function callbacks via decorator:
  - Use `@callback(stage: str='maxwell_2', interval: int|float|Callable = 1)`.
  - Stages default to `Simulation.STAGES`; override `sim.stages` for per-simulation customization.
    - Use `start`, `maxwell_2` in most cases. Use `current_deposition` for per-species operations.
  - Interval semantics:
    - `int > 0`: call every N steps
    - `0 < float < 1`: call by physical time interval in seconds
    - `Callable(sim) -> bool`: custom firing condition

- Class‑based callbacks by subclassing `Callback` and defining `_call(self, sim)`; set `stage` and `interval` on the instance. See `get_doc('lambdapic.callback.callback.Callback')`.

MPI best practices in callbacks:
- Gate output/aggregation to rank 0; use `reduce`/`gather` as needed.
- Plot only on rank 0.
- Use `get_fields(sim, fields, slice_at=None)` to assemble rank‑0 field snapshots safely. In 3D, set `slice_at=zpos` (default `sim.Lz/2`).
- Respect guard cells in arrays; slice interior ranges for physics sums.
- Keep heavy work vectorized and avoid Python loops in hot paths.


## Callback Examples

### Plot fields
```python
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from lambdapic import callback, e, m_e, c, pi
from lambdapic.callback.utils import get_fields

um = 1e-6
l0 = 0.8 * um
omega0 = 2 * pi * c / l0

@callback(interval=100)
def plot_results(sim):
    it = sim.itime
    ex, ey, ez, bx, by, bz, jy, rho = get_fields(sim, ['ex','ey','ez','bx','by','bz','jy','rho'])
    if sim.mpi.rank > 0:
        return

    # normalize Ey using a0 scale
    ey *= e / (m_e * c * omega0)
    # a0 = e*Ey / (m_e*c*omega0)
    bwr_alpha = LinearSegmentedColormap(
        'bwr_alpha', dict(
            red=[(0,0,0),(0.5,1,1),(1,1,1)],
            green=[(0,0.5,0),(0.5,1,1),(1,0,0)],
            blue=[(0,1,1),(0.5,1,1),(1,0,0)],
            alpha=[(0,1,1),(0.5,0,0),(1,1,1)]
        )
    )
    fig, ax = plt.subplots(figsize=(5,3), layout='constrained')
    h1 = ax.imshow(rho.T, extent=[0, sim.Lx, 0, sim.Ly], origin='lower', cmap='Greys')
    h2 = ax.imshow(ey.T, extent=[0, sim.Lx, 0, sim.Ly], origin='lower', cmap=bwr_alpha)
    fig.colorbar(h1); fig.colorbar(h2)
    Path('figs').mkdir(exist_ok=True)
    fig.savefig(Path('figs')/f'{it:04d}.png', dpi=300)
    plt.close()
```

### Set external fields during interpolation.
```python
import numpy as np
from lambdapic import callback

@callback("_interpolator")
def set_static_fields(sim):
    for p in sim.patches:
        for part in p.particles:
            part.bz_part[:] += 10.0           # 10 T static Bz
            part.ex_part[:] += np.sin(sim.time)      # time dependent Ex
            part.ey_part[:] += np.sin(part.x/1e-6)   # space dependent Ey
```

### Numba‑accelerated variant (per‑particle math offloaded to JIT):
```python
import numpy as np
from numba import njit, prange
from lambdapic import callback

@njit(parallel=True)
def set_static_fields_jit(x, is_dead, t, ex_part):
    for i in prange(ex_part.size):
        if is_dead[i]:
            continue
        ex_part[i] += 10.0
        ex_part[i] += np.sin(t)
        ex_part[i] += np.sin(x[i]/1e-6)

@callback("_interpolator")
def set_static_fields(sim):
    for p in sim.patches:
        # assume 'ele' species exists; use its index
        part = p.particles[ele.ispec]
        set_static_fields_jit(part.x, part.is_dead, sim.time, part.ex_part)
```

### Sum EM energy and electron kinetic energy.
```python
import numpy as np
from lambdapic import callback, epsilon_0, mu_0, m_e

@callback('start', interval=100)
def sum_em_energy(sim):
    Eem = 0.0
    for p in sim.patches:
        f = p.fields
        s = np.s_[:sim.nx_per_patch, :sim.ny_per_patch]  # exclude guard cells
        Eem += (
            0.5*epsilon_0*(f.ex[s]**2 + f.ey[s]**2 + f.ez[s]**2)
            + 0.5/mu_0*(f.bx[s]**2 + f.by[s]**2 + f.bz[s]**2)
        ).sum()
    Eem = sim.mpi.comm.reduce(Eem)
    if sim.mpi.rank == 0:
        print(f'E_em={Eem:g}')

@callback('start', interval=100)
def sum_ek(sim):
    ek = 0.0
    # assume 'ele = Electron(...); sim.add_species([ele, ...])' exists above
    for p in sim.patches:
        part = p.particles[ele.ispec]
        alive = part.is_alive
        ek += ((1/part.inv_gamma[alive] - 1) * ele.m/m_e * part.w[alive]).sum()  # mc^2 units
    ek = sim.mpi.comm.reduce(ek)
    if sim.mpi.rank == 0:
        print(f'Ek={ek:g}')
```

## Looping Through Patches, Particles, and Fields

- Iterate patches:
  - `for p in sim.patches:` yields each patch. Use `p.index`, `p.ipatch_x`, `p.ipatch_y[, p.ipatch_z]` and `p.x0, p.y0[, p.z0]` for topology and coordinates.
- Access fields on a patch:
  - `f = p.fields`; arrays: `f.ex, f.ey, f.ez, f.bx, f.by, f.bz, f.jx, f.jy, f.jz, f.rho`.
  - Interior region (exclude guards): use slices `np.s_[:sim.nx_per_patch, :sim.ny_per_patch[, :sim.nz_per_patch]]`.
  - Coordinates: `f.xaxis, f.yaxis[, f.zaxis]` include guards; origins `f.x0, f.y0[, f.z0]`.
- Access particles on a patch:
  - By species index: `part = p.particles[species.ispec]` or enumerate with `for ispec, s in enumerate(sim.patches.species): ...`.
  - Alive mask and IDs: `alive = part.is_alive`; `ids = part.id` (uint64 view).
  - Interpolated fields at particles: `part.ex_part, part.ey_part, ...`.
- Species‑loop stages:
  - During `_push_position_1`, `interpolator`, `qed`, `push_momentum`, `push_position_2`, `current_deposition` callbacks, `sim.ispec` holds the current species index being processed.
  - After the species loop, `sim.ispec` is reset to `None`.
- Global assembly:
  - Use `get_fields(sim, [...])` to assemble rank‑0 global 2D/3D slices safely; other ranks return `None`.
- MPI aggregation:
  - Use `sim.mpi.comm.reduce(value)` to sum across ranks; gate I/O on `sim.mpi.rank == 0`.

## Data Structures

Fields (per patch):
- Object: `lambdapic.core.fields.Fields2D` / `Fields3D` with attributes:
  - Arrays: `ex, ey, ez, bx, by, bz, jx, jy, jz, rho`
  - Shape includes guard cells: `(nx+2*ng, ny+2*ng[, nz+2*ng])`
  - Coordinate axes: `xaxis, yaxis[, zaxis]` include guard cells; physical origin at `x0, y0[, z0]`
- Interior vs guards:
  - Interior indices are `:nx_per_patch, :ny_per_patch[, :nz_per_patch]` from the start of each dimension
  - Guard cells occupy the margins; to exclude guards use e.g. `f.ex[:nxp, :nyp]`
- Global assembly:
  - Use `get_fields(sim, ['ex', ...], slice_at=zpos)` to return global rank‑0 arrays. In 3D, this returns a z‑slice; default `slice_at` is `sim.Lz/2`.

Particles (per species per patch):
- Base: `lambdapic.core.particles.ParticlesBase`
- Core arrays: positions `x,y,z`, weights `w`, momentum `ux,uy,uz`, `inv_gamma`
- Interpolated fields at particle positions: `ex_part, ey_part, ez_part, bx_part, by_part, bz_part`
- Status/ID: `is_dead` mask, `id` (property over `_id` as uint64), `npart`, and dynamic resize via `extend()/prune()`
- Variants add attributes: `QEDParticles` (`chi, tau, delta, event`), `SpinParticles` (`sx, sy, sz`)

Species and registry:
- Create species via `Electron`, `Proton`, `Photon`, or generic `Species`; add with `sim.add_species([...])`
- Access per‑patch particle arrays with `p.particles[species.ispec]`

## Practical Lookup Flow (for AI)

When implementing diagnostics or physics tweaks:
1) Discover names: run `list_simulations()` and `list_callbacks()`
2) Inspect docs: prefer `get_doc('fully.qualified.Name')`
3) Only if needed, view code via `get_code('fully.qualified.Name')`

Common targets to inspect:
- `lambdapic.simulation.Simulation` / `Simulation3D`
- `lambdapic.callback.callback.callback` (decorator) and `Callback` (base)
- `lambdapic.callback.utils.get_fields`
- `lambdapic.callback.hdf5.SaveFieldsToHDF5`, `SaveSpeciesDensityToHDF5`, `SaveParticlesToHDF5`
- Plotting helpers: `lambdapic.callback.plot.PlotFields`
- Lasers: `lambdapic.callback.laser.SimpleLaser2D/3D`, `GaussianLaser2D/3D`


## Short Tips

- Stage choice: field diagnostics often fit `start` or `maxwell_2`; momentum‑space diagnostics fit after `push_momentum`; field modification fits `interpolator`.
- Interval choice: prefer `int` step intervals for deterministic scheduling; use `Callable(sim)` for cross‑stage conditions (e.g., run at end of species loop).
- MPI hygiene: aggregate on rank 0; do not perform matplotlib/HDF5 write on worker ranks.
- Guard cells: exclude guards for physics integrals; include guards only for stencil ops.

For anything not covered here, call `get_doc` for the relevant symbol.


## Basic Example

A basic 2D run with a laser incident on a target. Use `get_doc('lambdapic.simulation.Simulation')` to inspect full constructor options.

```python
from lambdapic import (
    Electron,
    ExtractSpeciesDensity,
    GaussianLaser2D,
    PlotFields,
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
    x0=10e-6
)

sim = Simulation(
    nx=nx, ny=ny,
    dx=dx, dy=dy,
    sim_time=100e-15,
    log_file='laser-target.log',
)

ele = Electron(density=density(10*nc), ppc=10)
proton = Proton(density=density(10*nc/8*2), ppc=10)
carbon = Species(name="C", charge=6, mass=12*1800, density=density(10*nc/8), ppc=10)

sim.add_species([ele, carbon, proton])
    
if __name__ == "__main__":
    sim.run(callbacks=[
            laser, 
            n_ele := ExtractSpeciesDensity(sim, ele, 500),
            PlotFields(
                [
                    dict(field=n_ele.density, scale=1/nc, cmap='Grays', vmin=0, vmax=20), 
                    dict(field='ey',  scale=e/(m_e*c*omega0), cmap='bwr_alpha', vmin=-laser.a0, vmax=laser.a0)
                ],
                prefix='laser-target', interval=10e-15,
            ),
            SaveFieldsToHDF5('laser-target/fields', 500, ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'rho']),
            SaveSpeciesDensityToHDF5(carbon, 'laser-target/density', 500),
            SaveSpeciesDensityToHDF5(ele, 'laser-target/density', 500),
        ]
    )

```
