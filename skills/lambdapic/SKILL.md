---
name: lambdapic
description: Generate and adjust lambdapic (Particle-In-Cell) simulation scripts. Use when working on laser-plasma interaction, laser wakefield acceleration (LWFA), PIC simulation callbacks or diagnostics, HDF5 output setup, or debugging lambdapic runs. Not for single-particle tracking in prescribed fields or non-PIC codes.
license: GPL-3.0
---

# lambdapic

## Agent Decision Tree (START HERE)

**Prerequisites Check (MANDATORY FIRST STEP)**
```python
import lambdapic
```

IF import fails:
    Read `references/installation.md` and set up the environment yourself:
    create a venv in the user's working directory (`uv venv --python 3.13 .venv`),
    activate it, install mpi4py and lambdapic per the guide, then retry the import check.
    Tell the user what you are doing before you start.

IF import succeeds:
    Continue with the user's request

IF request type is NEW simulation:
    CONFIRM with the user: target physics, 2D or 3D, approximate grid size
    COPY: the closest match in `examples/` to the working directory
    MODIFY: parameters to the user's physics — all SI units

IF request type is ADD diagnostic/callback:
    USE: `examples/custom_callback.py`
    Read `references/lambdapic_quick_reference.md` when you need stage names, interval semantics, or data-structure details

IF request type is DEBUG issue:
    Read `references/troubleshooting.md` for known failure modes and first fixes

IF the request submits a large run (many patches or MPI ranks):
    CONFIRM grid size and core count with the user before launching

MANDATORY: Always gate plot/print to rank 0 (`if sim.mpi.rank == 0:`)

---

## Quick Start: New Simulation

1. Copy the closest example from `examples/` to the working directory (`laser-target.py`, `lwfa.py`, ...)
2. Modify physics parameters — check units are SI:
   - Laser: `wavelength` (m), `a0`, spot size (m), duration (s)
   - Plasma: `density` (m⁻³), `thickness` (m), species mass/charge
   - Grid: `nx`, `ny`, `Lx`, `Ly` — keep `dx <= wavelength/20`
3. Register callbacks via `sim.run(callbacks=[...])` (see Using Callbacks)
4. Verify before delivering (next section)

---

## Verify Before Delivering

After writing or modifying a simulation script:

1. **Syntax gate**: `python -m py_compile <script.py>` — fix any error before proceeding
2. **Smoke run**: set `nsteps` (or `sim_time`) to a few steps, run single-process, confirm it completes without exceptions
3. **Output check**: confirm expected outputs (plots, HDF5 files) appear on disk
4. Restore full run parameters and hand the script back

A script is done only when it compiles, runs a few steps cleanly, and produces its expected outputs.

---

## Simulation Structure Reference

### Core Objects
| Object | Purpose |
|--------|---------|
| `Simulation2D` | Main 2D simulation container |
| `Simulation3D` | Main 3D simulation container |
| `species` | List of particle species |
| `patches` | Iterable of patch objects (MPI distributed) |

### Key Attributes
**Timekeeping:** `sim.itime` (step), `sim.time` (s), `sim.dt` (s)

**Grid (2D):** `sim.nx, sim.ny`, `sim.dx, sim.dy`, `sim.Lx, sim.Ly`

**Patching:** `sim.npatch_x, sim.npatch_y`, `sim.nx_per_patch, sim.ny_per_patch`, `sim.n_guard`

**MPI:** `sim.mpi.rank`, `sim.mpi.size`, `sim.mpi.comm`

### Patch Contents
Each patch `p` contains:
- `p.fields` - grid arrays (ex, ey, ez, bx, by, bz, jx, jy, jz, rho)
- `p.particles[ispec]` - particles for species index `ispec`
- `p.index` - global patch index; `p.ipatch_x, p.ipatch_y` - patch coordinates
- `p.x0, p.y0` - patch origin position

Read `references/lambdapic_quick_reference.md` when you need attribute details, stage names, or particle/field data structures beyond this page.

---

## Using Callbacks

### Builtin Callbacks

```python
# Visualization
from lambdapic.callback.plot import PlotFields

# Data output (HDF5)
from lambdapic.callback.hdf5 import SaveFieldsToHDF5
from lambdapic.callback.hdf5 import SaveSpeciesDensityToHDF5
from lambdapic.callback.hdf5 import SaveParticlesToHDF5

# Restart/Checkpoint
from lambdapic.callback.restart import RestartDump

# Laser injection
from lambdapic.callback.laser import SimpleLaser2D, SimpleLaser3D
from lambdapic.callback.laser import GaussianLaser2D, GaussianLaser3D

# Utilities
from lambdapic.callback.utils import ExtractSpeciesDensity
from lambdapic.callback.utils import MovingWindow
from lambdapic.callback.utils import SetTemperature
from lambdapic.callback.utils import LoadParticles

# Use help() to see documentation for each callback
help(PlotFields)
help(SaveFieldsToHDF5)
```

### Interval Options
- **Integer N**: Call every N steps → `interval=100`
- **Float T (0<T<1)**: Call by physical time → `interval=10e-15`
- **Callable**: Custom condition → `interval=lambda sim: sim.time > 1e-12`

### Callback Registration Template

```python
# Callbacks are passed to sim.run() — there is no sim.callbacks list
if __name__ == "__main__":
    sim.run(callbacks=[
        laser,  # laser injection callback
        PlotFields(
            [dict(field='ey', cmap='bwr_alpha', vmin=-10, vmax=10)],
            prefix='output/ey', interval=100,
        ),
        SaveFieldsToHDF5('output/fields', 50, ['ex', 'ey', 'bz']),
        SaveSpeciesDensityToHDF5(ele, 'output/density', 100),
    ])
```

---

## Units And Conventions

**All SI units:**
- Length: meters
- Time: seconds
- Mass: kilograms
- Charge: coulombs
- Density: m⁻³
- Electric field: V/m
- Magnetic field: Tesla

**Available constants:**
- `c` - speed of light
- `e` - elementary charge
- `epsilon_0` - permittivity of free space
- `mu_0` - permeability of free space
- `m_e` - electron mass
- `pi` - π

**Typical scales:**
- Laser wavelength: 0.8 µm = 0.8e-6 m
- Critical density: `nc = epsilon_0 * m_e * (2πc/λ)² / e²`
- Normalized vector potential: `a0 = e*E / (m_e*c*ω)`
- Cell size: λ/20 to λ/50

**Resolution Requirements:**
- **Minimum**: resolve skin_depth/2 AND laser_wavelength/10
- Use `scripts/calculate_skin_depth.py` to compute skin depth for your target density

---

## Anti-Patterns

| Never | Why it bites | Do instead |
|---|---|---|
| I/O without `if sim.mpi.rank == 0:` | Duplicate output, corrupted/overwritten files under MPI | Gate every `print`/`plot`/file write to rank 0 |
| Installing into the user's global/system environment unasked | Breaks the user's environment or version | Create a project-local venv in the working directory (`uv venv`), install there, per `references/installation.md` |
| Launching a large run unconfirmed | Burns the user's compute allocation | Confirm grid size and core count first |
| Mixing µm/fs with SI m/s | Orders-of-magnitude errors; the run diverges immediately | Convert all parameters to m/s/kg/C before writing |
| Writing `p.fields` guard cells without sync | Inconsistent patch boundaries, MPI errors | Read fields only; if mutation is required, ensure guard-cell exchange afterwards |
| Editing files in `examples/` directly | Destroys the reference for future reuse | Copy to the working directory, then modify |

---

## Resources

### Examples Directory (`examples/`)
- `laser-target.py` / `laser-target-3d.py` - Laser-plasma interaction (2D/3D)
- `lwfa.py` - Laser wakefield acceleration
- `photons.py` - Photon emission and QED processes
- `custom_callback.py` - Writing customized callbacks (advanced usage)
- `custom_densities.py` - Custom density profiles: closures and numba-accelerated layered targets

### References Directory (`references/`)
- `lambdapic_quick_reference.md` — complete API reference and stage cheat sheet
- `troubleshooting.md` — known failure modes and first fixes
- `installation.md` — installation and setup guide

### Scripts
- `scripts/calculate_skin_depth.py` — skin depth for a target density

---

## Bug Reporting

Report bugs and unexpected behavior at https://github.com/xsgeng/lambdapic/issues — include the full traceback, reproduction steps, and the relevant simulation script.

---

## Agent Summary (Quick Reference)

1. CHECK: `import lambdapic` (if it fails → follow `references/installation.md` to set up a local venv, then retry)
2. CONFIRM: target physics, 2D/3D — and grid size/core count before any large run
3. USE: closest example, plus the reference file that matches the request type
4. PROVIDE: working code, not just explanations
5. VERIFY: compile → smoke run → outputs, before delivering
