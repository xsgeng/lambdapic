# lambdapic Quick Reference

## Contents
- Simulation Structure
- Stage Selection Cheat Sheet
- Units
- Writing Callbacks
- Data Structures
- Looping Through Patches, Particles, and Fields
- Practical Lookup Flow (for Skill Usage)
- Short Tips

## Simulation Structure

- Core objects: `lambdapic.simulation.Simulation` (2D) and `Simulation3D` (3D).
- Domain tiling: global grid split into rectangular "patches" distributed over MPI ranks.
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

## Stage Selection Cheat Sheet

All stages from `Simulation.STAGES`:

| Stage | Description | Typical Use |
|-------|-------------|-------------|
| `init` | One‑shot hook right after `initialize()` | Setting initial conditions |
| `start` | Per‑step before updates each step | Diagnostics reading fields/particles before updates |
| `maxwell_1` | After first half‑step field update | Observing/modifying fields after first half‑step |
| `current_deposition` | Per‑species after current deposition | Observing/modifying deposited currents; density extractions |
| `qed_create_particles` | After species loop, new particle creation | Reacting to new particles created |
| `maxwell_2` | After second half‑step field update | Diagnostics after second half‑step field update |
| `end` | Alias that runs immediately after `maxwell_2` | Default stage for most diagnostics |
| `final` | One‑shot hook after run loop exits and final MPI barrier completes | Final cleanup/output |

**Guidelines**:
- Field diagnostics fit `start` (before updates) or `maxwell_2`/`end` (after the full update).
- To modify **grid fields**, use `maxwell_1` or `maxwell_2` (affects the Maxwell solver); to modify **fields at particle positions**, use `_interpolator`.
- Momentum‑space diagnostics fit after `_push_momentum`.
- Per‑species operations use `current_deposition` or species‑loop stages (`_push_position_1`, etc.) where `sim.ispec` holds the current species index.
- One‑shot setup goes in `init`; cleanup and summary output in `final`.

**Notes**:
- The default stage for callbacks is `end` (`Simulation.DEFAULT_STAGE`).
- Stages with an underscore prefix belong to the particle pusher: accessible, but intended for advanced use — prefer non-underscore stages unless you need precise timing inside the species loop.
- Override the stage list per simulation via `sim.stages`.

## Units

- SI units throughout: meters, seconds, kilograms, coulombs.
- Densities are in m^-3; fields are SI electric/magnetic fields.
- `sim.time` and `dt` are in seconds; CFL determines `dt` from cell sizes.
- Common constants: `c` (speed of light), `e` (elementary charge), `epsilon_0`, `mu_0`, `m_e` (electron mass), `pi`.

## Writing Callbacks

Two styles supported:

### Function Callbacks via Decorator
```python
from lambdapic import callback

@callback(stage='end', interval=1)
def my_diagnostic(sim):
    # Your code here
    pass
```

- Stages default to `Simulation.STAGES`; override `sim.stages` for per‑simulation customization.
- Interval semantics:
  - `int > 0`: call every N steps
  - `0 < float < 1`: call by physical time interval in seconds
  - `Callable(sim) -> bool`: custom firing condition

### Class‑based Callbacks
Subclass `Callback` and define `_call(self, sim)`; set `stage` and `interval` on the instance.

```python
from lambdapic.callback.callback import Callback

class MyCallback(Callback):
    stage = 'end'
    interval = 100

    def _call(self, sim):
        # Your code here
        pass
```

### MPI Best Practices in Callbacks
- Gate output/aggregation to rank 0; use `reduce`/`gather` as needed.
- Plot only on rank 0.
- Use `get_fields(sim, fields, slice_at=None)` to assemble rank‑0 field snapshots safely. In 3D, set `slice_at=zpos` (default `sim.Lz/2`).
- Respect guard cells in arrays; slice interior ranges for physics sums.
- Keep heavy work vectorized and avoid Python loops in hot paths.

## Data Structures

### Fields (per patch)
- Object: `lambdapic.core.fields.Fields2D` / `Fields3D`
- Arrays: `ex, ey, ez, bx, by, bz, jx, jy, jz, rho`
- Shape includes guard cells: `(nx+2*ng, ny+2*ng[, nz+2*ng])`
- Coordinate axes: `xaxis, yaxis[, zaxis]` include guard cells; physical origin at `x0, y0[, z0]`
- Interior vs guards: interior indices are `:nx_per_patch, :ny_per_patch[, :nz_per_patch]` from the start of each dimension. Guard cells occupy the margins.
- Global assembly: use `get_fields(sim, ['ex', ...], slice_at=zpos)` to return global rank‑0 arrays. In 3D, this returns a z‑slice; default `slice_at` is `sim.Lz/2`.

### Particles (per species per patch)
- Base: `lambdapic.core.particles.ParticlesBase`
- Core arrays: positions `x,y,z`, weights `w`, momentum `ux,uy,uz`, `inv_gamma`
- Interpolated fields at particle positions: `ex_part, ey_part, ez_part, bx_part, by_part, bz_part`
- Status/ID: `is_dead` mask, `id` (property over `_id` as uint64), `npart`, and dynamic resize via `extend()/prune()`
- Variants add attributes: `QEDParticles` (`chi, tau, delta, event`), `SpinParticles` (`sx, sy, sz`)

### Species and Registry
- Create species via `Electron`, `Proton`, `Photon`, or generic `Species`; add with `sim.add_species([...])`
- Access per‑patch particle arrays with `p.particles[species.ispec]`

## Looping Through Patches, Particles, and Fields

- Iterate patches: `for p in sim.patches:` yields each patch. Use `p.index`, `p.ipatch_x`, `p.ipatch_y[, p.ipatch_z]` and `p.x0, p.y0[, p.z0]` for topology and coordinates.
- Access fields on a patch: `f = p.fields`. Use slices `np.s_[:sim.nx_per_patch, :sim.ny_per_patch[, :sim.nz_per_patch]]` to exclude guard cells.
- Access particles on a patch: by species index `part = p.particles[species.ispec]` or enumerate with `for ispec, s in enumerate(sim.patches.species): ...`.
- Alive mask and IDs: `alive = part.is_alive`; `ids = part.id` (uint64 view).
- Interpolated fields at particles: `part.ex_part, part.ey_part, ...`.
- Species‑loop stages: during `_push_position_1`, `_interpolator`, `_qed`, `_push_momentum`, `_push_position_2`, `current_deposition` callbacks, `sim.ispec` holds the current species index being processed. After the species loop, `sim.ispec` is reset to `None`.
- Global assembly: Use `get_fields(sim, [...])` to assemble rank‑0 global 2D/3D slices safely; other ranks return `None`.
- MPI aggregation: Use `sim.mpi.comm.reduce(value)` to sum across ranks; gate I/O on `sim.mpi.rank == 0`.

## Practical Lookup Flow (for Skill Usage)

When implementing diagnostics or physics tweaks:

1. Identify the stage you need (see Stage Selection Cheat Sheet).
2. Choose callback style (function decorator or class‑based).
3. Write callback code using the data structures and loops described.
4. Test with a small simulation.

Common patterns are provided in the examples directory (`examples/custom_callback.py` for callback patterns, `examples/custom_densities.py` for density profiles).

## Short Tips

- Stage choice: field diagnostics often fit `start` or `maxwell_2`/`end`; momentum‑space diagnostics fit after `_push_momentum`; field modification fits `_interpolator`.
- Interval choice: prefer `int` step intervals for deterministic scheduling; use `Callable(sim)` for cross‑stage conditions (e.g., run at end of species loop).
- MPI hygiene: aggregate on rank 0; do not perform matplotlib/HDF5 write on worker ranks.
- Guard cells: exclude guards for physics integrals; include guards only for stencil ops.
