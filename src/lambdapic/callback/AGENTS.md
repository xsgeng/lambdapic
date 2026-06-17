# CALLBACK SYSTEM

**Scope:** `src/lambdapic/callback/`

λPIC is callback-centric. This directory holds the decorator, base class, and all built-in callbacks that hook into simulation stages.

## STRUCTURE
```
src/lambdapic/callback/
├── callback.py   # @callback decorator + Callback base class
├── laser.py      # Laser injection (Simple, Gaussian, 2D/3D)
├── hdf5.py       # SaveFieldsToHDF5, SaveSpeciesDensityToHDF5, SaveParticlesToHDF5
├── plot.py       # PlotFields with custom alpha-blended colormaps
├── restart.py    # RestartDump checkpointing
├── utils.py      # get_fields, MovingWindow, SetTemperature, LoadParticles, etc.
├── vtk.py        # VTK output
└── __init__.py   # Re-exports public callbacks
```

## WHERE TO LOOK
| Task | Location | Notes |
|---|---|---|
| Add a new callback | `callback.py` | Subclass `Callback` or use `@callback(stage, interval)` |
| Add/modify laser injection | `laser.py` | `_laser` stage; side `xmin` only |
| Add output diagnostics | `hdf5.py`, `plot.py`, `vtk.py` | `end` stage by default |
| Implement moving window | `utils.py` | `MovingWindow` at `start` stage |
| Load particles from file | `utils.py` | `LoadParticles` at `init` stage |
| Restart/checkpoint | `restart.py` | Per-rank pickle shards; SIGINT/SIGTERM handling |

## CONVENTIONS
- **Stage strings**: `init`, `start`, `maxwell_1`, `_push_position_1`, `_interpolator`, `_qed`, `_push_momentum`, `_push_position_2`, `current_deposition`, `qed_create_particles`, `_laser`, `maxwell_2`/`end`, `final`.
- **Interval types**: `int` = timestep modulo; `float` = time-based; `callable` = predicate.
- **MPI**: callbacks run on all ranks; `Barrier()` is called after each callback.
- **Default stage**: callbacks without `.stage` are binned to `end`.

## ANTI-PATTERNS
- Do not put heavy MPI-communication logic in callbacks without guarding for rank.
- Do not use `RestartDump` detection by subclassing; `Simulation.run()` detects it by class name string.

## NOTES
- `SimpleLaser` uses sin² temporal envelope + Gaussian transverse profile.
- `GaussianLaser` includes waist evolution, Gouy phase, wavefront curvature, and Laguerre-Gaussian modes.
- Lasers combine via `__add__` producing a `_CombinedLaser`.
