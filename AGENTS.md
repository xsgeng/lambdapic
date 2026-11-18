# PROJECT KNOWLEDGE BASE

## OVERVIEW
λPIC is a callback-centric Python Particle-In-Cell (PIC) framework for laser-plasma simulations. Core physics kernels mix Numba-JIT Python with compiled C extensions; MPI parallelism is patch-based.

## STRUCTURE
```
.
├── src/lambdapic/           # Main package
│   ├── simulation/          # Simulation classes (package, not a module)
│   │   ├── simulation.py    # Simulation (2D), Simulation3D, SimulationCallbacks, STAGES list
│   │   ├── _mask_simulation.py     # Sparse-domain (masked) simulation variant
│   │   └── utils.py
│   ├── callback/            # Callback decorator + built-in callbacks (per-dir AGENTS.md)
│   ├── cli/                 # Typer CLI: main.py, auto_reload.py, stat.py
│   └── core/                # Physics engine
│       ├── patch/           # Patch/Patches containers + sync + Hilbert/Metis (per-dir AGENTS.md)
│       ├── mpi/             # MPI manager + load balancer + C sync extensions
│       ├── pusher/          # Boris/Photon/TBMT pushers + unified C kernels
│       ├── maxwell/         # solver/ (Yee FDTD + CPML) subpackage
│       ├── current/         # Esirkepov current deposition (C extensions)
│       ├── interpolation/   # Field interpolation to particles
│       ├── qed/             # Radiation + pair production
│       ├── collision/       # Binary collisions (C extension)
│       ├── boundary/        # CPML PML classes; has_pml/get_pml live in boundary/utils.py
│       ├── sort/            # Particle sorting (C extensions)
│       ├── species.py       # Species, Electron, Proton, Photon, Positron
│       ├── particles.py     # ParticlesBase, QEDParticles, SpinParticles
│       ├── fields.py        # Fields, Fields2D, Fields3D
│       └── utils/
├── tests/                   # pytest suite (per-dir AGENTS.md; core/ and mpi/ subtrees)
├── example/                 # Runnable examples (lwfa, weibel, laser-target, photons)
├── skills/                  # Agent skills (lambdapic usage skill for coding agents)
└── docs/source/             # Sphinx documentation
```

## WHERE TO LOOK
| Task | Location | Notes |
|---|---|---|
| Add a callback | `src/lambdapic/callback/` | Use `@callback(stage, interval)` or subclass `Callback` |
| Inject particles at a boundary | `src/lambdapic/callback/injector.py` | `ParticleInjector`: flux source at `start` stage; cold or thermal (drifting Maxwell-Jüttner) beams |
| Change particle push | `src/lambdapic/core/pusher/` | Boris/Photon/TBMT; unified is a C kernel |
| Change field solver | `src/lambdapic/core/maxwell/` | `solver/` (2D/3D Yee + CPML) subpackage |
| Change current deposition | `src/lambdapic/core/current/` | Python facade + `cpu2d.c`/`cpu3d.c` |
| Change patch sync / load balance | `src/lambdapic/core/patch/` + `src/lambdapic/core/mpi/` | Patch container + MPI manager/load balancer |
| Add QED physics | `src/lambdapic/core/qed/` | Radiation and pair production |
| Add collision physics | `src/lambdapic/core/collision/` | Binary collisions |
| Run simulations | `src/lambdapic/simulation/simulation.py` | Main `Simulation.run()` loop and stage dispatch; stage list lives in `STAGES` |
| CLI commands | `src/lambdapic/cli/main.py` | Typer app: `autoreload`, `timer-stat` (batch is TODO) |
| Tests | `tests/` | See `tests/AGENTS.md` |

## CODE MAP
| Symbol | Type | Location | Role |
|---|---|---|---|
| `Simulation` | Class | `src/lambdapic/simulation/simulation.py:154` | 2D PIC engine; `Simulation2D` is an alias |
| `Simulation3D` | Class | `src/lambdapic/simulation/simulation.py:1310` | 3D variant overrides |
| `Patches` | Class | `src/lambdapic/core/patch/patch.py:387` | Patch collection + sync orchestration |
| `Patch2D` / `Patch3D` | Class | `src/lambdapic/core/patch/patch.py:217` / `:299` | Per-subdomain fields + particles container |
| `MPIManager` | Class | `src/lambdapic/core/mpi/mpi_manager.py:8` | MPI sync factory/orchestrator |
| `LoadBalancer` | Class | `src/lambdapic/core/mpi/load_balancer.py:14` | METIS-based patch redistribution |
| `callback` | Function | `src/lambdapic/callback/callback.py:48` | Decorator attaching stage + interval |
| `Callback` | Class | `src/lambdapic/callback/callback.py:111` | Base class for class-based callbacks |
| `BorisPusher` | Class | `src/lambdapic/core/pusher/pusher.py:113` | Standard Boris pusher |
| `BorisTBMTPusher` | Class | `src/lambdapic/core/pusher/pusher.py:146` | Boris pusher + TBMT spin precession (`pusher="boris+tbmt"`, requires `polarization`) |
| `MaxwellSolver2D` / `3D` | Class | `src/lambdapic/core/maxwell/solver/solver.py:139` / `:195` | Yee FDTD + CPML |
| `CurrentDeposition2D` / `3D` | Class | `src/lambdapic/core/current/deposition.py:171` / `:241` | Current deposition facade |
| `Species` | Class | `src/lambdapic/core/species.py:55` | Species definition + particle factory |

## CONVENTIONS
- **Environment first**: Always check and use the project `.venv` before anything else — run `.venv/bin/python`, `.venv/bin/pytest` etc. It is a **uv**-managed venv: use `uv pip` instead of `pip` for installs; if `.venv` does not exist, create it with `uv venv` first.
- **Python 3.10+**: type hints with `|` unions, `| None` for optional.
- **Naming**: camelCase classes, snake_case functions/variables.
- **Docstrings**: Google-style with Parameters/Returns/Notes/Examples.
- **Tests**: pytest, mirror `src/` where practical, construct real `Simulation` instances, realistic plasma params.
- **C extensions**: Hot paths (sync, deposition, pusher, interpolation, binary collisions) are `.c` files compiled to `.so` with matching `.pyi` stubs.
- **Numba pattern**: `typed.List` of numpy arrays per patch, iterated with `prange(npatches)`.
- **Callback stages**: `init`, `start`, `maxwell_1`, `_push_position_1`, `_interpolator`, `_qed`, `_push_momentum`, `_push_position_2`, `current_deposition`, `qed_create_particles`, `_laser`, `maxwell_2`/`end`, `final`.

## ANTI-PATTERNS (THIS PROJECT)
- **DO NOT use `Union` or `Optional`** in type hints; use `|` syntax.
- **DO NOT mock `Simulation`** in tests; construct a real instance.

## COMMANDS
```bash
# Lint (E9+F; same scope as the pre-commit hook, requires uv tool ruff)
ruff check src/ tests/

# Install (editable)
pip install -e ".[test]"

# Run tests (parallel)
pytest -n 10

# Run tests without MPI/slow
pytest -m "not mpi and not slow" -n 10

# Run MPI tests
mpirun -n 9 python -m pytest --with-mpi -n 0  -m mpi tests/mpi/

# Rebuild C extensions after a merge touches .c files
.venv/bin/python setup.py build_ext --inplace

# Build source distribution
python -m build -s
```

## NOTES
- `Simulation` is the 2D class; `Simulation2D` is only an alias.
- Setting `npatch_x=0` / `npatch_y=0` triggers MPI-aware auto-patching.
- A `'pml'` boundary in a direction requires `npatch_* >= 2` in that direction (auto-patching enforces `npatch_*_min = 2`; manual values are validated at construction, so single-patch grids must use periodic boundaries).
- The `batch` CLI subcommand is a TODO stub.
- There is no automated test CI workflow; the only GitHub Action is `release.yml` for PyPI releases.
- Some `TODO` markers note deferred work (e.g., parallel numba particle extend).
- `Simulation.run()` now abort the entire MPI job on an unhandled exception, preventing deadlocks when one rank fails while others are in a collective call.
