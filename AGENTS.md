# PROJECT KNOWLEDGE BASE

## OVERVIEW
λPIC is a callback-centric Python Particle-In-Cell (PIC) framework for laser-plasma simulations. Core physics kernels mix Numba-JIT Python with compiled C extensions; MPI parallelism is patch-based.

## STRUCTURE
```
.
├── src/lambdapic/           # Main package
│   ├── simulation/        # Simulation, Simulation2D, Simulation3D, SimulationCallbacks
│   ├── callback/            # Callback decorator + built-in callbacks
│   ├── cli/                 # Typer CLI (lambdapic command)
│   └── core/                # Physics engine
│       ├── patch/           # Patch/Patches containers + sync + Hilbert/Metis
│       ├── mpi/             # MPI manager + load balancer + C sync extensions
│       ├── pusher/          # Boris/Photon pushers + unified C kernels
│       ├── maxwell/         # Yee FDTD + CPML solvers
│       ├── current/         # Esirkepov current deposition (C extensions)
│       ├── interpolation/   # Field interpolation to particles
│       ├── qed/             # Radiation + pair production
│       ├── collision/       # Collisions 
│       ├── boundary/        # Boundary conditions
│       ├── sort/            # Particle sorting
│       ├── ionization/      # Field ionization
│       ├── species.py       # Species, Electron, Proton, Photon, Positron
│       ├── particles.py     # ParticlesBase, QEDParticles, SpinParticles
│       ├── fields.py        # Fields, Fields2D, Fields3D
│       └── stages/          # Simulation stage definitions
├── tests/                   # pytest suite (mirrors src partially)
├── example/                 # Runnable examples (lwfa, weibel, laser-target, photons)
├── benchmarks/              # Weibel benchmark vs Smilei
└── docs/source/             # Sphinx documentation
```

## WHERE TO LOOK
| Task | Location | Notes |
|---|---|---|
| Add a callback | `src/lambdapic/callback/` | Use `@callback(stage, interval)` or subclass `Callback` |
| Change particle push | `src/lambdapic/core/pusher/` | Boris/Photon/TBMT; unified is are C kernel |
| Change field solver | `src/lambdapic/core/maxwell/` | 2D/3D Yee + CPML variants |
| Change current deposition | `src/lambdapic/core/current/` | Python facade + `cpu2d.c`/`cpu3d.c` |
| Change patch sync / load balance | `src/lambdapic/core/patch/` + `src/lambdapic/core/mpi/` | Patch container + MPI manager/load balancer |
| Add QED physics | `src/lambdapic/core/qed/` | Radiation and pair production |
| Add collision physics | `src/lambdapic/core/collision/` | Binary collisions |
| Run simulations | `src/lambdapic/simulation.py` | Main `Simulation.run()` loop and stage dispatch |
| CLI commands | `src/lambdapic/cli/main.py` | Typer app: `autoreload`, `timer-stat` (batch is TODO) |
| Tests | `tests/` | See `tests/AGENTS.md` |

## CODE MAP
| Symbol | Type | Location | Role |
|---|---|---|---|
| `Simulation` | Class | `src/lambdapic/simulation.py:118` | 2D PIC engine; `Simulation2D` is an alias |
| `Simulation3D` | Class | `src/lambdapic/simulation.py:1209` | 3D variant overrides |
| `Patches` | Class | `src/lambdapic/core/patch/patch.py:388` | Patch collection + sync orchestration |
| `Patch2D` / `Patch3D` | Class | `src/lambdapic/core/patch/patch.py` | Per-subdomain fields + particles container |
| `MPIManager` | Class | `src/lambdapic/core/mpi/mpi_manager.py` | MPI sync factory/orchestrator |
| `LoadBalancer` | Class | `src/lambdapic/core/mpi/load_balancer.py` | METIS-based patch redistribution |
| `callback` | Function | `src/lambdapic/callback/callback.py:50` | Decorator attaching stage + interval |
| `Callback` | Class | `src/lambdapic/callback/callback.py:113` | Base class for class-based callbacks |
| `BorisPusher` | Class | `src/lambdapic/core/pusher/pusher.py` | Standard Boris pusher |
| `MaxwellSolver2D` / `3D` | Class | `src/lambdapic/core/maxwell/solver.py` | Yee FDTD + CPML |
| `CurrentDeposition2D` / `3D` | Class | `src/lambdapic/core/current/deposition.py` | Current deposition facade |
| `Species` | Class | `src/lambdapic/core/species.py:47` | Species definition + particle factory |

## CONVENTIONS
- **Python 3.10+**: type hints with `|` unions, `| None` for optional.
- **Naming**: camelCase classes, snake_case functions/variables.
- **Docstrings**: Google-style with Parameters/Returns/Notes/Examples.
- **Tests**: pytest, mirror `src/` where practical, construct real `Simulation` instances, realistic plasma params.
- **C extensions**: Hot paths (sync, deposition, pusher, interpolation) are `.c` files compiled to `.so` with matching `.pyi` stubs.
- **Numba pattern**: `typed.List` of numpy arrays per patch, iterated with `prange(npatches)`.
- **Callback stages**: `init`, `start`, `maxwell_1`, `_push_position_1`, `_interpolator`, `_qed`, `_push_momentum`, `_push_position_2`, `current_deposition`, `qed_create_particles`, `_laser`, `maxwell_2`/`end`, `final`.

## ANTI-PATTERNS (THIS PROJECT)
- **DO NOT push to `upstream`**; it is the public open-source remote. Use `tea-cli` and push to `origin` only.
- **DO NOT use `Union` or `Optional`** in type hints; use `|` syntax.
- **DO NOT mock `Simulation`** in tests; construct a real instance.
- **DO NOT use the internal base classes directly** where the public API is documented (`docs/source/AI_prompt.md` warns about two classes).

## COMMANDS
```bash
# Install (editable)
pip install -e ".[test]"

# Run tests (parallel)
pytest -n 10

# Run tests without MPI/slow
pytest -m "not mpi and not slow" -n 10

# Run MPI tests
mpirun -n 9 python -m pytest --with-mpi -n 0  -m mpi tests/mpi/

# Build source distribution
python -m build -s
```

## NOTES
- `Simulation` is the 2D class; `Simulation2D` is only an alias.
- Setting `npatch_x=0` / `npatch_y=0` triggers MPI-aware auto-patching.
- The `batch` CLI subcommand is a TODO stub.
- Some `TODO` markers note deferred work (e.g., fusion iterator overflow risk, parallel numba particle extend).
- `Simulation.run()` now abort the entire MPI job on an unhandled exception, preventing deadlocks when one rank fails while others are in a collective call.
