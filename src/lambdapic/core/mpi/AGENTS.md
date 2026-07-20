# MPI & LOAD BALANCING

**Scope:** `src/lambdapic/core/mpi/`

MPI manager, dynamic load balancer, and compiled C extensions for cross-rank particle/field synchronization.

## STRUCTURE
```
src/lambdapic/core/mpi/
├── mpi_manager.py      # MPIManager factory + MPIManager2D/3D
├── load_balancer.py    # METIS-based dynamic patch redistribution
├── sync_particles_*.c  # C extension: cross-boundary particle migration
├── sync_fields*.c      # C extension: E/B guard-cell and current halo exchange
└── __init__.py         # Exports MPIManager, MPIManager2D/3D, LoadBalancer
```

## WHERE TO LOOK
| Task | Location | Notes |
|---|---|---|
| Change how ranks sync particles | `mpi_manager.py` + `sync_particles_2d.c`/`sync_particles_3d.c` | Two-phase count then fill |
| Change guard/halo field exchange | `mpi_manager.py` + `sync_fields2d.c`/`sync_fields3d.c` | E/B and J halo exchange |
| Change load balancing | `load_balancer.py` | Uses PyMETIS; patches moved via serialized send/recv |
| Tune rebalance thresholds | `load_balancer.py` | Adaptive inc/dec factors prevent thrashing |

## CONVENTIONS
- **Factory pattern**: `MPIManager.create()` returns a `MPIManager2D` or `MPIManager3D`.
- **Single-rank fast path**: sync methods short-circuit when `size == 1` (`_start` returns `None`, `_wait(None)` is a no-op).
- **Split start/wait API**: every sync has `sync_*_start()` (posts non-blocking MPI, returns an opaque capsule handle) and `sync_*_wait(handle)` (completes + unpacks). The blocking `sync_*()` wrappers chain both and remain the default for callbacks.
- **Dedicated communicators**: particle sync and current sync use `Dup()`'d communicators (`comm_particles`, `comm_currents`) so different sync kinds can overlap in flight without tag collisions. Guard-field syncs use the base `comm`. Dup'd comms are dropped on pickling and re-`Dup()`'d in `__setstate__` (checkpoint/restart safe).
- **Species-namespaced tags**: particle sync tags are `(patch_index*NUM_BOUNDARIES + ibound)*nspec + ispec`, so per-species particle syncs may run concurrently (start all species, then wait all).
- **Overlap invariants** (relied on by `Simulation.run`):
  - `mpi.sync_*` only touches different-rank boundaries (`neighbor_rank >= 0`); `patches.sync_*` only same-rank ones. They write disjoint guard cells and may overlap.
  - Current sync packs send buffers at `_start`, so fields may be modified freely until `_wait`; but `_wait` must precede the E-field update (J consumer) and any load-balance patch migration.
  - Guard-field `Isend` reads field arrays directly (derived datatypes), so `_wait` must precede the next kernel writing those fields.
- **Boundary derived datatypes are cached** per `(nx, ny[, nz], ng)` in `sync_fields*.c`; never freed (leak on geometry change is intentional).
- **Serialization**: `LoadBalancer` uses `dill` (not std `pickle`) to move whole `Patch` objects between ranks.
- **Load metric**: particle count + cell count weights drive METIS partitioning.

## ANTI-PATTERNS
- Do not call sync routines directly in normal code; go through `Patches.sync_*()`.
- Do not rely on `MPIManager.get_defailt_size()` (typo in base); it has no known callers.

## NOTES
- All C extensions have `.pyi` type stubs for IDE/type-checker support.
