# PATCH SYSTEM

**Scope:** `src/lambdapic/core/patch/`

Patch/Patches containers, neighbor topology, guard-cell sync, Hilbert/Metis ordering, and particle fill logic.

## STRUCTURE
```
src/lambdapic/core/patch/
├── patch.py          # Patch (base), Patch2D, Patch3D, Patches, Boundary2D/3D
├── cpu.py            # Numba-parallel count/fill particles per patch from density+PPC
├── hilbert.py        # Gilbert/Hilbert 2D/3D space-filling curve ordering
├── metis.py          # PyMETIS partitioning helpers
├── sync_*.c          # C extensions: field/current/particle sync
└── __init__.py       # Exports Patch2D, Patch3D, Patches only
```

## WHERE TO LOOK
| Task | Location | Notes |
|---|---|---|
| Add a new patch field/particle container | `patch.py` | `Patch` base + `Patch2D`/`Patch3D` |
| Change neighbor topology | `patch.py:24-69` | `Boundary2D`/`Boundary3D` IntEnums; values must stay in sync with C sync code |
| Change particle initialization | `patch.py:388-899` + `cpu.py` | `Patches.fill_particles`, `calculate_npart` |
| Change guard sync | `patch.py` + `sync_*.c` | `sync_particles`, `sync_guard_fields`, `sync_currents` dispatch to C extensions |
| Load balancing / patch ordering | `hilbert.py`, `metis.py` | Used by `LoadBalancer` and `Simulation._auto_patch` |

## CONVENTIONS
- **Neighbor triplet**: every patch carries three parallel arrays indexed by `Boundary` enum: `neighbor_index`, `neighbor_rank`, `neighbor_ipatch`. Missing neighbors use `-1`.
- **`copy_skeleton()`** (line 188): metadata-only copy for rebalancing; skips `fields`/`particles`/`pml_boundary`.
- **PML boundary limits**: `Patch2D` allows max 2 PML boundaries (one X + one Y); `Patch3D` allows max 3.
- **`Patches.nx/ny/nz/dx/dy/dz`** read from `self[0].fields`; assumes uniform grid.
- **Species ordering**: particles are stored per-patch in `patch.particles[ispec]` indexed by `Patches.species` list order.

## ANTI-PATTERNS
- Do not change `Boundary2D`/`Boundary3D` enum values without updating the matching C sync code.
- Do not access `.fields` on a skeleton patch; it raises `AttributeError` by design.
- Do not assume non-uniform patch sizes in `Patches` geometry properties.

## NOTES
- `update_lists()` / `update_particle_lists()` are empty stubs overridden by the MPI transfer subclass.
- Hilbert ordering requires even dimensions (`nx%2==0`, `ny%2==0`).
