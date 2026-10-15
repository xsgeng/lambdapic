# COLLISIONS

**Scope:** `src/lambdapic/core/collision/`

Binary (Coulomb-scattering) collisions between charged particle species.

## STRUCTURE
```
src/lambdapic/core/collision/
├── collision.py            # Inter/intra-species binary collision handler (facade over cpu.so)
├── cpu.c                   # C kernels: pairing, Debye length, Coulomb scattering (OpenMP)
├── cpu.pyi                 # Type stubs for the compiled extension
├── utils.h                 # Shared static-inline C helpers (PCG32, particle views, pairing
│                           # iterators)
├── utils.py                # ParticleData namedtuple (test helper; kernels no longer use it)
└── __init__.py             # Public exports
```

## WHERE TO LOOK
| Task | Location | Notes |
|---|---|---|
| Change binary collision model | `cpu.c` | Compiled kernels: Nanbu-Perez Coulomb scattering in COM frame; `collision.py` is the Python facade |
| Change RNG / pairing | `utils.h` | PCG32, per-patch streams; state lives in `Collision.rng_state_list` (uint64 `(npatches, 2)`), seeded from the facade's `gen` and reseeded whenever `generate_field_lists()` runs (rebalance/restart) |
| Hook into simulation | `../../simulation/simulation.py` | `add_collision` |

## CONVENTIONS
- Collisions are cell-local and operate on particles within the same cell.
- The `.c` files share helpers through `utils.h`; everything is `static
  inline`, so each `.so` is independent — no cross-module linkage.
- Facades pass lists of `ParticlesBase` objects to the C extensions, constructed at the call
  site (`*_list` locals); the extensions fetch the array attributes per call
  (`prefetch_particle` in `utils.h`), so buffers
  reallocated by `extend()` are picked up automatically — no view refresh machinery exists.
  `generate_particle_lists()` is a no-op hook kept for the init/rebalance call sites.
- Sorter-derived bucket bounds are ALSO fetched at call time
  (`self.sorter[ispec].bucket_bound_min/max_list` in `calculate_debye_length`/`__call__`):
  the sorters replace those arrays wholesale on every `generate_field_lists()`, so
  generate-time copies would go stale in an ordering-dependent way. The facade keeps no
  sorter aliases at all.
- Sorters inside a collision group must share the same bucket ordering
  (`reverse_x`); `Collision.generate_field_lists()` enforces this via
  `_check_shared_bucket_ordering` (mismatched orderings silently no-op the
  inter-species collision step).

## NOTES
- Collision tests: `tests/test_collision.py` (facade behaviour),
  `tests/test_collision_thermalization.py`,
  `tests/test_collision_thread_reproducibility.py` (bitwise reproducibility
  across OpenMP thread counts), and the cell-level
  `tests/test_intra_collision_cell.py` / `tests/test_inter_collision_cell.py`.
- Collisions run on a compiled C extension; results are self-consistently
  reproducible (same seed/machine/run → bitwise identical, independent of OMP thread count)
  but do NOT reproduce Numba-version random streams (PCG32 replaced `np.random.Generator`).
- The PCG32 XSH-RR output function in the 02 port was wrong
  (`(s>>18)^(s>>27)` instead of the official `(s>>18)^s`, truncated to 19 significant bits);
  it skewed draws toward zero (collision events fired 25x too often) and was fixed in `utils.h`.
  Collision streams changed accordingly.
- `utils.py` retains only the `ParticleData` namedtuple for the cell-level collision tests;
  the module is numba-free.
- Known quirk ported faithfully: `total_density_list` accumulates across steps without
  reset in `debye_length_patches` (pre-existing in the Numba version; weakens the
  interparticle-distance floor over time).
