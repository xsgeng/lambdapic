# PARTICLE PUSHERS

**Scope:** `src/lambdapic/core/pusher/`

Boris, photon, and TBMT spin pushers. **The primary hot path is the unified C kernel**; the Numba typed-list pipeline is a fallback for non-standard callbacks or QED/spin species.

## STRUCTURE
```
src/lambdapic/core/pusher/
├── pusher.py                 # PusherBase, BorisPusher, PhotonPusher, BorisTBMTPusher
├── unified/                  # C extensions + .pyi stubs: primary interpolation+push+deposit kernel
├── boris.py                  # Pure Boris algorithm (numba fallback)
├── photon.py                 # Photon gamma update kernel
└── cpu.py                    # Numba-parallel patch dispatch kernels (fallback path)
```

## WHERE TO LOOK
| Task | Location | Notes |
|---|---|---|
| Modify primary Boris path | `unified/*.c` / `*.pyi` | One-pass interpolation + push + current deposition |
| Change unified dispatch | `pusher.py` | `BorisPusher.__call__(unified=True)` |
| Modify fallback Boris | `boris.py`, `cpu.py` | `boris_push_patches` typed-list path |
| Modify photon pusher | `photon.py`, `cpu.py` | `photon_push_patches` |
| Add spin (TBMT) | `pusher.py` | `BorisTBMTPusher` expects `SpinParticles` |

## CONVENTIONS
- **Primary path**: unified C kernel (`unified_boris_pusher_cpu_2d/3d`) whenever a species is Boris + no QED + no callbacks in pusher stages.
- **Fallback path**: Numba typed-list pipeline (`boris_push_patches`, `push_position_patches_2d`, etc.) for QED, spin, or species with pusher-stage callbacks.
- **`@jit_spinner`**: custom decorator on fallback `@njit(parallel=True)` functions for JIT progress.
- **`EnableMixin`**: pushers can be toggled via `@if_enabled`.

## ANTI-PATTERNS
- Do not add new features to the Numba fallback without also considering the unified C path.
- Do not assume the typed-list path is used in production runs; `Simulation.run()` automatically selects unified when possible.
- Do not instantiate `PusherBase` directly; use `BorisPusher`, `PhotonPusher`, etc.

## NOTES
- `Simulation.run()` switches a species to `pusher(dt, unified=True)` when it is Boris, QED-free, and has no callbacks in pusher stages.
- `PhotonPusher.__call__` is a stub; the gamma update is invoked via `photon_push_patches` elsewhere.
- `np.int64(_ipatch)` casts are required for numba typed-list indexing in fallback `prange` loops.
