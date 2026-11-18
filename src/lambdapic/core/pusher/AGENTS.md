# PARTICLE PUSHERS

**Scope:** `src/lambdapic/core/pusher/`

Boris, photon, and TBMT spin pushers. **The primary hot path is the unified C kernel**; the Numba typed-list pipeline is a fallback for non-standard callbacks or QED species.

## STRUCTURE
```
src/lambdapic/core/pusher/
├── pusher.py                 # PusherBase, BorisPusher, PhotonPusher, BorisTBMTPusher
├── unified/                  # C extensions + .pyi stubs: primary interpolation+push+deposit kernel
├── boris.py                  # Pure Boris algorithm (numba fallback)
├── tbmt.py                   # Boris + TBMT spin kernel (numba fallback)
├── photon.py                 # Photon gamma update kernel
└── cpu.py                    # Numba-parallel patch dispatch kernels (fallback path)
```

## WHERE TO LOOK
| Task | Location | Notes |
|---|---|---|
| Modify primary Boris path | `unified/*.c` / `*.pyi` | One-pass interpolation + push + current deposition |
| Change unified dispatch | `pusher.py` | `BorisPusher.__call__(unified=True)` |
| Modify fallback Boris | `boris.py`, `cpu.py` | `boris_push_patches` typed-list path |
| Modify fallback TBMT spin | `tbmt.py`, `cpu.py` | standalone `tbmt_spin_patches`, applied after the Boris push |
| Modify photon pusher | `photon.py`, `cpu.py` | `photon_push_patches` |
| Add spin (TBMT) | `pusher.py` | `BorisTBMTPusher` (subclass of `BorisPusher`) expects `SpinParticles`, i.e. the species must set `polarization` |

## CONVENTIONS
- **Primary path**: unified C kernel (`unified_boris_pusher_cpu_2d/3d`, `unified_boris_tbmt_pusher_cpu_2d/3d` for spin species) whenever a species is Boris + no QED + no callbacks in pusher stages.
- **Fallback path**: Numba typed-list pipeline (`boris_push_patches` + `tbmt_spin_patches`, `push_position_patches_2d`, etc.) for QED or species with pusher-stage callbacks.
- **`@jit_spinner`**: custom decorator on fallback `@njit(parallel=True)` functions for JIT progress.
- **`EnableMixin`**: pushers can be toggled via `@if_enabled`.

## ANTI-PATTERNS
- Do not add new features to the Numba fallback without also considering the unified C path.
- Do not assume the typed-list path is used in production runs; `Simulation.run()` automatically selects unified when possible.
- Do not instantiate `PusherBase` directly; use `BorisPusher`, `PhotonPusher`, etc.

## NOTES
- `Simulation.run()` switches a species to `pusher(dt, unified=True)` when it is Boris, QED-free, and has no callbacks in pusher stages.
- **TBMT spin**: `BorisTBMTPusher` pushes the orbit exactly like `BorisPusher` and then rotates the spin in a **standalone pass** around the TBMT effective field `B_eff = (a+1/γ)B − (aγ/(γ+1))(β·B)β − (a+1/(γ+1))β×E/c` (no Stern-Gerlach feedback). The effective field is evaluated at the mid-step momentum: the standalone `tbmt_spin` kernel reconstructs u⁻ exactly from the final momentum via the inverse Boris rotation (Boris preserves |u|, so γ⁻ = sqrt(1+|u⁺−eE|²)). In the unified C kernels the spin pass runs per strip right after the Boris push of that strip (u/fields stay L1-resident); both the boris and tbmt exports share one static `unified_push_xd` helper with nullable sx/sy/sz. `a` comes from `Species.anomalous_magnetic_moment` (Electron/Positron default to the Schwinger `a_e`; base default 0.0). Spin is initialized from `Species.polarization` in `Patches.fill_particles`, `MovingWindow._fill_particles`, and `ParticleInjector`; the `SetSpin` init callback (`callback/utils.py`) overrides it with constants or spatially varying f(x, y[, z]). A polarized species with the default `boris` pusher is auto-upgraded to `boris+tbmt` with a warning in `Simulation._init_pushers` (explicit `boris+tbmt` sets it silently; `pusher='boris+tbmt'` without polarization raises). QED-created pairs do not set spin yet.
- `PhotonPusher.__call__` is a stub; the gamma update is invoked via `photon_push_patches` elsewhere.
- `np.int64(_ipatch)` casts are required for numba typed-list indexing in fallback `prange` loops.
- Unified kernels are strip-mined (`STRIP_SIZE=256`, sized so the per-strip working set fits a typical 32 KB L1d) with a per-strip `clean_strip` precheck (no dead/NaN particles in the strip): clean strips run a split path whose Boris+push loop is branch-free and vectorizes well (vector sqrt + FMA, requires `-fno-math-errno`); dirty strips run the scalar fused loops with per-particle checks. Strip granularity keeps a few dead particles from poisoning a whole patch, and the strip working set stays L1-resident across passes. Keep new per-particle work in BOTH paths.
- Keep `#pragma omp parallel for` STATIC (default): dynamic scheduling destroys per-thread patch/field cache locality across timesteps and is significantly slower in benchmarks.
- `current_deposit.h` shares its Esirkepov implementation with `cpu2d.c`; the `dcell==0` (no cell crossing) case is instantiated with compile-time constant loop bounds for full unrolling. Stencil index wrap handles arbitrary out-of-range positions (base wrapped into `[0,n)` first).
