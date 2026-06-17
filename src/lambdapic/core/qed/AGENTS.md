# QED MODULES

**Scope:** `src/lambdapic/core/qed/`

Quantum-electrodynamic processes: radiation reaction (Compton/Neutron-like photon emission) and electron-positron pair production.

## STRUCTURE
```
src/lambdapic/core/qed/
├── radiation.py       # RadiationBase + implementation
├── pair_production.py # PairProductionBase + implementation
└── __init__.py        # Public exports
```

## WHERE TO LOOK
| Task | Location | Notes |
|---|---|---|
| Add/modify photon emission | `radiation.py` | Handles χ (quantum nonlinearity parameter) |
| Add/modify pair production | `pair_production.py` | Breit-Wheeler process |
| Change QED particle types | `../particles.py` | `QEDParticles`, `SpinQEDParticles` |
| Hook QED into simulation | `../../simulation.py` | `_qed` and `qed_create_particles` stages |

## CONVENTIONS
- QED operates on `QEDParticles`/`SpinQEDParticles` which add `chi`, `tau`, `delta`, `event` arrays.
- Photon species must be connected to its parent `Electron` via `set_photon`.
- Pair-produced electrons/positrons are linked on `Photon` via `set_bw_pair`.
- QED callbacks run at `_qed` (reaction sampling) and `qed_create_particles` (particle creation) stages.

## ANTI-PATTERNS
- Do not bypass the `ParticlesBase.extend`/`prune` lifecycle for QED-created particles.
- Do not assume photons have mass; `inv_gamma` is computed differently than for massive species.

## NOTES
- QED tables/kernels are typically precomputed or sampled; check for `.h5`/lookup data in this directory.
