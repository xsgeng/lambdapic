# TESTS

**Scope:** `tests/`

pytest suite. Partially mirrors `src/lambdapic/`; flat feature tests at the root and component tests under `tests/core/`.

## STRUCTURE
```
tests/
├── conftest.py                # Stale-build guard + shared fixtures (isolated_registry); auto-injected
├── test_*.py                  # Feature-level tests (callback, collision, laser, etc.)
├── test_simulation_config.py  # Construction-time config validation (PML patch rule)
├── core/
│   ├── current/
│   │   └── test_current_deposition.py
│   ├── interpolation/
│   │   ├── test_field_interpolation_2d.py
│   │   └── test_field_interpolation_3d.py
│   └── pusher/
│       ├── test_unified_pusher_2d.py
│       ├── test_unified_pusher_3d.py
│       └── test_boris_tbmt.py
└── mpi/
    ├── conftest.py            # MPI-failure abort hook (pytest_runtest_makereport)
    ├── test_rebalance.py
    └── test_syncparticles.py
```

## CONVENTIONS
- **No mocking**: construct real `Simulation` instances.
- **Realistic params**: density~1e27 m^-3, cell~1e-8 m, dt~1e-17 s, ppc~10.
- **Parallel by default**: `pytest.ini` sets `-n 10` (xdist).
- **Markers**: `mpi`, `slow`, `unit`, `integration`, `component`.
- **Shared fixtures via `conftest.py`**: fixtures needed by more than one test file live in `tests/conftest.py` (e.g. `isolated_registry`, which snapshots/isolates the global `_ALL_SPECIES` registry) and are auto-injected. File-local helpers (`_setup_sim`) stay in their test module; promote a helper to `conftest.py` only once a second file needs it.
- **Mixed frameworks**: most use pytest; `test_sort.py` uses `unittest.TestCase`.
- **Prove the scenario fired**: assert the trigger's observable before the post-state — a test that also passes when nothing happened is a vacuous pass (e.g. before asserting on a shifted moving window, assert the shift occurred: `sim.patches.xmin_global < 0`).

## ANTI-PATTERNS
- Do not use `Union`/`Optional` in test type hints.
- Do not add heavy MPI tests that require >9 ranks without guarding with `pytest.skip`.

## COMMANDS
```bash
# Default (parallel, all markers)
pytest -n 10

# Exclude MPI/slow
pytest -m "not mpi and not slow" -n 10

# MPI tests (run under mpirun)
mpirun -n 9 python -m pytest --with-mpi -n 0  -m mpi tests/mpi/
```

## NOTES
- MPI tests in `tests/mpi/` expect exactly 9 ranks.
- Directory names enter pytest keywords, so any `mpi/` directory makes the pytest-mpi plugin skip its tests without `--with-mpi`. Unit tests for `src/lambdapic/core/mpi/` modules live in root-level files (e.g. `test_mpi_utils.py`), never under `tests/core/mpi/`.
- Pusher tests use section-comment dividers and one-test-class-per-behavior pattern.
