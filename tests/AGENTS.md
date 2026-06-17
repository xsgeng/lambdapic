# TESTS

**Scope:** `tests/`

pytest suite. Partially mirrors `src/lambdapic/`; flat feature tests at the root and component tests under `tests/core/`.

## STRUCTURE
```
tests/
├── test_*.py                  # Feature-level tests (callback, collision, laser, etc.)
├── core/
│   ├── current/
│   │   └── test_current_deposition.py
│   ├── interpolation/
│   │   ├── test_field_interpolation_2d.py
│   │   └── test_field_interpolation_3d.py
│   └── pusher/
│       ├── test_unified_pusher_2d.py
│       └── test_unified_pusher_3d.py
└── mpi/
    ├── test_rebalance.py
    └── test_syncparticles.py
```

## CONVENTIONS
- **No mocking**: construct real `Simulation` instances.
- **Realistic params**: density~1e27 m^-3, cell~1e-8 m, dt~1e-17 s, ppc~10.
- **Parallel by default**: `pytest.ini` sets `-n 10` (xdist).
- **Markers**: `mpi`, `slow`, `unit`, `integration`, `component`.
- **No `conftest.py`**: fixtures/helpers are file-local or module-level helpers (`_setup_sim`).
- **Mixed frameworks**: most use pytest; `test_sort.py` use `unittest.TestCase`.

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
mpirun -n 9 python -m pytest tests/mpi/ -m mpi
```

## NOTES
- MPI tests in `tests/mpi/` expect exactly 9 ranks.
- Pusher tests use section-comment dividers and one-test-class-per-behavior pattern.
