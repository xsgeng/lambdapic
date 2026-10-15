# Changelog

All notable changes to λPIC are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- Entry conventions: .agents/skills/release/CHANGELOG_GUIDE.md -->

## [Unreleased]

## [0.16.0] - 2026-10-15

### Added

- Add `ParticleInjector` callback to continuously inject particles of an existing species from a domain boundary face, as a one-way flux source carrying the exact analytical flux `n * v * A * dt` per step
  - Density, drift momenta (`ux`, `uy`, `uz`), and temperature are scalars or profiles over the face's transverse coordinates, optionally followed by time; thermal beams are sampled from a drifting Maxwell-Jüttner distribution
  - New `counter-streaming.py` example: two pair streams injected from opposite faces collide and develop the two-stream/Weibel instability
- Add `timer_flush_interval` parameter to `Simulation` to set the number of steps between timer gather/flush cycles (default 100)

### Changed

- Change performance timer output: records are buffered in memory and periodically gathered on rank 0 and written to the timer file (e.g. `log.timer.txt`) instead of streaming every record through the logger; `lambdapic timer-stat` accepts both the new plain-text format and the previous one

### Fixed

- Fix particle synchronization after `MovingWindow` shifts: the global domain bounds now advance and retreat with the moved window
- Fix collisions reading stale sorter bucket bounds after load rebalance or restart; bounds are now fetched at call time

### Performance

- Speed up binary collisions by porting the Numba kernels to compiled C extensions with one random stream per patch; collision results are now bitwise-reproducible regardless of OpenMP thread count

## [0.15.1] - 2026-08-10

### Fixed
- Fix 3D MPI segfault or silent corruption after `MovingWindow` shift: `init_neighbor_rank_3d` now consults the `patch_rank_map` instead of indexing the local patch list with the global neighbor index, matching the 2D implementation

## [0.15.0] - 2026-07-22

### Added
- Add `enable_timer` parameter to `Simulation` to globally enable performance timers, written to a separate timer log file (e.g. `log.txt` -> `log.timer.txt`)

### Changed
- Timer output is now disabled by default and goes to a separate `*.timer.txt` file instead of the main log; pass `enable_timer=True` to `Simulation` to re-enable it

### Fixed
- Abort the entire MPI job when `Simulation.run()` raises an unhandled exception, instead of deadlocking while other ranks wait in collective calls
- Raise `ValueError` when inter-species collision partners get inconsistent bucket ordering (`reverse_x`), instead of silently skipping collisions
- Harden asynchronous MPI sync: fix nondeterministic handle state on errors between sync start/wait, and validate particle sync tags against `MPI_TAG_UB` for large runs

### Performance
- Speed up the unified Boris pusher kernel by ~22% in 2D and ~16% in 3D
- Overlap inter-rank MPI synchronization with intra-rank sync, hiding communication latency in parallel runs
- Speed up particle sorting via mismatch-list gather and lazy particle-list refresh
- Cache MPI derived datatypes used in boundary guard-field sync

[Unreleased]: https://github.com/xsgeng/lambdapic/compare/v0.16.0...HEAD
[0.16.0]: https://github.com/xsgeng/lambdapic/compare/v0.15.1...v0.16.0
[0.15.1]: https://github.com/xsgeng/lambdapic/compare/v0.15.0...v0.15.1
[0.15.0]: https://github.com/xsgeng/lambdapic/compare/v0.14.0...v0.15.0
