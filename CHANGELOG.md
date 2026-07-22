# Changelog

All notable changes to λPIC are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- Entry conventions: .agents/skills/release/CHANGELOG_GUIDE.md -->

## [Unreleased]

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

[Unreleased]: https://github.com/xsgeng/lambdapic/compare/v0.15.0...HEAD
[0.15.0]: https://github.com/xsgeng/lambdapic/compare/v0.14.0...v0.15.0
