# Changelog

All notable changes to λPIC are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- Entry conventions: .agents/skills/release/CHANGELOG_GUIDE.md -->

## [Unreleased]

## [0.18.0] - 2026-11-19

### Added

- Add `Simulation.register_typed_list_holder` to manage third-party per-patch `numba.typed.List` objects for callbacks that read particle or field arrays in Numba kernels
  - Declare the lists once (`particle_lists={"x_list": "x"}`, `field_lists={"ex_list": "ex"}`) and the registry builds them and keeps them live: particle lists follow the watched species through `extend`/`prune`, and all lists are rebuilt after load-balance migration or `MovingWindow` patch changes
  - Add `Simulation.unregister_typed_list_holder` and `Simulation.check_typed_list_freshness`, which reports stale lists, unbound registrations after a checkpoint restore, and unregistered callback holders; `Simulation.run()` emits unbound and unregistered-holder warnings once at startup
  - Checkpoints preserve registrations but drop holder objects; re-register each holder under the same name after restoring to rebuild its lists from the restored arrays, and checkpoints written before the registry existed restore as an empty registry

### Fixed

- Fix stale per-patch `typed.List` references in the pusher, sorter, and QED modules when particles are created or removed in a late stage (`qed_create_particles` and later) or after a checkpoint restore: `Simulation.run()` now reconciles particle arrays at the start of every timestep
- Fix species pusher assignment when a species requests an unsupported `pusher`; `Simulation` now validates every species before constructing any pusher, so an invalid configuration raises `ValueError` at initialization instead of shifting later species onto the wrong pusher

## [0.17.0] - 2026-11-18

### Added

- Add spin-particle support in 2D and 3D with Boris+TBMT spin precession: set `Species.polarization` to create spin-carrying particles and initialize their spin during startup, `ParticleInjector` injection, and `MovingWindow` refills; the orbit retains the standard Boris dynamics
- Add `Species.anomalous_magnetic_moment` for TBMT spin precession; `Electron` and `Positron` default to the Schwinger value
- Add `SetSpin` callback to set spin components to constants or vectorized functions of particle positions at initialization
- Add an installable λPIC agent skill for coding assistants, with simulation examples, API references, setup and troubleshooting guides

### Changed

- Change polarized species using the default `pusher="boris"` to automatically use `pusher="boris+tbmt"` with a warning; selecting `pusher="boris+tbmt"` without `polarization` now raises `ValueError`
- Change PML configuration validation: `Simulation` and `Simulation3D` now reject a `'pml'` boundary with fewer than two patches in that direction at construction; use `'periodic'` or `npatch_* >= 2`
- Change multi-rank MPI failure reporting to print the original traceback before aborting the job

### Fixed

- Fix `MovingWindow` shifts with X-direction PML boundaries:
  - Keep and regenerate the absorbing layers on the new edge patches after every shift, preserving laser injection
  - Refill the former PML strip with particles to prevent a vacuum band from drifting through the plasma
  - Remove particles inside PML shells immediately after the shift, before they can be pushed or deposit current

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

[Unreleased]: https://github.com/xsgeng/lambdapic/compare/v0.18.0...HEAD
[0.18.0]: https://github.com/xsgeng/lambdapic/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/xsgeng/lambdapic/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/xsgeng/lambdapic/compare/v0.15.1...v0.16.0
[0.15.1]: https://github.com/xsgeng/lambdapic/compare/v0.15.0...v0.15.1
[0.15.0]: https://github.com/xsgeng/lambdapic/compare/v0.14.0...v0.15.0
