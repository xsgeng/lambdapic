# Changelog Conventions

Rules for `CHANGELOG.md`. The release workflow extracts GitHub Release notes from this file, so the format below is enforced by CI.

## File structure

- Based on [Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).
- Newest release at the top, directly below the `## [Unreleased]` heading.
- Entries are normally written at release time by the `release` skill. If `[Unreleased]` has accumulated entries, move them under the new version heading.
- History before the first changelog entry is not backfilled — see git history for older releases.

## Version heading (mandatory)

```markdown
## [0.15.0] - 2026-07-22
```

Exactly `## [X.Y.Z] - YYYY-MM-DD`. CI extracts notes between this heading and the next `## ` heading. Pre-releases use the full version: `## [0.15.0rc1] - 2026-07-22`.

## Categories

Use only these categories, in this order; omit empty ones:

| Category | Content |
|---|---|
| `### Breaking` | Changes requiring user action: renamed/removed API, new required parameters, changed defaults or output meaning |
| `### Added` | New user-facing features: callbacks, species, physics modules, CLI commands, simulation parameters |
| `### Changed` | Behavior or output changes that remain backward-compatible |
| `### Deprecated` | Features that still work but are planned for removal |
| `### Removed` | Removed features |
| `### Fixed` | Bug fixes: wrong physics, crashes, incorrect output |
| `### Performance` | Speed or memory improvements |

## Entry rules

- One bullet per user-visible change; English; start with a verb (`Add`, `Fix`, `Change`, `Remove`, `Speed up`).
- Write what the user can do or must change — not the implementation:

  ```markdown
  - Add `SetMomentum` callback to initialize particle momentum from a function (#105)
  ```

  not `Refactor callback manager into stage runner`.
- Code identifiers in backticks, using **public API names** (`Simulation`, `@callback`, `MovingWindow`) — never internal base classes.
- Append the PR or issue reference when known: `(#107)`.
- Use indented sub-bullets for necessary detail; they stay part of the release notes.
- Omit invisible work: internal refactors, tests, CI, build system, typos — unless they change user-visible behavior.

## Version links

Keep link definitions at the bottom of the file; CI strips them from the extracted notes:

```markdown
[Unreleased]: https://github.com/xsgeng/lambdapic/compare/v0.15.0...HEAD
[0.15.0]: https://github.com/xsgeng/lambdapic/compare/v0.14.0...v0.15.0
```

When releasing: add a link for the new version, and update `[Unreleased]` to compare the new tag against `HEAD`.

## Full example

```markdown
## [0.15.0] - 2026-07-22

### Breaking
- Rename `Patch.is_boundary` to `Patch.boundary_rank`; custom patches must be updated (#98)

### Added
- Add `SetMomentumAndTemperature` callback for Maxwellian initialization (#101)
- Support `np.s_` slice syntax in HDF5 field and density callbacks (#96)

### Fixed
- Fix QED interpolation producing NaN on non-uniform grids (#103)

### Performance
- Speed up 2D current deposition by ~30% via branchless spline weights (#99)
```
