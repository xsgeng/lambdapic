---
name: release
description: Release a new λPIC version to PyPI with an AI-generated changelog and GitHub Release. Use when the user asks to release, publish, tag, or bump a new version.
---

# Release

## Overview

Releasing = pushing a `v*` tag to `origin`. The tag triggers `.github/workflows/release.yml`, which verifies the version, builds the sdist, creates a GitHub Release (notes extracted from `CHANGELOG.md`), and publishes to PyPI.

Version numbers come from git tags via `setuptools_scm` — never edit a version file.

## Prerequisites

Stop and report to the user if any are missing:

- On `main`, clean worktree, in sync with `origin/main` (`git fetch origin && git status`).
- `gh` CLI authenticated (`gh auth status`) — needed to monitor the workflow run.
- Tests pass locally (step 2 below).

## Workflow

### 1. Determine the next version

```bash
git describe --tags --abbrev=0          # last tag, e.g. v0.14.0
git log <last-tag>..HEAD --oneline      # commits to be released
```

Suggest a semver bump and **confirm with the user** before proceeding:

- Breaking API change → major; `feat:` / user-visible feature → minor; only `fix:` → patch
- Pre-release tags are allowed, e.g. `v0.15.0rc1` (the workflow marks them as prerelease)

### 2. Test gate (mandatory)

```bash
pytest -m "not mpi and not slow" -n 10
```

All tests must pass. On failure, stop and report — do not release.

### 3. Draft the CHANGELOG entry

Follow the conventions in [CHANGELOG_GUIDE.md](CHANGELOG_GUIDE.md) (categories, entry style, version links). Key points:

- Base the entry on `git log <last-tag>..HEAD` and PR titles. Read the actual implementation when a commit message is unclear — do not invent changes.
- Heading format is mandatory — CI extracts release notes by it: `## [0.15.0] - 2026-07-22`
- Insert the entry at the top of `CHANGELOG.md` (below `## [Unreleased]`, above previous releases) and update the version links at the bottom of the file.
- Show the draft to the user for approval.

### 4. Commit, tag, push

Ask for user confirmation before the git mutations. **Push to `origin` only — never to `upstream`.**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): release v0.15.0"
git tag -a v0.15.0 -m "Release v0.15.0"
git push origin main v0.15.0
```

### 5. Monitor and verify

```bash
gh run list --workflow=release.yml --limit 1
gh run watch                        # follow until the run completes
gh release view v0.15.0             # notes + sdist asset attached
```

Finally check that https://pypi.org/project/lambdapic/ shows the new version (the `publish-pypi` job may wait for approval in the `release` environment).

## Failure handling

- **`verify` job failed** (tag ≠ setuptools_scm version): delete the tag, fix, re-tag:
  `git push origin :refs/tags/vX.Y.Z && git tag -d vX.Y.Z`
- **`github-release` failed** ("No CHANGELOG.md entry"): heading format wrong — fix `CHANGELOG.md`, then delete and re-push the tag.
- **`publish-pypi` failed**: fix the cause, then "Re-run failed jobs" in the Actions UI — no new tag needed.
- **Bad release already on PyPI**: yank it in the PyPI web UI, `gh release delete vX.Y.Z`, delete the tag. Publish a fixed patch version; never reuse a yanked version number.

## Common mistakes

- Tagging before the changelog commit is in the tagged commit → the release job reads `CHANGELOG.md` from the tagged commit and fails.
- Pushing the tag to `upstream` — forbidden (see AGENTS.md).
- Skipping the test gate because "CI will catch it" — there is no test CI; this gate is the only check.
