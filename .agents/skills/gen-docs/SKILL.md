---
name: gen-docs
description: Update λPIC Sphinx documentation after meaningful code changes that affect product behavior, API, or user experience.
---

# Gen Docs

## Overview

This repository maintains Sphinx-based documentation under `docs/source/`. Documentation is written in **reStructuredText (.rst)** and built with **Sphinx** (theme: Furo). API reference pages use `autodoc` / `automodule` to pull content directly from Python docstrings.

Use this skill to update the corresponding documentation whenever the codebase has changes that affect user-facing behavior, public API, or simulation usage.

## Prerequisites

This skill depends on the following being in place. If any are missing, stop and report to the user before continuing:

- `docs/source/` directory with Sphinx configuration (`conf.py`) set up.
- `makefile` present at repository root (used for build/test targets).

## Workflow

1. **Inspect changes**

   - `git log main..HEAD --oneline` — commits on the current branch
   - `git diff main..HEAD --stat` — file-level scope
   - Identify which source files changed and whether they affect public APIs or user workflows.

2. **Understand user-facing impact**

   For each change, read the actual implementation when needed; **do not infer behavior from commit messages alone**. Skip:

   - Internal refactors with no externally visible behavior change
   - Tests, CI, type-only changes
   - Tooling / build-system changes that do not change how users invoke the code

   If after the scan you conclude there is no user-facing impact, say so and stop.

3. **Update docstrings in source code (if API changed)**

   Because API reference pages (`api.rst`, `simulation.rst`, `species.rst`, `callbacks.rst`, `core.rst`) use `autodoc` / `automodule`, the primary documentation for classes and functions lives in their **Python docstrings**.

   - If a public class or function signature changed, update its Google-style docstring in the source file.
   - If a new public class or function was added, ensure it has a complete docstring with `Parameters`, `Returns`, `Attributes`, `Examples`, etc.
   - If callbacks behavior changed, update the docstrings in `lambdapic/callbacks/`.

4. **Update narrative documentation (if behavior or usage changed)**

   Edit the affected `.rst` pages under `docs/source/`:

   - `introduction.rst` — for high-level concept or getting-started changes
   - `installation.rst` — for dependency or install procedure changes
   - `example.rst` — for new examples or changes to existing examples in `example/`
   - `write_callbacks.rst` — for changes to callback authoring patterns
   - `extension.rst` — for changes to the extension/plugin mechanism
   - `AI_prompt.rst` / `AI_prompt.md` — for changes to the AI prompt reference

   Match the existing reStructuredText style and section hierarchy.

5. **Build and verify docs locally**

   If a `make docs` target exists, run:
   ```bash
   make docs
   ```

   If it does not exist, run Sphinx directly:
   ```bash
   sphinx-build -b html docs/source docs/build
   ```

   Check for **warnings** (Sphinx warnings often indicate broken cross-references or autodoc failures). Fix them before finishing.

   To serve docs locally for visual review:
   ```bash
   python -m http.server -d docs/build 8000
   ```

## Rules and conventions

- **autodoc is the source of truth for API docs**: Prefer updating Python docstrings over manually duplicating API info in `.rst` files. The `.rst` API pages should remain thin wrappers (`automodule`, `autoclass`, `autofunction`).
- **Docstring style**: Follow Google-style docstrings (one-line summary, `Parameters:`, `Returns:`, `Examples:`). Use type hints in code; Napoleon will pick them up.
- **Scope discipline**: Only update sections affected by the recent changes. Do not opportunistically rewrite unrelated docs.
- **Literal includes**: `example.rst` uses `literalinclude` to pull code from `example/`. If you update an example script, the docs will auto-update on rebuild; verify the rendered output looks correct.
- **Cross-references**: Use RST cross-references (``:ref:``, ``:doc:``, ``:class:``, ``:func:``) rather than hard-coding URLs or module paths.

## Common mistakes

- Describing what code changed instead of what the user can now do (or can no longer do).
- Adding a new section heading per feature instead of weaving the change into existing prose.
- Manually writing API documentation in `.rst` files when `autodoc` will generate it from docstrings.
- Forgetting to run a local Sphinx build and missing warnings about broken references or failed autodoc.
- Editing only `.rst` wrappers and forgetting to update the underlying Python docstrings.

## Suggested makefile additions

If the root `makefile` does not yet have doc targets, consider adding:

```makefile
.PHONY: docs docs-serve

docs:
	sphinx-build -b html docs/source docs/build

docs-serve:
	python -m http.server -d docs/build 8000
```
