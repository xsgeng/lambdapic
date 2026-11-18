"""Shared fixtures for the whole test suite.

Fixtures needed by more than one test file live here and are injected
automatically by pytest. File-local helpers (``_setup_sim`` style) stay
in their test module; promote them here only when a second file needs
them.
"""
from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path

import pytest


def _fail_on_stale_c_extensions() -> None:
    """Fail collection when a ``.c`` source is newer than its built ``.so``.

    Runs before the lambdapic import below so a stale or missing
    extension fails with the rebuild command, not a distant
    ``ModuleNotFoundError`` (a rebase that rewrites a ``.c`` leaves the
    old ``.so`` in place — git never rewrites build artifacts). A
    missing ``.so`` counts as stale; header-only edits are not tracked.
    """
    spec = importlib.util.find_spec("lambdapic")
    if spec is None or not spec.submodule_search_locations:
        return  # not importable at all; the import below reports that

    package_dir = Path(spec.submodule_search_locations[0])
    stale = []
    for c_file in sorted(package_dir.rglob("*.c")):
        built = [
            p for p in c_file.parent.iterdir()
            if p.suffix == ".so" and (p.name == f"{c_file.stem}.so" or p.name.startswith(f"{c_file.stem}."))
        ]
        if not built:
            stale.append(f"{c_file.relative_to(package_dir)}: no .so built")
        elif max(p.stat().st_mtime for p in built) < c_file.stat().st_mtime:
            stale.append(
                f"{c_file.relative_to(package_dir)}: newer than "
                + ", ".join(p.name for p in built)
            )
    if stale:
        raise ImportError(
            "stale C extensions: the .c sources below are newer than their "
            "compiled .so (or not built yet); rebuild with\n"
            "    .venv/bin/python setup.py build_ext --inplace\n"
            + "\n".join(stale)
        )


_fail_on_stale_c_extensions()

from lambdapic.core.species import _ALL_SPECIES  # noqa: E402


@pytest.fixture
def isolated_registry() -> Iterator[list]:
    """Snapshot and isolate the global _ALL_SPECIES registry for the test.

    Yields the emptied registry so tests may inspect what gets
    auto-registered; the pre-test content is restored afterwards,
    including on failure.
    """
    saved = list(_ALL_SPECIES)
    _ALL_SPECIES.clear()
    yield _ALL_SPECIES
    _ALL_SPECIES.clear()
    _ALL_SPECIES.extend(saved)
