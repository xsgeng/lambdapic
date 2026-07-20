"""MPI test plugin: abort the whole MPI job on any test failure.

Without this, a failing rank exits the test while the remaining ranks may be
blocked inside a collective MPI call, causing pytest to hang indefinitely.
"""
from __future__ import annotations

import sys

import pytest
from mpi4py import MPI

_MPI_COMM = MPI.COMM_WORLD


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):  # type: ignore[no-untyped-def]
    """Call ``MPI_Abort`` as soon as any phase (setup/call/teardown) fails."""
    outcome = yield
    report = outcome.get_result()

    if report.failed and _MPI_COMM.Get_size() > 1:
        rank = _MPI_COMM.Get_rank()
        print(
            f"rank {rank}: MPI test failed, aborting job:\n{report.longrepr}",
            file=sys.stderr,
        )
        sys.stderr.flush()
        _MPI_COMM.Abort(1)
