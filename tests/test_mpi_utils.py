"""abort_on_mpi_error: multi-rank failures print the traceback before aborting.

MPI_Comm_abort terminates the process before the re-raised exception can
be formatted and printed by normal exception handling, so the decorator
must emit the traceback itself. Without it, a multi-rank failure
surfaces only as the one-line abort message and the cause has to be
found by bisection (plain unit tests; no mpirun needed — the fake comm
is the seam under test, the real Abort is never called).
"""
from types import SimpleNamespace

import pytest

from lambdapic.core.mpi.utils import abort_on_mpi_error


class _FakeComm:
    """Minimal comm stand-in: records Abort, never kills the process."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.aborted: int | None = None

    def Get_size(self) -> int:
        return self.size

    def Get_rank(self) -> int:
        return 0

    def Abort(self, code: int) -> None:
        self.aborted = code


class _Job:
    """Smallest host the decorator accepts: a `mpi` attribute with a comm."""

    def __init__(self, comm: _FakeComm) -> None:
        self.mpi = SimpleNamespace(comm=comm)

    @abort_on_mpi_error
    def run(self) -> None:
        raise RuntimeError("boom")


def test_multi_rank_prints_traceback_before_abort(capsys):
    comm = _FakeComm(size=2)
    with pytest.raises(RuntimeError, match="boom"):
        _Job(comm).run()

    err = capsys.readouterr().err
    assert "aborting MPI job" in err
    assert "Traceback" in err, "the traceback must precede Abort, which kills the process"
    assert "RuntimeError: boom" in err
    assert comm.aborted == 1


def test_single_rank_reraises_without_abort(capsys):
    comm = _FakeComm(size=1)
    with pytest.raises(RuntimeError, match="boom"):
        _Job(comm).run()

    # single-rank: normal Python traceback handling, decorator stays silent
    assert capsys.readouterr().err == ""
    assert comm.aborted is None
