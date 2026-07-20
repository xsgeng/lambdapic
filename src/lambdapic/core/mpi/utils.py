"""Small MPI helper utilities used across the package."""
from __future__ import annotations

import functools
import sys
from typing import Callable, TypeVar

from mpi4py import MPI

F = TypeVar("F", bound=Callable[..., object])


def abort_on_mpi_error(method: F) -> F:
    """Decorator that aborts the MPI job if *method* raises.

    This is meant for ``Simulation.run()``-style entry points. If one rank
    raises an unhandled exception while the others are inside a collective
    MPI call, the default behaviour is a deadlock. By catching the exception
    and calling ``MPI_Comm_abort`` before re-raising, the whole job exits
    immediately.

    For single-rank runs the exception is re-raised normally so that normal
    Python traceback handling is preserved.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            comm = getattr(getattr(self, "mpi", None), "comm", None)
            if comm is None:
                comm = MPI.COMM_WORLD
            if comm.Get_size() > 1:
                rank = comm.Get_rank()
                print(
                    f"rank {rank}: unhandled exception in {method.__qualname__}; "
                    "aborting MPI job.",
                    file=sys.stderr,
                )
                sys.stderr.flush()
                comm.Abort(1)
            raise

    return wrapper  # type: ignore[return-value]
