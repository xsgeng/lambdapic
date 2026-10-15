from pathlib import Path
from time import perf_counter_ns
from typing import Any

from ..mpi.mpi_manager import MPIManager

_timer_enabled: bool = False

# In-memory buffer of (rank, name, interval_ms) records produced by Timer
# contexts. Flushed periodically to disk by flush_timers().
_records: list[tuple[int, str, float]] = []

# Rank-0-owned path to the timer log file. None => records are still buffered
# (so timer-stat could be run) but never written.
_timer_file: str | None = None

# Communicator used for the periodic gather. None => single-rank fast path.
_comm = None

# Rank of this process, captured once at enable time. Stable for the lifetime
# of the process, so it is reused on every Timer exit and flush without
# re-querying MPI.
_rank: int = 0

# Module-level constant so Timer.__init__ does not rebuild a dict literal on
# every call.
_UNIT_FACTORS = {
    "s": 1e9,
    "ms": 1e6,
    "us": 1e3,
    "ns": 1.0,
}


def set_timer_enabled(
    enabled: bool,
    timer_file: str | None = None,
    comm: Any = None,
    truncate: bool = True,
) -> None:
    """Globally enable or disable the :class:`Timer` context manager.

    When disabled, every :class:`Timer` instance becomes a no-op with minimal
    overhead. This is used by :class:`lambdapic.simulation.Simulation` when
    ``enable_timer=True`` is passed.

    When enabled, timer measurements are accumulated in an in-memory list and
    periodically gathered on rank 0 and written to ``timer_file`` via
    :func:`flush_timers`.

    Parameters:
        enabled: If ``True``, timers will measure and accumulate intervals.
        timer_file: Path to the dedicated timer log file. Only rank 0 writes;
            other ranks still buffer locally. If ``None``, timers run but are
            never flushed to disk.
        comm: MPI communicator to use for the periodic gather. If ``None``,
            ``MPI.COMM_WORLD`` is used (single-rank runs take a fast path).
        truncate: If ``True`` and ``timer_file`` exists, truncate it once on
            enable (rank 0 only).
    """
    global _timer_enabled, _timer_file, _comm, _records, _rank
    _timer_enabled = enabled
    _records = []
    if enabled:
        _timer_file = timer_file
        if comm is None:
            try:
                comm = MPIManager.get_default_comm()
            except Exception:
                comm = None
        _comm = comm
        _rank = 0
        if comm is not None:
            try:
                _rank = comm.Get_rank()
            except Exception:
                _rank = 0
        if timer_file is not None and _rank == 0 and truncate:
            p = Path(timer_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(timer_file, "w") as f:
                f.truncate()
    else:
        _timer_file = None
        _comm = None
        _rank = 0


def timer_enabled() -> bool:
    """Return the current global timer enable state."""
    return _timer_enabled


def timer_sink_path(sink: str) -> str:
    """Derive the dedicated timer log file path from the main log file path.

    Parameters:
        sink: Path to the main log file.

    Returns:
        Path to the timer log file. For example ``log.txt`` becomes
        ``log.timer.txt``.
    """
    p = Path(sink)
    return str(p.with_name(p.stem + ".timer" + p.suffix))


def flush_timers() -> None:
    """Gather buffered timer records on rank 0 and append them to the file.

    Each rank contributes its local ``_records`` buffer; rank 0 concatenates
    them and appends one line per record in the form::

        Rank <rank> <name> took <interval>ms

    After the gather all ranks clear their local buffer. No-op when timers
    are disabled or no timer file is configured.
    """
    if not _timer_enabled:
        return

    comm = _comm

    if comm is not None:
        gathered = comm.gather(_records, root=0)
    else:
        gathered = [_records] if _rank == 0 else None

    if _rank == 0:
        out = _timer_file
        if out is not None and gathered is not None:
            lines: list[str] = []
            for per_rank in gathered:
                if not per_rank:
                    continue
                for r, name, interval in per_rank:
                    lines.append(f"Rank {r} {name} took {interval:.1f}ms")
            if lines:
                with open(out, "a") as f:
                    f.write("\n".join(lines))
                    f.write("\n")

    _records.clear()


class Timer:
    def __init__(
        self,
        name: str | None = None,
        norm: float = 1.0,
        unit: str = "ms",
        disable: bool = False
    ):
        """Performance timer context manager.

        On exit the measured interval (in the requested unit) is appended to
        the in-memory record buffer maintained by this module. The buffer is
        periodically gathered on rank 0 and written to the timer file by
        :func:`flush_timers`.

        Parameters:
            name: Timer name for logging.
            norm: Normalization factor for time values.
            unit: Time unit (``s``, ``ms``, ``us``, ``ns``).
            disable: If ``True``, disables timing for this instance.
        """
        self.disable = disable or not _timer_enabled
        if self.disable:
            return

        self.name = name

        if unit not in _UNIT_FACTORS:
            raise ValueError(
                f"Invalid time unit '{unit}'. "
                f"Valid options: {list(_UNIT_FACTORS.keys())}"
            )

        self.norm = _UNIT_FACTORS[unit] * float(norm)
        self.unit = unit

    def __enter__(self):
        if self.disable:
            return self

        self.start = perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable:
            return

        self.end = perf_counter_ns()
        interval = (self.end - self.start) / self.norm

        if interval > 0.1:
            name = self.name if self.name else "Completed"
            _records.append((_rank, name, interval))
