from time import perf_counter_ns
from typing import Any

from ..mpi.mpi_manager import MPIManager
from .logger import logger

_timer_enabled: bool = False


def set_timer_enabled(enabled: bool) -> None:
    """Globally enable or disable the :class:`Timer` context manager.

    When disabled, every :class:`Timer` instance becomes a no-op with minimal
    overhead. This is used by :class:`lambdapic.simulation.Simulation` when
    ``enable_timer=True`` is passed.

    Parameters:
        enabled: If ``True``, timers will measure and log their intervals.
    """
    global _timer_enabled
    _timer_enabled = enabled


def timer_enabled() -> bool:
    """Return the current global timer enable state."""
    return _timer_enabled


class Timer:
    def __init__(
        self,
        name: str | None = None,
        norm: float = 1.0,
        unit: str = "ms",
        log_kwargs: dict[str, Any] | None = None,
        disable: bool = False
    ):
        """Performance timer context manager.

        Parameters:
            name: Timer name for logging.
            norm: Normalization factor for time values.
            unit: Time unit (``s``, ``ms``, ``us``, ``ns``).
            log_kwargs: Additional keyword arguments for the logger.
            disable: If ``True``, disables timing and logging for this instance.
        """
        self.disable = disable or not _timer_enabled
        if self.disable:
            return

        self.name = name
        self.log_kwargs = log_kwargs or {}

        unit_factors = {
            "s": 1e9,
            "ms": 1e6,
            "us": 1e3,
            "ns": 1.0
        }

        if unit not in unit_factors:
            raise ValueError(
                f"Invalid time unit '{unit}'. "
                f"Valid options: {list(unit_factors.keys())}"
            )

        self.norm = unit_factors[unit] * float(norm)
        self.unit = unit

    def __enter__(self):
        if self.disable:
            return self

        self.start = perf_counter_ns()
        return self

    def __exit__(self, *args):
        if self.disable:
            return

        self.end = perf_counter_ns()
        self.interval = (self.end - self.start) / self.norm

        if self.name:
            message = (
                f"Rank {MPIManager.get_default_rank()} "
                f"{self.name} took {self.interval:.1f}{self.unit}"
            )
        else:
            message = (
                f"Rank {MPIManager.get_default_rank()} "
                f"Completed in {self.interval:.1f}{self.unit}"
            )

        if self.interval > 0.1:
            logger.log("TIMER", message, **self.log_kwargs)
