from time import perf_counter_ns
from .logger import logger
from typing import Optional

from ..mpi.mpi_manager import MPIManager

class Timer:
    def __init__(
        self,
        name: Optional[str] = None,
        norm: float = 1.0,
        unit: str = "ms",
        log_kwargs: Optional[dict] = None,
        disable: bool = False
    ):
        """Performance timer context manager
        
        Args:
            name: Timer name for logging
            norm: Normalization factor for time values
            unit: Time unit (s, ms, us, ns)
            log_kwargs: Additional keyword arguments for logger
            disable: If True, disables timing and logging
        """
        self.disable = disable
        if self.disable:
            return
            
        self.name = name
        self.log_kwargs = log_kwargs or {}
        
        # Unit conversion factors
        unit_factors = {
            "s": 1e9,
            "ms": 1e6,
            "us": 1e3,
            "ns": 1.0
        }
        
        if unit not in unit_factors:
            raise ValueError(f"Invalid time unit '{unit}'. Valid options: {list(unit_factors.keys())}")
        
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
            message = f"Rank {MPIManager.get_rank()} {self.name} took {self.interval:.1f}{self.unit}"
        else:
            message = f"Rank {MPIManager.get_rank()} Completed in {self.interval:.1f}{self.unit}"
        
        if self.interval > 0.1:
            logger.log("TIMER", message, **self.log_kwargs)
