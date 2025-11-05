from functools import wraps
from typing import Callable, Optional

from yaspin import yaspin

from ..core.utils.timer import Timer
from ..simulation import Simulation

def _validate_interval(interval: int | float | Callable) -> None:
    if not isinstance(interval, (int, float, Callable)):
        raise TypeError(f"Invalid interval: {interval}. Must be int, float, or Callable")

    if isinstance(interval, float):
        if interval <= 0 or interval >= 1:
            raise ValueError(f"Invalid interval: {interval}. Must be between 0 and 1s if it is a float")
    if isinstance(interval, int) and interval < 1:
        raise ValueError(f"Invalid interval: {interval}. Must be greater than 0 if it is an integer")


def _interval_triggered(sim: Simulation, interval: int | float | Callable) -> bool:
    if callable(interval):
        return bool(interval(sim))

    if isinstance(interval, int):
        return sim.itime % interval == 0

    if isinstance(interval, float):
        time_value = getattr(sim, "time", None)
        if time_value is None:
            raise AttributeError(
                "Simulation instance must provide `time` when using float interval callbacks."
            )

        return (time_value % interval) < sim.dt

    return True

def callback(stage: Optional[str] = None, interval: int|float|Callable = 1) -> Callable:
    """
    A decorator for implementing callbacks in PIC simulations.
    
    This decorator allows functions to be attached to specific simulation stages,
    enabling dynamic behavior modification without changing the core simulation code.
    
    Args:
        stage: The simulation stage at which this callback should be executed.
               Defaults to ``Simulation.default_callback_stage()`` when not specified.
        interval (int|float|Callable): if int, The number of iterations between calls to the callback function.

            if float, The time interval in seconds between calls to the callback function.
            
            if Callable, The function to determine whether to call the callback function.
               The function should take a Simulation object as an argument and return a boolean value.
        
            Defaults to 1 (call every iteration).
    
    Returns:
        Callable: The decorated callable object (an instance of a Callback subclass).
    
    Example:
        >>> @callback(stage="maxwell_1", interval=100)
        ... def custom_field_modification(sim):
        ...     for patch in sim.patches:
        ...         patch.fields.ex *= 1.1  # Amplify Ex field by 10%
    """
    def decorator(func: Callable) -> Callable:
        _validate_interval(interval)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            sim = args[-1]
            
            if not _interval_triggered(sim, interval):
                return
            
            if sim.mpi.rank == 0:
                with yaspin(text=f"Running callback: {func.__name__}") as sp:
                    with Timer(f"callback: {func.__name__}"):
                        ret = func(*args, **kwargs)
                    sim.mpi.comm.Barrier()
            else:
                with Timer(f"callback: {func.__name__}"):
                    ret = func(*args, **kwargs)
                sim.mpi.comm.Barrier()
            
            return ret
        
        # Add stage attribute and execute method
        wrapper.stage = stage
        
        return wrapper
    
    return decorator

class Callback:
    """A base class for implementing callbacks in PIC simulations."""

    interval: int | float | Callable
    stage: str
    
    def __call__(self, sim: Simulation):
        _validate_interval(self.interval)

        if not _interval_triggered(sim, self.interval):
            return
        
        if sim.mpi.rank == 0:
            with yaspin(text=f"Running callback: {self.__class__.__name__}") as sp:
                with Timer(f"callback: {self.__class__.__name__}"):
                    ret = self._call(sim)
                sim.mpi.comm.Barrier()
        else:
            with Timer(f"callback: {self.__class__.__name__}"):
                ret = self._call(sim)
            sim.mpi.comm.Barrier()

        return ret

    def _call(self, sim: Simulation):
        raise NotImplementedError
