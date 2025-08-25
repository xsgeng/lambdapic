from enum import Enum
from functools import update_wrapper, wraps
from typing import Any, Callable, Optional, Union

from yaspin import yaspin

from ..core.utils import logger
from ..core.utils.timer import Timer
from ..simulation import Simulation

class SimulationStage(str, Enum):
    """Enumeration of possible simulation stages for callbacks."""
    START = "start"
    MAXWELL_FIRST = "maxwell first"
    PUSH_POSITION_FIRST = "push position first"
    INTERPOLATOR = "interpolator"
    QED = "qed"
    PUSH_MOMENTUM = "push momentum"
    PUSH_POSITION_SECOND = "push position second"
    CURRENT_DEPOSITION = "current deposition"
    QED_CREATE_PARTICLEs = "qed create particles"
    LASER = "_laser"
    MAXWELL_SECOND = "maxwell second"

    @classmethod
    def all_stages(cls) -> list[str]:
        """Get a list of all available stages."""
        return [stage.value for stage in cls]

def _check_inverval(interval: int|float|Callable) -> None:
    if not isinstance(interval, (int, float, Callable)):
            raise TypeError(f"Invalid interval: {interval}. Must be int, float, or Callable")
        
    if isinstance(interval, float):
        if interval <= 0 or interval >= 1:
            raise ValueError(f"Invalid interval: {interval}. Must be between 0 and 1s if it is a float")
    if isinstance(interval, int):
        if interval < 1:
            raise ValueError(f"Invalid interval: {interval}. Must be greater than 0 if it is an integer")
        

def callback(stage: Optional[str] = None, interval: int|float|Callable = 1) -> Callable:
    """
    A decorator for implementing callbacks in PIC simulations.
    
    This decorator allows functions to be attached to specific simulation stages,
    enabling dynamic behavior modification without changing the core simulation code.
    
    Args:
        stage: The simulation stage at which this callback should be executed.
               Must be one of the values defined in SimulationStage.
               Defaults to "maxwell second" if not specified.
        interval (int|float|Callable): if int, The number of iterations between calls to the callback function.

            if float, The time interval in seconds between calls to the callback function.
            
            if Callable, The function to determine whether to call the callback function.
               The function should take a Simulation object as an argument and return a boolean value.
        
            Defaults to 1 (call every iteration).
    
    Returns:
        Callable: The decorated callable object (an instance of a Callback subclass).
    
    Example:
        >>> @callback(stage="maxwell first", interval=100)
        ... def custom_field_modification(sim):
        ...     for patch in sim.patches:
        ...         patch.fields.ex *= 1.1  # Amplify Ex field by 10%
    """
    def decorator(func: Callable) -> Callable:
        stage_value = "maxwell second" if stage is None else stage
        
        if stage_value not in SimulationStage.all_stages():
            raise ValueError(f"Invalid stage: {stage_value}. Must be one of: {SimulationStage.all_stages()}")
        
        _check_inverval(interval)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            sim = args[-1]
            if callable(interval):
                if not interval(sim):
                    return
            elif isinstance(interval, int):
                if sim.itime % interval != 0:
                    return
            elif isinstance(interval, float):
                if sim.time % interval >= sim.dt:
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
        wrapper.stage = stage_value
        
        return wrapper
    
    return decorator

class Callback:
    """A base class for implementing callbacks in PIC simulations."""

    interval: int | float | Callable
    stage: str
    
    def __call__(self, sim: Simulation) -> None:
        _check_inverval(self.interval)

        if callable(self.interval):
            if not self.interval(sim):
                return
        elif isinstance(self.interval, int):
            if sim.itime % self.interval != 0:
                return
        elif isinstance(self.interval, float):
            if sim.time % self.interval >= sim.dt:
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

    def _call(self, sim: Simulation) -> None:
        raise NotImplementedError
