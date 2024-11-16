from enum import Enum
from typing import Callable, Optional
from functools import wraps, update_wrapper

class SimulationStage(str, Enum):
    """Enumeration of possible simulation stages for callbacks."""
    START = "start"
    MAXWELL_FIRST = "maxwell first"
    PUSH_POSITION_FIRST = "push position first"
    INTERPOLATOR = "interpolator"
    PUSH_MOMENTUM = "push momentum"
    PUSH_POSITION_SECOND = "push position second"
    CURRENT_DEPOSITION = "current deposition"
    MAXWELL_SECOND = "maxwell second"

    @classmethod
    def all_stages(cls) -> list[str]:
        """Get a list of all available stages."""
        return [stage.value for stage in cls]

class Callback:
    """
    A decorator class for implementing aspect-oriented callbacks in PIC simulations.
    
    This decorator allows functions to be attached to specific simulation stages,
    enabling dynamic behavior modification without changing the core simulation code.
    
    Args:
        stage (str): The simulation stage at which this callback should be executed.
                    Must be one of the values defined in SimulationStage.
    
    Example:
        @Callback(stage="maxwell first")
        def custom_field_modification(sim):
            for patch in sim.patches:
                patch.fields.ex *= 1.1  # Amplify Ex field by 10%
            
        # The decorated function can still be called directly:
        custom_field_modification(sim)  # Works normally
        
        # And the simulation can use it as a callback:
        sim.run(nsteps=1000, callbacks=[custom_field_modification])
    """
    
    def __init__(self, stage: str = None) -> None:
        """Initialize the callback with a specific simulation stage."""
        if stage is not None and stage not in SimulationStage.all_stages():
            raise ValueError(f"Invalid stage '{stage}'. Must be one of: {SimulationStage.all_stages()}")
        self.stage = stage or SimulationStage.MAXWELL_SECOND.value
        self.func = None
    
    def __call__(self, func_or_sim):
        """
        Dual-purpose __call__ method that works both as a decorator and for execution.
        
        When used as a decorator (first call):
            @Callback(stage="some_stage")
            def func(sim): ...
        
        When used for execution (subsequent calls):
            decorated_func(sim)
        
        Args:
            func_or_sim: Either the function being decorated (first call) or
                        the simulation instance (subsequent calls)
        """
        # If this is the first call (decorator mode)
        if self.func is None:
            self.func = func_or_sim
            # Make the callback instance behave like the original function
            update_wrapper(self, func_or_sim)
            return self
        
        # If this is a subsequent call (execution mode)
        return self.func(func_or_sim)
    
    def execute(self, sim) -> None:
        """
        Execute the callback function with the given simulation instance.
        Used internally by the simulation system.
        
        Args:
            sim: The simulation instance to pass to the callback function.
        """
        return self(sim)


class StageCallback:
    def start(self):
        pass

    def maxwell_first(self):
        pass

    def push_position_first(self):
        pass

    def interpolator(self):
        pass

    def push_momentum(self):
        pass

    def push_position_second(self):
        pass

    def current_deposition(self):
        pass

    def maxwell_second(self):
        pass