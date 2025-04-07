from enum import Enum
from typing import Callable, Optional, Any, Union
from functools import wraps, update_wrapper

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

def callback(stage: Optional[str] = None) -> Callable:
    """
    A decorator for implementing callbacks in PIC simulations.
    
    This decorator allows functions to be attached to specific simulation stages,
    enabling dynamic behavior modification without changing the core simulation code.
    
    Args:
        stage: The simulation stage at which this callback should be executed.
               Must be one of the values defined in SimulationStage.
               Defaults to "maxwell second" if not specified.
    
    Returns:
        Callable: The decorated function with added stage and execute attributes.
    
    Example:
        @callback(stage="maxwell first")
        def custom_field_modification(sim):
            for patch in sim.patches:
                patch.fields.ex *= 1.1  # Amplify Ex field by 10%
        
        @callback()  # defaults to "maxwell second"
        def custom_field_modification(sim):
            for patch in sim.patches:
                patch.fields.ex *= 1.1
    """
    def decorator(func: Callable) -> Callable:
        stage_value = "maxwell second" if stage is None else stage
        
        if stage_value not in SimulationStage.all_stages():
            raise ValueError(f"Invalid stage: {stage_value}. Must be one of: {SimulationStage.all_stages()}")
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Add stage attribute and execute method
        wrapper.stage = stage_value
        wrapper.execute = wrapper
        
        return wrapper
    
    return decorator
