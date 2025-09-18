from functools import wraps
from yaspin import yaspin
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()

def jit_spinner(func=None, *, spinner_text=None):
    """
    A decorator to add a compilation spinner for numba.jit (simplified version)
    
    Parameters:
        spinner_text (str): Custom text to display in the spinner
    
    Usage:
        @jit_with_spinner
        @jit_with_spinner(spinner_text="Custom compiling message...")
    """
    def decorator(jit_func):
        if rank > 0:
            return jit_func
        
        compiled = False
        text = spinner_text or f"Compiling numba function {jit_func.__name__}..."
        
        @wraps(jit_func)
        def wrapper(*args, **kwargs):
            nonlocal compiled
            if not compiled:
                with yaspin(text=text, color="cyan") as sp:
                    result = jit_func(*args, **kwargs)
                compiled = True
                return result
            return jit_func(*args, **kwargs)
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)