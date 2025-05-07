from time import perf_counter_ns
from loguru import logger
import sys

logger.remove()
logger.level('Timer', no=15)
logger.add(sys.stdout, level='Timer', format="{level}: {message}")

class Timer:
    def __init__(self, name=None, norm=1.0, unit="ms"):
        self.name = name
        self.unit = unit
        if unit == "s":
            self.norm = 1e9
        elif unit == "ms":
            self.norm = 1e6
        elif unit == "us":
            self.norm = 1e3
        elif unit == "ns":
            self.norm = 1.0
        else:
            raise ValueError(f"Unknown time unit {unit}")
        
        self.norm *= float(norm)
        
    def __enter__(self):
        self.start = perf_counter_ns()
        return self
    
    def __exit__(self, *args):
        self.end = perf_counter_ns()
        self.interval = (self.end - self.start) / self.norm
        logger.log('Timer', f"{self.name} took {self.interval:.1f} {self.unit}")