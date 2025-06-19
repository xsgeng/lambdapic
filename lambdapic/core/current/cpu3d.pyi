from typing import List
import numpy as np
from ..particles import ParticlesBase
from ..fields import Fields

def current_deposition_cpu_3d(
    fields_list: List[Fields],
    particles_list: List[ParticlesBase],
    npatches: int,
    dt: float,
    q: float
) -> None: ...
