from typing import List
import numpy as np
from ...particles import ParticlesBase
from ...fields import Fields

def unified_boris_pusher_cpu_3d(
    particles_list: List[ParticlesBase], 
    fields_list: List[Fields], 
    npatches: int, 
    dt: float, q: float, m: float
):
    """
    This pusher integrates interpolation, boris pusher and current deposition.

    Parameters
    ----------
    particles_list : List[ParticlesBase]
        List of particles of all patches. 
    fields_list : List[Fields]
        List of fields of all patches. 
    npatches : int
        Number of patches. 
    dt : float
        Time step. 
    q : float
        Charge of the species. 
    m : float
        Mass
    """
    pass