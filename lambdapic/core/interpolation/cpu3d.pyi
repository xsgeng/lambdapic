from ..particles import ParticlesBase
from ..fields import Fields
from typing import List

def interpolation_patches_3d(
    particles_list: List[ParticlesBase], 
    fields_list: List[Fields], 
    npatches: int, 
) -> None: ...