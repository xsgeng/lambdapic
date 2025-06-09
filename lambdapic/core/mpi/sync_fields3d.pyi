from typing import List
import numpy as np
from ..fields import Fields
from ..patch import Patch

def sync_currents_3d(
    fields_list: List[Fields],
    patches_list: List[Patch],
    comm,
    npatches: int, nx: int, ny: int, nz: int, ng: int
):
    """
    Synchronize currents between patches.
    
    Parameters
    ----------
    fields_list : List[Fields]
        List of fields of all patches.
    patches_list : List[Patch]
        List of patches
    npatches : int
        Number of patches.
    nx : int
        Number of cells in x direction.
    ny : int
        Number of cells in y direction.
    ng : int
        Number of guard cells.
    """
    pass

def sync_guard_fields_3d(
    fields_list: List[Fields],
    patches_list: List[Patch],
    comm,
    attrs: list[str],
    npatches: int, nx: int, ny: int, nz: int, ng: int
):
    """
    Synchronize guard cells between patches for E and B fields.
    
    Parameters
    ----------
    fields_list : List[Fields]
        List of fields of all patches containing E and B fields
    patches_list : List[Patch]
        List of patches
    npatches : int
        Number of patches
    nx : int
        Number of cells in x direction (excluding guards)
    ny : int
        Number of cells in y direction (excluding guards)
    ng : int
        Number of guard cells
    """
    pass
