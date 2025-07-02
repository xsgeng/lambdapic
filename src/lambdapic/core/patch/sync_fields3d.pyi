from typing import List
import numpy as np
from ...fields import Fields
from ...patch import Patch3D


def sync_currents_3d(
    fields_list: List[Fields],
    patches_list: List[Patch3D],
    npatches: int, nx: int, ny: int, nz: int, ng: int
):
    """
    Synchronize currents between patches in 3D.
    
    Parameters
    ----------
    fields_list : List[Fields]
        List of fields of all patches.
    patches_list : List[Patch3D]
        List of patches
    npatches : int
        Number of patches.
    nx : int
        Number of cells in x direction.
    ny : int
        Number of cells in y direction.
    nz : int
        Number of cells in z direction.
    ng : int
        Number of guard cells.
    """
    pass

def sync_guard_fields_3d(
    fields_list: List[Fields],
    patches_list: List[Patch3D],
    attrs: list[str],
    npatches: int, nx: int, ny: int, nz: int, ng: int,
):
    """
    Synchronize guard cells between patches for custom field attributes in 3D.
    
    Parameters
    ----------
    fields_list : List[Fields]
        List of fields of all patches
    patches_list : List[Patch3D]
        List of patches
    attrs : list[str]
        List of field attributes to synchronize
    npatches : int
        Number of patches
    nx : int
        Number of cells in x direction (excluding guards)
    ny : int
        Number of cells in y direction (excluding guards)
    nz : int
        Number of cells in z direction (excluding guards)
    ng : int
        Number of guard cells
    """
    pass

