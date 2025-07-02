import numpy as np
from typing import List

def sort_particles_patches_3d(
    grid_cell_count_list: List[np.ndarray],
    cell_bound_min_list: List[np.ndarray],
    cell_bound_max_list: List[np.ndarray],
    x0s: List[float],
    y0s: List[float],
    z0s: List[float],
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    npatches: int,
    particle_cell_indices_list: List[np.ndarray],
    sorted_indices_list: List[np.ndarray],
    x_list: List[np.ndarray],
    y_list: List[np.ndarray],
    z_list: List[np.ndarray],
    is_dead_list: List[np.ndarray],
    attrs_list: List[np.ndarray]
) -> None: ...

def _calculate_cell_index(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    is_dead: np.ndarray,
    npart: int,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    x0: float,
    y0: float,
    z0: float,
    particle_cell_indices: np.ndarray,
    grid_cell_count: np.ndarray
) -> None: ...

def _sorted_cell_bound(
    grid_cell_count: np.ndarray,
    cell_bound_min: np.ndarray,
    cell_bound_max: np.ndarray,
    nx: int,
    ny: int,
    nz: int
) -> None: ...

def _cycle_sort(
    cell_bound_min: np.ndarray,
    cell_bound_max: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    particle_cell_indices: np.ndarray,
    is_dead: np.ndarray,
    sorted_idx: np.ndarray
) -> int: ...
