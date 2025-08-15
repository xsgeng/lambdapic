import numpy as np
from typing import List

def sort_particles_patches_3d(
    x_list: List[np.ndarray],
    y_list: List[np.ndarray],
    z_list: List[np.ndarray],
    is_dead_list: List[np.ndarray],
    attrs_list: List[np.ndarray],
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
    
    bucket_count_list: List[np.ndarray],
    bucket_bound_min_list: List[np.ndarray],
    bucket_bound_max_list: List[np.ndarray],
    bucket_count_not_list: List[np.ndarray],
    bucket_start_counter_list: List[np.ndarray],

    particle_index_list: List[np.ndarray],
    particle_index_ref_list: List[np.ndarray],
    particle_index_target_list: List[np.ndarray],
    buf_list: List[np.ndarray]
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

def _bucket_sort_3d(
    bin_count: np.ndarray,
    bin_count_not: np.ndarray,
    bin_start_counter: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    bucket_index: np.ndarray,
    bucket_index_ref: np.ndarray,
    npart: int,
    bucket_index_target: np.ndarray,
    buf: np.ndarray,
    is_dead: np.ndarray,
    attrs: List[np.ndarray]
) -> int: ...
