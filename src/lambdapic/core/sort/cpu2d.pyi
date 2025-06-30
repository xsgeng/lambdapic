from typing import List

import numpy as np
from numpy.typing import NDArray

def sort_particles_patches_2d(
    x_list: List[NDArray[np.float64]],
    y_list: List[NDArray[np.float64]],
    is_dead_list: List[NDArray[np.bool_]],
    attrs_list: List[NDArray[np.float64]],
    x0s: List[float],
    y0s: List[float],
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    npatches: int,
    bin_count_list: List[NDArray[np.int64]],
    bin_count_not_list: List[NDArray[np.int64]],
    bin_start_counter_list: List[NDArray[np.int64]],
    bucket_index_list: List[NDArray[np.int64]],
    bucket_index_ref_list: List[NDArray[np.int64]],
    bucket_index_target_list: List[NDArray[np.int64]],
    buf_list: List[NDArray[np.float64]]
) -> None:
    """
    Sort particles into patches based on their cell indices.
    """
    ...

def _calculate_bucket_index(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    is_dead: NDArray[np.bool_],
    npart: int,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    x0: float,
    y0: float,
    particle_cell_indices: NDArray[np.int64],
    grid_cell_count: NDArray[np.int64]
) -> None:
    """
    Python interface to C function calculate_bucket_index.
    Calculate the cell index for each particle.
    """
    ...

def _bucket_sort(
    bin_count: NDArray[np.int64],
    bin_count_not: NDArray[np.int64],
    bin_start_counter: NDArray[np.int64],
    nx: int,
    ny: int,
    bucket_index: NDArray[np.int64],
    bucket_index_ref: NDArray[np.int64],
    npart: int,
    bucket_index_target: NDArray[np.int64],
    buf: NDArray[np.float64],
    is_dead: NDArray[np.bool_],
    attrs: List[NDArray[np.float64]],
    nattrs: int
) -> int:
    """
    Python interface to C function bucket_sort.
    Perform bucket sort on particles.
    """
    ...