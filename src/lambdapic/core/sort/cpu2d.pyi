from numpy.typing import NDArray
from typing import List

def sort_particles_patches_2d(
    # grid
    grid_cell_count_list: List[NDArray[int]], 
    cell_bound_min_list: List[NDArray[int]], cell_bound_max_list: List[NDArray[int]], 
    x0s: List[float], y0s: List[float],
    nx: int, ny: int, dx: float, dy: float, 
    npatches: int,
    # particle
    particle_cell_indices_list: List[NDArray[int]], 
    sorted_indices_list: List[NDArray[int]], 
    x_list: List[NDArray[float]], y_list: List[NDArray[float]], 
    is_dead_list: List[NDArray[bool]],
    attrs_list: List[NDArray[float]]
) -> None: 
    """
    Sort particles into patches based on their cell indices.
    """
    ...

def _calculate_cell_index(
    x: NDArray[float], y: NDArray[float], is_dead: NDArray[bool], 
    npart: int, nx: int, ny: int, dx: float, dy: float, x0: float, y0: float, 
    particle_cell_indices: NDArray[int], grid_cell_count: NDArray[int]
) -> None:
    """
    Calculate the cell index for each particle.
    """
    ...

def _sorted_cell_bound(
    grid_cell_count: NDArray[int], cell_bound_min: NDArray[int], cell_bound_max: NDArray[int], 
    nx: int, ny: int,
) -> None:
    """
    Calculate the cell bounds for each cell.
    """
    ...

def _sorted_cell_bound(
    grid_cell_count: NDArray[int], cell_bound_min: NDArray[int], cell_bound_max: NDArray[int], 
    nx: int, ny: int,
) -> None:
    """
    Calculate the cell bounds for each cell.
    """
    ...