import numpy as np
from numpy.typing import NDArray


def sort_particles_patches_2d(
    x_list: list[NDArray[np.float64]],
    y_list: list[NDArray[np.float64]],
    is_dead_list: list[NDArray[np.bool_]],
    attrs_list: list[NDArray[np.float64]],
    x0s: list[float],
    y0s: list[float],
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    npatches: int,
    bucket_count_list: list[NDArray[np.int64]],
    bucket_bound_min_list: list[NDArray[np.int64]],
    bucket_bound_max_list: list[NDArray[np.int64]],
    bucket_count_not_list: list[NDArray[np.int64]],
    bin_start_counter_list: list[NDArray[np.int64]],
    particle_index_list: list[NDArray[np.int64]],
    particle_index_ref_list: list[NDArray[np.int64]],
    particle_index_target_list: list[NDArray[np.int64]],
    buf_list: list[NDArray[np.float64]],
) -> None:
    """Sort particles in every 2D patch by bucket index.

    Parameters
    ----------
    x_list : list[NDArray[np.float64]]
        Per-patch particle x-coordinate arrays.
    y_list : list[NDArray[np.float64]]
        Per-patch particle y-coordinate arrays.
    is_dead_list : list[NDArray[np.bool_]]
        Per-patch particle death-state arrays.
    attrs_list : list[NDArray[np.float64]]
        Flattened per-patch particle attribute arrays to reorder.
    x0s : list[float]
        Per-patch lower x origins for bucket indexing.
    y0s : list[float]
        Per-patch lower y origins for bucket indexing.
    nx : int
        Number of buckets in x for each patch.
    ny : int
        Number of buckets in y for each patch.
    dx : float
        Bucket spacing in x.
    dy : float
        Bucket spacing in y.
    npatches : int
        Number of patches represented by the input lists.
    bucket_count_list : list[NDArray[np.int64]]
        Per-patch arrays receiving particle counts per bucket.
    bucket_bound_min_list : list[NDArray[np.int64]]
        Per-patch arrays receiving inclusive bucket starts.
    bucket_bound_max_list : list[NDArray[np.int64]]
        Per-patch arrays receiving exclusive bucket ends.
    bucket_count_not_list : list[NDArray[np.int64]]
        Per-patch work arrays for misplaced-particle counts.
    bin_start_counter_list : list[NDArray[np.int64]]
        Per-patch work arrays for bucket write counters.
    particle_index_list : list[NDArray[np.int64]]
        Per-patch work arrays receiving current bucket indices.
    particle_index_ref_list : list[NDArray[np.int64]]
        Per-patch work arrays receiving sorted bucket indices.
    particle_index_target_list : list[NDArray[np.int64]]
        Per-patch work arrays receiving move targets.
    buf_list : list[NDArray[np.float64]]
        Per-patch floating-point scratch buffers.

    Returns
    -------
    None
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
    particle_index: NDArray[np.int64],
    bucket_count: NDArray[np.int64],
) -> None:
    """Calculate 2D bucket indices and bucket counts for one patch.

    Parameters
    ----------
    x : NDArray[np.float64]
        Particle x-coordinate array.
    y : NDArray[np.float64]
        Particle y-coordinate array.
    is_dead : NDArray[np.bool_]
        Particle death-state array.
    npart : int
        Number of particles to process.
    nx : int
        Number of buckets in x.
    ny : int
        Number of buckets in y.
    dx : float
        Bucket spacing in x.
    dy : float
        Bucket spacing in y.
    x0 : float
        Lower x origin for bucket indexing.
    y0 : float
        Lower y origin for bucket indexing.
    particle_index : NDArray[np.int64]
        Output array receiving each particle's bucket index.
    bucket_count : NDArray[np.int64]
        Output array receiving particle counts per bucket.

    Returns
    -------
    None
    """
    ...


def _calculate_bucket_bound(
    bucket_count: NDArray[np.int64],
    bucket_bound_min: NDArray[np.int64],
    bucket_bound_max: NDArray[np.int64],
    nx: int,
    ny: int,
) -> None:
    """Calculate contiguous 2D bucket bounds from bucket counts.

    Parameters
    ----------
    bucket_count : NDArray[np.int64]
        Input array containing particle counts per bucket.
    bucket_bound_min : NDArray[np.int64]
        Output array receiving inclusive bucket starts.
    bucket_bound_max : NDArray[np.int64]
        Output array receiving exclusive bucket ends.
    nx : int
        Number of buckets in x.
    ny : int
        Number of buckets in y.

    Returns
    -------
    None
    """
    ...


def _bucket_sort(
    bucket_count: NDArray[np.int64],
    bucket_count_not: NDArray[np.int64],
    bucket_start_counter: NDArray[np.int64],
    nx: int,
    ny: int,
    particle_index: NDArray[np.int64],
    particle_index_ref: NDArray[np.int64],
    npart: int,
    particle_index_target: NDArray[np.int64],
    buf: NDArray[np.float64],
    is_dead: NDArray[np.bool_],
    attrs: list[NDArray[np.float64]],
    nattrs: int,
) -> int:
    """Reorder one patch's particle arrays into 2D bucket order.

    Parameters
    ----------
    bucket_count : NDArray[np.int64]
        Input array containing particle counts per bucket.
    bucket_count_not : NDArray[np.int64]
        Work array receiving counts of misplaced particles.
    bucket_start_counter : NDArray[np.int64]
        Work array used as per-bucket write counters.
    nx : int
        Number of buckets in x.
    ny : int
        Number of buckets in y.
    particle_index : NDArray[np.int64]
        Current bucket index for each particle.
    particle_index_ref : NDArray[np.int64]
        Work array receiving sorted bucket indices.
    npart : int
        Number of particles to process.
    particle_index_target : NDArray[np.int64]
        Work array receiving indices that must be moved.
    buf : NDArray[np.float64]
        Floating-point scratch buffer.
    is_dead : NDArray[np.bool_]
        Particle death-state array, reordered with the attributes.
    attrs : list[NDArray[np.float64]]
        Particle attribute arrays to reorder.
    nattrs : int
        Number of attribute arrays in attrs.

    Returns
    -------
    int
        Number of particles moved through the scratch buffer.
    """
    ...
