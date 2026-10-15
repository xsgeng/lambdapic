import numpy as np
from numpy.typing import NDArray

from ..particles import ParticlesBase


def seed_rng_states(
    seeds: NDArray[np.int64],
) -> NDArray[np.uint64]:
    """Create PCG32 RNG states from per-patch seeds.

    Parameters
    ----------
    seeds : NDArray[np.int64]
        One seed per patch.

    Returns
    -------
    NDArray[np.uint64]
        Array of shape ``(n, 2)`` holding ``[state, inc]`` per patch.
    """
    ...


def debye_length_cell(
    ux: NDArray[np.float64],
    uy: NDArray[np.float64],
    uz: NDArray[np.float64],
    inv_gamma: NDArray[np.float64],
    w: NDArray[np.float64],
    is_dead: NDArray[np.bool_],
    m: float,
    q: float,
    ip_start: int,
    ip_end: int,
    cell_vol: float,
) -> tuple[float, float]:
    """Compute inverse squared Debye length and density for one cell.

    Returns
    -------
    tuple[float, float]
        ``(inv_debye_length_square, density)``; ``(-1.0, 0.0)`` when no live
        particles, ``inf`` inverse-square when temperature is zero.
    """
    ...


def debye_length_patch(
    ux: NDArray[np.float64],
    uy: NDArray[np.float64],
    uz: NDArray[np.float64],
    inv_gamma: NDArray[np.float64],
    w: NDArray[np.float64],
    is_dead: NDArray[np.bool_],
    m: float,
    q: float,
    bucket_bound_min: NDArray[np.int64],
    bucket_bound_max: NDArray[np.int64],
    cell_vol: float,
    debye_inv2: NDArray[np.float64],
) -> None:
    """Accumulate per-cell inverse squared Debye length for one patch."""
    ...


def debye_length_patches(
    particles_list: list[ParticlesBase],
    m: float,
    q: float,
    bucket_bound_min_list: list[NDArray[np.int64]],
    bucket_bound_max_list: list[NDArray[np.int64]],
    cell_vol: float,
    debye_inv2_list: list[NDArray[np.float64]],
    total_density_list: list[NDArray[np.float64]],
    reset: int,
) -> None:
    """Accumulate per-cell inverse squared Debye length and density over all
    patches (OpenMP over patches).

    Parameters
    ----------
    particles_list : list[ParticlesBase]
        Particle objects for each patch; array attributes are fetched per
        call, so extended buffers are picked up automatically.
    reset : int
        Truthy to zero ``debye_inv2_list`` first (first species pass).
    """
    ...


def constrain_debye_length_patches(
    debye_inv2_list: list[NDArray[np.float64]],
    total_density_list: list[NDArray[np.float64]],
) -> None:
    """Floor the Debye length at the mean interparticle distance (OpenMP over
    patches)."""
    ...


def intra_collision_patches(
    particles_list: list[ParticlesBase],
    m: float,
    q: float,
    bucket_bound_min_list: list[NDArray[np.int64]],
    bucket_bound_max_list: list[NDArray[np.int64]],
    lnLambda: float,
    debye_inv2_list: list[NDArray[np.float64]],
    cell_vol: float,
    dt: float,
    rng_state_list: NDArray[np.uint64],
) -> None:
    """Intra-species binary collisions over all patches (OpenMP over patches).

    Parameters
    ----------
    particles_list : list[ParticlesBase]
        Particle objects for each patch; array attributes are fetched per
        call, so extended buffers are picked up automatically.
    lnLambda : float
        Fixed Coulomb logarithm; ``<= 0`` selects the varying (Debye-limited)
        path.
    rng_state_list : NDArray[np.uint64]
        ``(npatches, 2)`` PCG32 states, advanced in place.
    """
    ...


def inter_collision_patches(
    particles1_list: list[ParticlesBase],
    m1: float,
    q1: float,
    bucket_bound_min1_list: list[NDArray[np.int64]],
    bucket_bound_max1_list: list[NDArray[np.int64]],
    particles2_list: list[ParticlesBase],
    m2: float,
    q2: float,
    bucket_bound_min2_list: list[NDArray[np.int64]],
    bucket_bound_max2_list: list[NDArray[np.int64]],
    lnLambda: float,
    debye_inv2_list: list[NDArray[np.float64]],
    cell_vol: float,
    dt: float,
    rng_state_list: NDArray[np.uint64],
) -> None:
    """Inter-species binary collisions over all patches (OpenMP over
    patches)."""
    ...


def intra_collision_cell(
    ux: NDArray[np.float64],
    uy: NDArray[np.float64],
    uz: NDArray[np.float64],
    inv_gamma: NDArray[np.float64],
    w: NDArray[np.float64],
    is_dead: NDArray[np.bool_],
    m: float,
    q: float,
    ip_start: int,
    ip_end: int,
    lnLambda: float,
    debye_inv2: float,
    cell_vol: float,
    dt: float,
    rng: NDArray[np.uint64],
) -> None:
    """Intra-species collisions for one cell; ``rng`` is a ``(2,)`` uint64
    PCG32 state, advanced in place."""
    ...


def inter_collision_cell(
    ux1: NDArray[np.float64],
    uy1: NDArray[np.float64],
    uz1: NDArray[np.float64],
    inv_gamma1: NDArray[np.float64],
    w1: NDArray[np.float64],
    is_dead1: NDArray[np.bool_],
    m1: float,
    q1: float,
    ip_start1: int,
    ip_end1: int,
    ux2: NDArray[np.float64],
    uy2: NDArray[np.float64],
    uz2: NDArray[np.float64],
    inv_gamma2: NDArray[np.float64],
    w2: NDArray[np.float64],
    is_dead2: NDArray[np.bool_],
    m2: float,
    q2: float,
    ip_start2: int,
    ip_end2: int,
    lnLambda: float,
    debye_inv2: float,
    cell_vol: float,
    dt: float,
    rng: NDArray[np.uint64],
) -> None:
    """Inter-species collisions for one cell; ``rng`` is a ``(2,)`` uint64
    PCG32 state, advanced in place."""
    ...
