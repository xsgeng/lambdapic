import numpy as np
from mpi4py.MPI import Comm

from ..particles import ParticlesBase
from ..patch.patch import Patch


def get_npart_to_extend_2d(
    particles_list: list[ParticlesBase],
    patch_list: list[Patch],
    comm: Comm,
    npatches: int,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count incoming particles and required particle-array extension in 2D.

    Parameters
    ----------
    particles_list : list[ParticlesBase]
        Particle containers for the local patches. Each particle
        container must provide ``x``, ``y``, ``npart``, and ``is_dead``.
    patch_list : list[Patch]
        Patch containers matching ``particles_list`` order. Each patch
        provides neighbor indices, neighbor ranks, global patch index, and 2D
        patch bounds.
    comm : Comm
        MPI communicator used to exchange outgoing-particle counts.
    npatches : int
        Number of local patches represented in ``particles_list`` and
        ``patch_list``.
    dx : float
        Cell size in the x direction.
    dy : float
        Cell size in the y direction.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of three NumPy arrays: ``npart_to_extend`` with shape
        ``(npatches,)``, ``npart_incoming`` with shape ``(npatches, 8)``, and
        ``npart_outgoing`` with shape ``(npatches, 8)``.
    """
    ...


def fill_particles_from_boundary_2d(
    particles_list: list[ParticlesBase],
    patch_list: list[Patch],
    npart_incoming_array: np.ndarray,
    npart_outgoing_array: np.ndarray,
    comm: Comm,
    npatches: int,
    dx: float,
    dy: float,
    xmin_global: float,
    xmax_global: float,
    ymin_global: float,
    ymax_global: float,
    attrs: list[str],
) -> None:
    """Move outgoing particles into neighboring 2D patches across MPI ranks.

    Parameters
    ----------
    particles_list : list[ParticlesBase]
        Particle containers for the local patches. Each particle
        container must provide ``x``, ``y``, ``is_dead``, and every array named
        in ``attrs``.
    patch_list : list[Patch]
        Patch containers matching ``particles_list`` order. Each patch
        provides neighbor indices, neighbor ranks, global patch index, and 2D
        patch bounds.
    npart_incoming_array : np.ndarray
        Incoming particle counts per local patch and 2D
        boundary, as returned by ``get_npart_to_extend_2d``.
    npart_outgoing_array : np.ndarray
        Outgoing particle counts per local patch and 2D
        boundary, as returned by ``get_npart_to_extend_2d``.
    comm : Comm
        MPI communicator used to exchange particle attribute buffers.
    npatches : int
        Number of local patches represented in ``particles_list`` and
        ``patch_list``.
    dx : float
        Cell size in the x direction.
    dy : float
        Cell size in the y direction.
    xmin_global : float
        Minimum x coordinate of the global domain.
    xmax_global : float
        Maximum x coordinate of the global domain.
    ymin_global : float
        Minimum y coordinate of the global domain.
    ymax_global : float
        Maximum y coordinate of the global domain.
    attrs : list[str]
        Particle attribute names to exchange. Must include ``"x"`` and
        ``"y"`` so periodic wrapping can be applied.

    Returns
    -------
    None
    """
    ...
