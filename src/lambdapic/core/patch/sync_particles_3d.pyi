import numpy as np

from ..particles import ParticlesBase
from .patch import Patch3D


def get_npart_to_extend_3d(
    particles_list: list[ParticlesBase],
    patch_list: list[Patch3D],
    npatches: int,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Count particle storage changes required after 3D boundary exchange.

    Parameters
    ----------
    particles_list : list[ParticlesBase]
        Particle containers for each local patch. The C
        extension reads ``x``, ``y``, ``z``, ``npart``, and ``is_dead`` from
        each entry.
    patch_list : list[Patch3D]
        3D patch objects for each local patch. The C extension
        reads patch bounds and ``neighbor_ipatch`` from each entry.
    npatches : int
        Number of patches represented in ``particles_list`` and
        ``patch_list``.
    dx : float
        Cell size along x.
    dy : float
        Cell size along y.
    dz : float
        Cell size along z.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple ``(npart_to_extend_array, npart_incoming_array,
        npart_outgoing_array, npart_array)``. Each item is a NumPy integer
        array: required particle-capacity extension per patch, incoming
        particle count per patch, outgoing particle counts flattened by patch
        and boundary, and final live-particle count per patch.
    """
    ...


def fill_particles_from_boundary_3d(
    particles_list: list[ParticlesBase],
    patch_list: list[Patch3D],
    npart_incoming_array: np.ndarray,
    npart_outgoing_array: np.ndarray,
    npatches: int,
    dx: float,
    dy: float,
    dz: float,
    xmin_global: float,
    xmax_global: float,
    ymin_global: float,
    ymax_global: float,
    zmin_global: float,
    zmax_global: float,
    attrs: list[str],
) -> None:
    """Fill dead slots with incoming boundary particles in 3D.

    Parameters
    ----------
    particles_list : list[ParticlesBase]
        Particle containers for each local patch. Attribute
        arrays named by ``attrs`` are copied from outgoing neighbors into
        dead slots.
    patch_list : list[Patch3D]
        3D patch objects for each local patch. The C extension
        reads patch bounds and ``neighbor_ipatch`` from each entry.
    npart_incoming_array : np.ndarray
        NumPy integer array containing incoming particle
        counts per patch.
    npart_outgoing_array : np.ndarray
        NumPy integer array containing outgoing particle
        counts flattened by patch and boundary.
    npatches : int
        Number of patches represented in ``particles_list`` and
        ``patch_list``.
    dx : float
        Cell size along x.
    dy : float
        Cell size along y.
    dz : float
        Cell size along z.
    xmin_global : float
        Minimum x coordinate of the global domain.
    xmax_global : float
        Maximum x coordinate of the global domain.
    ymin_global : float
        Minimum y coordinate of the global domain.
    ymax_global : float
        Maximum y coordinate of the global domain.
    zmin_global : float
        Minimum z coordinate of the global domain.
    zmax_global : float
        Maximum z coordinate of the global domain.
    attrs : list[str]
        Particle attribute names to copy. Must include ``"x"``, ``"y"``,
        and ``"z"``.

    Returns
    -------
    None
    """
    ...
