import numpy as np
from mpi4py.MPI import Comm
from ..particles import Particles
from ..patch.patch import Patch

def get_npart_to_extend_3d(
    particles_list: list[Particles],
    patch_list: list[Patch],
    comm: Comm,
    npatches: int, dx: float, dy: float, dz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Count the number of particles to be extended and return the number of new particles.

    Parameters
    ----------
    particles_list : list[Particles]
        List of particle objects.
    patch_list : list[Patch]
        List of patch objects.
    comm : Comm
        MPI communicator.
    npatches : int
        Number of patches.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    dz : float
        Grid spacing in the z-direction
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing three numpy arrays:
        - npart_to_extend: Number of particles to extend.
        - npart_incoming: Number of incoming particles.
        - npart_outgoing: Number of outgoing particles.
    """
    pass

def fill_particles_from_boundary_3d(
    particles_list: list[Particles],
    patch_list: list[Patch],
    npart_incoming_array: np.ndarray,
    npart_outgoing_array: np.ndarray,
    comm: Comm,
    npatches: int,
    dx: float, dy: float, dz: float,
    attrs: list[str]
) -> None:
    """
    Fill particles from boundary using MPI.

    Parameters
    ----------
    particles_list : list[Particles]
        List of particle objects.
    patch_list : list[Patch]
        List of patch objects.
    npart_incoming_array : np.ndarray
        Array of incoming particle counts.
    npart_outgoing_array : np.ndarray
        Array of outgoing particle counts.
    comm : Comm
        MPI communicator.
    npatches : int
        Number of patches.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    dz : float
        Grid spacing in the z-direction
    attrs : list[str]
        List of attributes to be synced
    """
    pass
