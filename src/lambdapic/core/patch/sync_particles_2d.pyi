from ..particles import ParticlesBase
from numpy import ndarray
from .patch import Patch2D

def get_npart_to_extend_2d(
    particles_list: list[ParticlesBase],
    patches_list: list['Patch2D'],
    npatches: int, dx: float, dy: float
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Get the number of particles to extend in each patch.
    
    Parameters
    ----------
    particles_list : List[ParticlesBase]
        List of particles of all patches.
    patches_list : List['Patch2D']
        List of patches
    npatches : int
        Number of patches.
    dx : float
        Cell size in x direction.
    dy : float
        Cell size in y direction.
    
    Returns
    -------
    npart_to_extend : ndarray
        Number of particles to extend in each patch.
    npart_incoming : ndarray
        Number of incoming particles in each patch.
    npart_outgoing : ndarray
        Number of particles outgoing to each boundary in each patch.
    """
    ...
    
def fill_particles_from_boundary_2d(
    particles_list: list[ParticlesBase],
    patches_list: list['Patch2D'],
    npart_incoming: ndarray,
    npart_outgoing: ndarray,
    npatches: int, dx: float, dy: float,
    xmin_global: float, xmax_global: float, ymin_global: float, ymax_global: float,
    attrs: list[str]
) -> None:
    """
    Fill the particles from the boundary.
    
    Parameters
    ----------
    particles_list : List[ParticlesBase]
        List of particles of all patches.
    patches_list : List['Patch2D']
        List of patches.
    npart_incoming : ndarray
        Number of incoming particles in each patch.
    npart_outgoing : ndarray
        Number of particles outgoing to each boundary in each patch.
    npatches : int
        Number of patches.
    dx : float
        Cell size in x direction.
    dy : float
        Cell size in y direction.
    xmin_global : float
        Minimum x coordinate of the global domain.
    xmax_global : float
        Maximum x coordinate of the global domain.
    ymin_global : float
        Minimum y coordinate of the global domain.
    ymax_global : float
        Maximum y coordinate of the global domain.
    attrs : List[str]
        List of attributes to be synced of the particles.
    """
    ...