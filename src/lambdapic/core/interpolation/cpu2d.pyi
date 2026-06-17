from ..fields import Fields
from ..particles import ParticlesBase


def interpolation_patches_2d(
    particles_list: list[ParticlesBase],
    fields_list: list[Fields],
    npatches: int,
) -> None:
    """Interpolate 2D field values to particles on each patch.

    Parameters
    ----------
    particles_list : list[ParticlesBase]
        Particle objects for each patch.
    fields_list : list[Fields]
        Field objects for each patch.
    npatches : int
        Number of patches to process.
    """
    ...
