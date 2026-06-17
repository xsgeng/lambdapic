from ..fields import Fields
from ..particles import ParticlesBase


def current_deposition_cpu_3d(
    fields_list: list[Fields],
    particles_list: list[ParticlesBase],
    npatches: int,
    dt: float,
    q: float,
) -> None:
    """Deposit particle charge and current onto 3D field patches.

    Parameters
    ----------
    fields_list : list[Fields]
        Field objects for each patch.
    particles_list : list[ParticlesBase]
        Particle objects for each patch.
    npatches : int
        Number of patches to process.
    dt : float
        Time step.
    q : float
        Species charge.
    """
    ...


def reset_current_cpu_3d(
    fields_list: list[Fields],
    npatches: int,
) -> None:
    """Reset 3D current and charge-density arrays to zero.

    Parameters
    ----------
    fields_list : list[Fields]
        Field objects for each patch.
    npatches : int
        Number of patches to process.
    """
    ...
