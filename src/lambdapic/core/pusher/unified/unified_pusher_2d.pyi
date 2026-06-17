from ...fields import Fields
from ...particles import ParticlesBase


def unified_boris_pusher_cpu_2d(
    particles_list: list[ParticlesBase],
    fields_list: list[Fields],
    npatches: int,
    dt: float,
    q: float,
    m: float,
) -> None:
    """Run the unified 2D interpolation, Boris push, and current deposition kernel.

    Parameters
    ----------
    particles_list : list[ParticlesBase]
        Particle objects for each patch.
    fields_list : list[Fields]
        Field objects for each patch.
    npatches : int
        Number of patches to process.
    dt : float
        Time step.
    q : float
        Species charge.
    m : float
        Species mass.
    """
    ...
