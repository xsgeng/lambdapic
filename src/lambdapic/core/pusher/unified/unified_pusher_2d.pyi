from ...fields import Fields
from ...particles import ParticlesBase, SpinParticles


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


def unified_boris_tbmt_pusher_cpu_2d(
    particles_list: list[SpinParticles],
    fields_list: list[Fields],
    npatches: int,
    dt: float,
    q: float,
    m: float,
    ae: float,
) -> None:
    """Run the unified 2D interpolation, Boris+TBMT spin push, and current deposition kernel.

    Same as ``unified_boris_pusher_cpu_2d`` but additionally advances the
    spin of each particle with the TBMT (Thomas-Bargmann-Michel-Telegdi)
    equation. The orbit is identical to the plain Boris push.

    Parameters
    ----------
    particles_list : list[SpinParticles]
        Particle objects (with sx/sy/sz) for each patch.
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
    ae : float
        Anomalous magnetic moment a = (g-2)/2.
    """
    ...
