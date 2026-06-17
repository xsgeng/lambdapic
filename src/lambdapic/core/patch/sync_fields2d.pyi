from ..fields import Fields
from .patch import Patch


def sync_currents_2d(
    fields_list: list[Fields],
    patches_list: list[Patch],
    npatches: int,
    nx: int,
    ny: int,
    ng: int,
) -> None:
    """Synchronize current and charge-density guard contributions in 2D.

    Parameters
    ----------
    fields_list : list[Fields]
        Field containers for each local patch. The C extension reads
        ``jx``, ``jy``, ``jz``, and ``rho`` arrays from each entry.
    patches_list : list[Patch]
        Patch objects for each local patch. The C extension reads
        ``neighbor_ipatch`` from each entry.
    npatches : int
        Number of patches represented in ``fields_list`` and
        ``patches_list``.
    nx : int
        Number of non-guard cells per patch along x.
    ny : int
        Number of non-guard cells per patch along y.
    ng : int
        Number of guard cells on each patch boundary.

    Returns
    -------
    None
    """
    ...


def sync_guard_fields_2d(
    fields_list: list[Fields],
    patches_list: list[Patch],
    attrs: list[str],
    npatches: int,
    nx: int,
    ny: int,
    ng: int,
) -> None:
    """Copy selected field guard cells from neighboring patches in 2D.

    Parameters
    ----------
    fields_list : list[Fields]
        Field containers for each local patch.
    patches_list : list[Patch]
        Patch objects for each local patch. The C extension reads
        ``neighbor_ipatch`` from each entry.
    attrs : list[str]
        Names of field arrays to synchronize.
    npatches : int
        Number of patches represented in ``fields_list`` and
        ``patches_list``.
    nx : int
        Number of non-guard cells per patch along x.
    ny : int
        Number of non-guard cells per patch along y.
    ng : int
        Number of guard cells on each patch boundary.

    Returns
    -------
    None
    """
    ...
