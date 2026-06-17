from ..fields import Fields
from .patch import Patch3D


def sync_currents_3d(
    fields_list: list[Fields],
    patches_list: list[Patch3D],
    npatches: int,
    nx: int,
    ny: int,
    nz: int,
    ng: int,
) -> None:
    """Synchronize current and charge-density guard contributions in 3D.

    Parameters
    ----------
    fields_list : list[Fields]
        Field containers for each local patch. The C extension reads
        ``jx``, ``jy``, ``jz``, and ``rho`` arrays from each entry.
    patches_list : list[Patch3D]
        3D patch objects for each local patch. The C extension
        reads ``neighbor_ipatch`` from each entry.
    npatches : int
        Number of patches represented in ``fields_list`` and
        ``patches_list``.
    nx : int
        Number of non-guard cells per patch along x.
    ny : int
        Number of non-guard cells per patch along y.
    nz : int
        Number of non-guard cells per patch along z.
    ng : int
        Number of guard cells on each patch boundary.

    Returns
    -------
    None
    """
    ...


def sync_guard_fields_3d(
    fields_list: list[Fields],
    patches_list: list[Patch3D],
    attrs: list[str],
    npatches: int,
    nx: int,
    ny: int,
    nz: int,
    ng: int,
) -> None:
    """Copy selected field guard cells from neighboring patches in 3D.

    Parameters
    ----------
    fields_list : list[Fields]
        Field containers for each local patch.
    patches_list : list[Patch3D]
        3D patch objects for each local patch. The C extension
        reads ``neighbor_ipatch`` from each entry.
    attrs : list[str]
        Names of field arrays to synchronize.
    npatches : int
        Number of patches represented in ``fields_list`` and
        ``patches_list``.
    nx : int
        Number of non-guard cells per patch along x.
    ny : int
        Number of non-guard cells per patch along y.
    nz : int
        Number of non-guard cells per patch along z.
    ng : int
        Number of guard cells on each patch boundary.

    Returns
    -------
    None
    """
    ...
