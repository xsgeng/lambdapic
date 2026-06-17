from mpi4py.MPI import Comm

from ..fields import Fields
from ..patch.patch import Patch


def sync_currents_3d(
    fields_list: list[Fields],
    patches_list: list[Patch],
    comm: Comm,
    npatches: int,
    nx: int,
    ny: int,
    nz: int,
    ng: int,
) -> None:
    """Synchronize current and charge-density boundary values across MPI ranks.

    Parameters
    ----------
    fields_list : list[Fields]
        Field containers for the local patches. Each field object must
        provide ``jx``, ``jy``, ``jz``, and ``rho`` arrays.
    patches_list : list[Patch]
        Patch containers matching ``fields_list`` order. Each patch
        provides neighbor indices, neighbor ranks, and its global patch index.
    comm : Comm
        MPI communicator used for cross-rank current exchange.
    npatches : int
        Number of local patches represented in ``fields_list`` and
        ``patches_list``.
    nx : int
        Number of non-guard cells in the x direction per patch.
    ny : int
        Number of non-guard cells in the y direction per patch.
    nz : int
        Number of non-guard cells in the z direction per patch.
    ng : int
        Number of guard cells on each patch boundary.

    Returns
    -------
    None
    """
    ...


def sync_guard_fields_3d(
    fields_list: list[Fields],
    patches_list: list[Patch],
    comm: Comm,
    attrs: list[str],
    npatches: int,
    nx: int,
    ny: int,
    nz: int,
    ng: int,
) -> None:
    """Synchronize selected field guard-cell arrays across MPI ranks.

    Parameters
    ----------
    fields_list : list[Fields]
        Field containers for the local patches. Each field object must
        provide every array named in ``attrs``.
    patches_list : list[Patch]
        Patch containers matching ``fields_list`` order. Each patch
        provides neighbor indices, neighbor ranks, and its global patch index.
    comm : Comm
        MPI communicator used for cross-rank guard-cell exchange.
    attrs : list[str]
        Field attribute names to exchange, such as electric or magnetic
        field component names.
    npatches : int
        Number of local patches represented in ``fields_list`` and
        ``patches_list``.
    nx : int
        Number of non-guard cells in the x direction per patch.
    ny : int
        Number of non-guard cells in the y direction per patch.
    nz : int
        Number of non-guard cells in the z direction per patch.
    ng : int
        Number of guard cells on each patch boundary.

    Returns
    -------
    None
    """
    ...
