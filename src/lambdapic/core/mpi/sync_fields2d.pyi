from mpi4py.MPI import Comm

from ..fields import Fields
from ..patch.patch import Patch


def sync_currents_2d(
    fields_list: list[Fields],
    patches_list: list[Patch],
    comm: Comm,
    npatches: int,
    nx: int,
    ny: int,
    ng: int,
) -> None:
    """Synchronize current and charge-density boundary values across MPI ranks.

    This is a convenience wrapper that calls :func:`sync_currents_2d_start`
    followed by :func:`sync_currents_2d_wait`.

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
    ng : int
        Number of guard cells on each patch boundary.

    Returns
    -------
    None
    """
    ...


def sync_currents_2d_start(
    fields_list: list[Fields],
    patches_list: list[Patch],
    comm: Comm,
    npatches: int,
    nx: int,
    ny: int,
    ng: int,
) -> object:
    """Post non-blocking MPI sends/receives for cross-rank current exchange.

    Packs boundary current/charge values into send buffers, zeroes the
    source cells, and posts ``MPI_Isend`` / ``MPI_Irecv`` for every
    cross-rank boundary.  Returns an opaque capsule handle that must be
    passed to :func:`sync_currents_2d_wait` to finalise the exchange.

    The caller **must not** modify ``jx``/``jy``/``jz``/``rho`` arrays
    between ``_start`` and ``_wait``.

    Parameters
    ----------
    Same as :func:`sync_currents_2d`.

    Returns
    -------
    object
        Opaque capsule handle for :func:`sync_currents_2d_wait`.
    """
    ...


def sync_currents_2d_wait(handle: object) -> None:
    """Finalise an asynchronous current sync started by ``_start``.

    Waits for all receives, accumulates received currents into the field
    arrays, waits for sends, and frees all temporary buffers.

    Parameters
    ----------
    handle : object
        Capsule returned by :func:`sync_currents_2d_start`.

    Returns
    -------
    None
    """
    ...


def sync_guard_fields_2d(
    fields_list: list[Fields],
    patches_list: list[Patch],
    comm: Comm,
    attrs: list[str],
    npatches: int,
    nx: int,
    ny: int,
    ng: int,
) -> None:
    """Synchronize selected field guard-cell arrays across MPI ranks.

    This is a convenience wrapper that calls :func:`sync_guard_fields_2d_start`
    followed by :func:`sync_guard_fields_2d_wait`.

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
    ng : int
        Number of guard cells on each patch boundary.

    Returns
    -------
    None
    """
    ...


def sync_guard_fields_2d_start(
    fields_list: list[Fields],
    patches_list: list[Patch],
    comm: Comm,
    attrs: list[str],
    npatches: int,
    nx: int,
    ny: int,
    ng: int,
) -> object:
    """Post non-blocking MPI sends/receives for cross-rank guard-cell exchange.

    Creates MPI derived datatypes (subarrays) for every boundary, then
    posts ``MPI_Isend`` / ``MPI_Irecv`` for every cross-rank boundary and
    attribute.  Returns an opaque capsule handle that must be passed to
    :func:`sync_guard_fields_2d_wait`.

    The caller **must not** modify the guard-cell region of any field
    array named in ``attrs`` between ``_start`` and ``_wait``.

    Parameters
    ----------
    Same as :func:`sync_guard_fields_2d`.

    Returns
    -------
    object
        Opaque capsule handle for :func:`sync_guard_fields_2d_wait`.
    """
    ...


def sync_guard_fields_2d_wait(handle: object) -> None:
    """Finalise an asynchronous guard-field sync started by ``_start``.

    Waits for all MPI requests to complete and frees all temporary
    resources (datatypes, request arrays).

    Parameters
    ----------
    handle : object
        Capsule returned by :func:`sync_guard_fields_2d_start`.

    Returns
    -------
    None
    """
    ...
