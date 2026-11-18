#pragma once

#define LAMBDAPIC_BOUNDARY_2D_DEFINED

/*
    Shared 2D boundary topology enum.

    Single source of truth for the neighbour indexing used across the
    CPU C sync extensions, the GPU CUDA kernels, and the Python
    ``Boundary2D`` IntEnum in ``core/patch/patch.py``.

    The numeric values MUST stay in sync with the Python IntEnum.
    Renaming or reordering the enumerators here requires updating the
    Python enum in lockstep.

    Each translation unit includes this header and declares its own
    ``OPPOSITE_BOUNDARY_2D`` array with the storage qualifier it needs
    (``static const`` on CPU, ``__device__ __constant__`` on GPU) using
    the ``OPPOSITE_BOUNDARY_2D_INITIALIZER`` macro below.
*/

enum Boundary2D {
    XMIN = 0,
    XMAX,
    YMIN,
    YMAX,
    XMINYMIN,
    XMAXYMIN,
    XMINYMAX,
    XMAXYMAX,
    NUM_BOUNDARIES_2D
};

/*
    Opposite-boundary lookup table.

    ``OPPOSITE_BOUNDARY_2D[b]`` gives the boundary index that is the
    geometric opposite of ``b`` (e.g. XMIN <-> XMAX). Used to map
    outgoing particles on one patch to incoming particles on the
    neighbour.
*/
#define OPPOSITE_BOUNDARY_2D_INITIALIZER { \
    XMAX,      /* opposite of XMIN     */ \
    XMIN,      /* opposite of XMAX     */ \
    YMAX,      /* opposite of YMIN     */ \
    YMIN,      /* opposite of YMAX     */ \
    XMAXYMAX,  /* opposite of XMINYMIN */ \
    XMINYMAX,  /* opposite of XMAXYMIN */ \
    XMAXYMIN,  /* opposite of XMINYMAX */ \
    XMINYMIN   /* opposite of XMAXYMAX */ \
}
