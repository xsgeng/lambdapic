#pragma once

#define LAMBDAPIC_BOUNDARY_3D_DEFINED

/*
    Shared 3D boundary topology enum.

    Single source of truth for the neighbour indexing used across the
    CPU C sync extensions, the GPU CUDA kernels, and the Python
    ``Boundary3D`` IntEnum in ``core/patch/patch.py``.

    The numeric values MUST stay in sync with the Python IntEnum.
    Renaming or reordering the enumerators here requires updating the
    Python enum in lockstep.

    Each translation unit includes this header and declares its own
    ``OPPOSITE_BOUNDARY_3D`` array with the storage qualifier it needs
    (``static const`` on CPU, ``__device__ __constant__`` on GPU) using
    the ``OPPOSITE_BOUNDARY_3D_INITIALIZER`` macro below.
*/

enum Boundary3D {
    // faces
    XMIN = 0,
    XMAX,
    YMIN,
    YMAX,
    ZMIN,
    ZMAX,
    // edges
    XMINYMIN,
    XMINYMAX,
    XMINZMIN,
    XMINZMAX,
    XMAXYMIN,
    XMAXYMAX,
    XMAXZMIN,
    XMAXZMAX,
    YMINZMIN,
    YMINZMAX,
    YMAXZMIN,
    YMAXZMAX,
    // vertices
    XMINYMINZMIN,
    XMINYMINZMAX,
    XMINYMAXZMIN,
    XMINYMAXZMAX,
    XMAXYMINZMIN,
    XMAXYMINZMAX,
    XMAXYMAXZMIN,
    XMAXYMAXZMAX,
    NUM_BOUNDARIES_3D
};

/*
    Opposite-boundary lookup table.

    ``OPPOSITE_BOUNDARY_3D[b]`` gives the boundary index that is the
    geometric opposite of ``b`` (e.g. XMIN <-> XMAX,
    XMINYMIN <-> XMAXYMAX, XMINYMINZMIN <-> XMAXYMAXZMAX). Used to map
    outgoing particles on one patch to incoming particles on the
    neighbour.
*/
#define OPPOSITE_BOUNDARY_3D_INITIALIZER { \
    /* faces */ \
    XMAX,            /* opposite of XMIN          */ \
    XMIN,            /* opposite of XMAX          */ \
    YMAX,            /* opposite of YMIN          */ \
    YMIN,            /* opposite of YMAX          */ \
    ZMAX,            /* opposite of ZMIN          */ \
    ZMIN,            /* opposite of ZMAX          */ \
    /* edges */ \
    XMAXYMAX,        /* opposite of XMINYMIN      */ \
    XMAXYMIN,        /* opposite of XMINYMAX      */ \
    XMAXZMAX,        /* opposite of XMINZMIN      */ \
    XMAXZMIN,        /* opposite of XMINZMAX      */ \
    XMINYMAX,        /* opposite of XMAXYMIN      */ \
    XMINYMIN,        /* opposite of XMAXYMAX      */ \
    XMINZMAX,        /* opposite of XMAXZMIN      */ \
    XMINZMIN,        /* opposite of XMAXZMAX      */ \
    YMAXZMAX,        /* opposite of YMINZMIN      */ \
    YMAXZMIN,        /* opposite of YMINZMAX      */ \
    YMINZMAX,        /* opposite of YMAXZMIN      */ \
    YMINZMIN,        /* opposite of YMAXZMAX      */ \
    /* vertices */ \
    XMAXYMAXZMAX,    /* opposite of XMINYMINZMIN  */ \
    XMAXYMAXZMIN,    /* opposite of XMINYMINZMAX  */ \
    XMAXYMINZMAX,    /* opposite of XMINYMAXZMIN  */ \
    XMAXYMINZMIN,    /* opposite of XMINYMAXZMAX  */ \
    XMINYMAXZMAX,    /* opposite of XMAXYMINZMIN  */ \
    XMINYMAXZMIN,    /* opposite of XMAXYMINZMAX  */ \
    XMINYMINZMAX,    /* opposite of XMAXYMAXZMIN  */ \
    XMINYMINZMIN     /* opposite of XMAXYMAXZMAX  */ \
}
