#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include "../utils/cutils.h"

#undef INDEX2
#undef INDEX3

#define INDEX2(i, j) \
    ((j) >= 0 ? (j) : (j) + (NY)) + \
    ((i) >= 0 ? (i) : (i) + (NX)) * (NY)

#define INDEX3(i, j, k) \
    ((k) >= 0 ? (k) : (k) + (NZ)) + \
    ((j) >= 0 ? (j) : (j) + (NY)) * (NZ) + \
    ((i) >= 0 ? (i) : (i) + (NX)) * (NY) * (NZ)


enum Boundary3D {
    // faces
    XMIN = 0,
    XMAX,
    YMIN,
    YMAX,
    ZMIN,
    ZMAX,
    // egdes
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
    NUM_BOUNDARIES
};

static const enum Boundary3D OPPOSITE_BOUNDARY[NUM_BOUNDARIES] = {
    // faces
    XMAX,
    XMIN,
    YMAX,
    YMIN,
    ZMAX,
    ZMIN,
    // egdes
    XMAXYMAX,
    XMAXYMIN,
    XMAXZMAX,
    XMAXZMIN,
    XMINYMAX,
    XMINYMIN,
    XMINZMAX,
    XMINZMIN,
    YMAXZMAX,
    YMAXZMIN,
    YMINZMAX,
    YMINZMIN,
    // vertices
    XMAXYMAXZMAX,
    XMAXYMAXZMIN,
    XMAXYMINZMAX,
    XMAXYMINZMIN,
    XMINYMAXZMAX,
    XMINYMAXZMIN,
    XMINYMINZMAX,
    XMINYMINZMIN
};

static PyObject* sync_currents_3d(PyObject* self, PyObject* args) {
    PyObject *fields_list, *patches_list;
    npy_intp npatches, nx, ny, nz, ng;

    if (!PyArg_ParseTuple(args, "OOnnnnn", 
        &fields_list, &patches_list,
        &npatches, &nx, &ny, &nz, &ng)) {
        return NULL;
    }
    npy_intp NX = nx+2*ng;
    npy_intp NY = ny+2*ng;
    npy_intp NZ = nz+2*ng;
    AUTOFREE double **jx = get_attr_array_double(fields_list, npatches, "jx");
    AUTOFREE double **jy = get_attr_array_double(fields_list, npatches, "jy"); 
    AUTOFREE double **jz = get_attr_array_double(fields_list, npatches, "jz");
    AUTOFREE double **rho = get_attr_array_double(fields_list, npatches, "rho");

    AUTOFREE npy_intp **neighbor_index_list = get_attr_array_int(patches_list, npatches, "neighbor_ipatch");

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp i = 0; i < npatches*4; i++) {
        int field_type = i % 4;
        npy_intp ipatch = i / 4;
        
        double** field = NULL;
        switch(field_type) {
            case 0: field = jx; break;
            case 1: field = jy; break;
            case 2: field = jz; break;
            case 3: field = rho; break;
        }

        #define SYNC_BOUNDARY_3D(src_patch, \
            ix_dst, ix_src, NX, \
            iy_dst, iy_src, NY, \
            iz_dst, iz_src, NZ) \
        for (npy_intp ix = 0; ix < NX; ix++) { \
            for (npy_intp iy = 0; iy < NY; iy++) { \
                for (npy_intp iz = 0; iz < NZ; iz++) { \
                    field[ipatch][INDEX3(ix_dst+ix, iy_dst+iy, iz_dst+iz)] += field[src_patch][INDEX3(ix_src+ix, iy_src+iy, iz_src+iz)]; \
                    field[src_patch][INDEX3(ix_src+ix, iy_src+iy, iz_src+iz)] = 0.0; \
                } \
            } \
        }

        const npy_intp* neighbor_index = neighbor_index_list[ipatch];
        // X direction sync
        if (neighbor_index[XMIN] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[XMIN], 
                0, nx, ng, 
                0, 0, ny, 
                0, 0, nz
            )
        }
        
        if (neighbor_index[XMAX] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[XMAX], 
                nx-ng, -ng, ng, 
                0, 0, ny, 
                0, 0, nz
            )
        }

        // Y direction sync
        if (neighbor_index[YMIN] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[YMIN], 
                0, 0, nx, 
                0, ny, ng, 
                0, 0, nz
            )
        }
        
        if (neighbor_index[YMAX] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[YMAX], 
                0, 0, nx, 
                ny-ng, -ng, ng, 
                0, 0, nz
            )
        }

        // Z direction sync
        if (neighbor_index[ZMIN] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[ZMIN], 
                0, 0, nx, 
                0, 0, ny, 
                0, nz, ng
            )
        }
        
        if (neighbor_index[ZMAX] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[ZMAX], 
                0, 0, nx, 
                0, 0, ny, 
                nz-ng, -ng, ng
            )
        }

        // Edge synchronization (3D)
        if (neighbor_index[XMINYMIN] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[XMINYMIN], 
                0, nx, ng, 
                0, ny, ng, 
                0, 0, nz
            )
        }
        
        if (neighbor_index[XMINYMAX] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[XMINYMAX], 
                0, nx, ng, 
                ny-ng, -ng, ng, 
                0, 0, nz
            )
        }

        if (neighbor_index[XMINZMIN] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[XMINZMIN], 
                0, nx, ng, 
                0, 0, ny, 
                0, nz, ng
            )
        }

        if (neighbor_index[XMINZMAX] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[XMINZMAX], 
                0, nx, ng, 
                0, 0, ny, 
                nz-ng, -ng, ng
            )
        }

        if (neighbor_index[XMAXYMIN] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[XMAXYMIN],
                nx-ng, -ng, ng,
                0, ny, ng,
                0, 0, nz
            )
        }
        
        if (neighbor_index[XMAXYMAX] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[XMAXYMAX],
                nx-ng, -ng, ng,
                ny-ng, -ng, ng,
                0, 0, nz
            )
        }

        if (neighbor_index[XMAXZMIN] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[XMAXZMIN],
                nx-ng, -ng, ng,
                0, 0, ny,
                0, nz, ng
            )
        }

        if (neighbor_index[XMAXZMAX] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[XMAXZMAX],
                nx-ng, -ng, ng,
                0, 0, ny,
                nz-ng, -ng, ng
            )
        }

        if (neighbor_index[YMINZMIN] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[YMINZMIN],
                0, 0, nx,
                0, ny, ng,
                0, nz, ng
            )
        }

        if (neighbor_index[YMINZMAX] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[YMINZMAX],
                0, 0, nx,
                0, ny, ng,
                nz-ng, -ng, ng
            )
        }

        if (neighbor_index[YMAXZMIN] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[YMAXZMIN],
                0, 0, nx,
                ny-ng, -ng, ng,
                0, nz, ng
            )
        }

        if (neighbor_index[YMAXZMAX] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[YMAXZMAX],
                0, 0, nx,
                ny-ng, -ng, ng,
                nz-ng, -ng, ng
            )
        }

        // Corner synchronization (3D)
        if (neighbor_index[XMINYMINZMIN] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[XMINYMINZMIN],
                0, nx, ng,
                0, ny, ng,
                0, nz, ng
            )
        }

        if (neighbor_index[XMINYMINZMAX] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[XMINYMINZMAX],
                0, nx, ng,
                0, ny, ng,
                nz-ng, -ng, ng
            )
        }

        if (neighbor_index[XMINYMAXZMIN] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[XMINYMAXZMIN],
                0, nx, ng,
                ny-ng, -ng, ng,
                0, nz, ng
            )
        }

        if (neighbor_index[XMINYMAXZMAX] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[XMINYMAXZMAX],
                0, nx, ng,
                ny-ng, -ng, ng,
                nz-ng, -ng, ng
            )
        }

        if (neighbor_index[XMAXYMINZMIN] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[XMAXYMINZMIN],
                nx-ng, -ng, ng,
                0, ny, ng,
                0, nz, ng
            )
        }

        if (neighbor_index[XMAXYMINZMAX] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[XMAXYMINZMAX],
                nx-ng, -ng, ng,
                0, ny, ng,
                nz-ng, -ng, ng
            )
        }

        if (neighbor_index[XMAXYMAXZMIN] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[XMAXYMAXZMIN],
                nx-ng, -ng, ng,
                ny-ng, -ng, ng,
                0, nz, ng
            )
        }

        if (neighbor_index[XMAXYMAXZMAX] >= 0) {
            SYNC_BOUNDARY_3D(neighbor_index[XMAXYMAXZMAX],
                nx-ng, -ng, ng,
                ny-ng, -ng, ng,
                nz-ng, -ng, ng
            )
        }

    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

static PyObject* sync_guard_fields_3d(PyObject* self, PyObject* args) {
    PyObject *fields_list, *patches_list, *attrs;
    npy_intp npatches, nx, ny, nz, ng;

    if (!PyArg_ParseTuple(args, "OOOnnnnn",
        &fields_list, &patches_list, &attrs,
        &npatches, &nx, &ny, &nz, &ng)) {
        return NULL;
    }

    npy_intp NX = nx+2*ng;
    npy_intp NY = ny+2*ng;
    npy_intp NZ = nz+2*ng;

    int nattrs = PyList_Size(attrs);
    AUTOFREE double ***attrs_list = malloc(nattrs * sizeof(double**));
    for (int i = 0; i < nattrs; i++) {
        attrs_list[i] = get_attr_array_double(fields_list, npatches, PyUnicode_AsUTF8(PyList_GetItem(attrs, i)));
    }

    AUTOFREE npy_intp **neighbor_index_list = get_attr_array_int(patches_list, npatches, "neighbor_ipatch");

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp i = 0; i < npatches*nattrs; i++) {
        int field_type = i % nattrs;
        npy_intp ipatch = i / nattrs;
        
        double** field = attrs_list[field_type];

        #define SYNC_GUARD_3D(src_patch, \
            ix_dst, ix_src, NX, \
            iy_dst, iy_src, NY, \
            iz_dst, iz_src, NZ) \
        for (npy_intp ix = 0; ix < NX; ix++) { \
            for (npy_intp iy = 0; iy < NY; iy++) { \
                for (npy_intp iz = 0; iz < NZ; iz++) { \
                    field[ipatch][INDEX3(ix_dst+ix, iy_dst+iy, iz_dst+iz)] = field[src_patch][INDEX3(ix_src+ix, iy_src+iy, iz_src+iz)]; \
                } \
            } \
        }

        const npy_intp* neighbor_index = neighbor_index_list[ipatch];

        // Faces
        if (neighbor_index[XMIN] >= 0) {
            SYNC_GUARD_3D(neighbor_index[XMIN], 
                -ng, nx-ng, ng, 
                0, 0, ny,
                0, 0, nz
            )
        }
        
        if (neighbor_index[XMAX] >= 0) {
            SYNC_GUARD_3D(neighbor_index[XMAX], 
                nx, 0, ng, 
                0, 0, ny,
                0, 0, nz
            )
        }

        if (neighbor_index[YMIN] >= 0) {
            SYNC_GUARD_3D(neighbor_index[YMIN], 
                0, 0, nx, 
                -ng, ny-ng, ng,
                0, 0, nz
            )
        }
        
        if (neighbor_index[YMAX] >= 0) {
            SYNC_GUARD_3D(neighbor_index[YMAX], 
                0, 0, nx, 
                ny, 0, ng,
                0, 0, nz
            )
        }

        if (neighbor_index[ZMIN] >= 0) {
            SYNC_GUARD_3D(neighbor_index[ZMIN], 
                0, 0, nx, 
                0, 0, ny,
                -ng, nz-ng, ng
            )
        }
        
        if (neighbor_index[ZMAX] >= 0) {
            SYNC_GUARD_3D(neighbor_index[ZMAX], 
                0, 0, nx, 
                0, 0, ny,
                nz, 0, ng
            )
        }

        // Edges
        if (neighbor_index[XMINYMIN] >= 0) {
            SYNC_GUARD_3D(neighbor_index[XMINYMIN], 
                -ng, nx-ng, ng, 
                -ng, ny-ng, ng,
                0, 0, nz
            )
        }
        
        if (neighbor_index[XMAXYMIN] >= 0) {
            SYNC_GUARD_3D(neighbor_index[XMAXYMIN], 
                nx, 0, ng, 
                -ng, ny-ng, ng,
                0, 0, nz
            )
        }
        
        if (neighbor_index[XMINYMAX] >= 0) {
            SYNC_GUARD_3D(neighbor_index[XMINYMAX], 
                -ng, nx-ng, ng, 
                ny, 0, ng,
                0, 0, nz
            )
        }
        
        if (neighbor_index[XMAXYMAX] >= 0) {
            SYNC_GUARD_3D(neighbor_index[XMAXYMAX], 
                nx, 0, ng, 
                ny, 0, ng,
                0, 0, nz
            )
        }

        if (neighbor_index[XMINZMIN] >= 0) {
            SYNC_GUARD_3D(neighbor_index[XMINZMIN], 
                -ng, nx-ng, ng, 
                0, 0, ny,
                -ng, nz-ng, ng
            )
        }
        
        if (neighbor_index[XMAXZMIN] >= 0) {
            SYNC_GUARD_3D(neighbor_index[XMAXZMIN], 
                nx, 0, ng, 
                0, 0, ny,
                -ng, nz-ng, ng
            )
        }

        if (neighbor_index[XMINZMAX] >= 0) {
            SYNC_GUARD_3D(neighbor_index[XMINZMAX], 
                -ng, nx-ng, ng, 
                0, 0, ny,
                nz, 0, ng
            )
        }

        if (neighbor_index[XMAXZMAX] >= 0) {
            SYNC_GUARD_3D(neighbor_index[XMAXZMAX], 
                nx, 0, ng, 
                0, 0, ny,
                nz, 0, ng
            )
        }

        if (neighbor_index[YMINZMIN] >= 0) {
            SYNC_GUARD_3D(neighbor_index[YMINZMIN], 
                0, 0, nx, 
                -ng, ny-ng, ng,
                -ng, nz-ng, ng
            )
        }

        if (neighbor_index[YMAXZMIN] >= 0) {
            SYNC_GUARD_3D(neighbor_index[YMAXZMIN],
                0, 0, nx, 
                ny, 0, ng,
                -ng, nz-ng, ng
            )
        }

        if (neighbor_index[YMINZMAX] >= 0) {
            SYNC_GUARD_3D(neighbor_index[YMINZMAX], 
                0, 0, nx, 
                -ng, ny-ng, ng,
                nz, 0, ng
            )
        }

        if (neighbor_index[YMAXZMAX] >= 0) {
            SYNC_GUARD_3D(neighbor_index[YMAXZMAX], 
                0, 0, nx, 
                ny, 0, ng,
                nz, 0, ng
            )
        }

        // Vertices
        if (neighbor_index[XMINYMINZMIN] >= 0) {
            SYNC_GUARD_3D(neighbor_index[XMINYMINZMIN], 
                -ng, nx-ng, ng, 
                -ng, ny-ng, ng,
                -ng, nz-ng, ng
            )
        }

        if (neighbor_index[XMINYMINZMAX] >= 0) {
            SYNC_GUARD_3D(neighbor_index[XMINYMINZMAX], 
                -ng, nx-ng, ng, 
                -ng, ny-ng, ng,
                nz, 0, ng
            )
        }

        if (neighbor_index[XMINYMAXZMIN] >= 0) {
            SYNC_GUARD_3D(neighbor_index[XMINYMAXZMIN], 
                -ng, nx-ng, ng, 
                ny, 0, ng,
                -ng, nz-ng, ng
            )
        }

        if (neighbor_index[XMINYMAXZMAX] >= 0) {
            SYNC_GUARD_3D(neighbor_index[XMINYMAXZMAX], 
                -ng, nx-ng, ng, 
                ny, 0, ng,
                nz, 0, ng
            )
        }

        if (neighbor_index[XMAXYMINZMIN] >= 0) {
            SYNC_GUARD_3D(neighbor_index[XMAXYMINZMIN], 
                nx, 0, ng, 
                -ng, ny-ng, ng,
                -ng, nz-ng, ng
            )
        }

        if (neighbor_index[XMAXYMINZMAX] >= 0) {
            SYNC_GUARD_3D(neighbor_index[XMAXYMINZMAX], 
                nx, 0, ng, 
                -ng, ny-ng, ng,
                nz, 0, ng
            )
        }

        if (neighbor_index[XMAXYMAXZMIN] >= 0) {
            SYNC_GUARD_3D(neighbor_index[XMAXYMAXZMIN], 
                nx, 0, ng, 
                ny, 0, ng,
                -ng, nz-ng, ng
            )
        }

        if (neighbor_index[XMAXYMAXZMAX] >= 0) {
            SYNC_GUARD_3D(neighbor_index[XMAXYMAXZMAX], 
                nx, 0, ng, 
                ny, 0, ng,
                nz, 0, ng
            )
        }
    }
    Py_END_ALLOW_THREADS
    
    for (int i = 0; i < nattrs; i++) {
        free(attrs_list[i]);
    }
    Py_RETURN_NONE;
}

static PyMethodDef Methods[] = {
    {"sync_currents_3d", sync_currents_3d, METH_VARARGS, "Synchronize currents between patches (3D)"},
    {"sync_guard_fields_3d", sync_guard_fields_3d, METH_VARARGS, "Synchronize guard cells between patches (3D)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "sync_fields3d",
    NULL,
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_sync_fields3d(void) {
    import_array();
    return PyModule_Create(&module);
}
