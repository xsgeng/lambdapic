#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>

#define GetPatchArrayData(list, ipatch) PyArray_DATA((PyArrayObject*)PyList_GetItem(list, ipatch))

static void calculate_cell_index(
    double* x, double* y, double* z, npy_bool* is_dead, 
    npy_intp npart, npy_intp nx, npy_intp ny, npy_intp nz,
    double dx, double dy, double dz, double x0, double y0, double z0,
    npy_intp* particle_cell_indices, npy_intp* grid_cell_count
) {
    npy_intp ix, iy, iz, ip, icell;
    for (ip = 0; ip < npart; ip++) {
        if (!is_dead[ip]) {
            ix = (npy_intp)floor((x[ip] - x0)/dx);
            iy = (npy_intp)floor((y[ip] - y0)/dy);
            iz = (npy_intp)floor((z[ip] - z0)/dz);
            icell = iz + iy*nz + ix*ny*nz;
            if (0 <= ix && ix < nx && 0 <= iy && iy < ny && 0 <= iz && iz < nz) {
                particle_cell_indices[ip] = icell;
                grid_cell_count[icell]++;
            } else {
                particle_cell_indices[ip] = -1;
            }
        } else {
            particle_cell_indices[ip] = -1;
        }
    }
}

static npy_intp cycle_sort(
    npy_intp* cell_bound_min, npy_intp* cell_bound_max,
    npy_intp nx, npy_intp ny, npy_intp nz,
    npy_intp* particle_cell_indices, npy_bool* is_dead, npy_intp* sorted_idx
) {
    npy_intp ops = 0;
    npy_intp ix, iy, iz, ip, ip_src, ip_dst, icell_src, icell_dst, idx_dst;
    
    for (ix = 0; ix < nx; ix++) {
        for (iy = 0; iy < ny; iy++) {
            for (iz = 0; iz < nz; iz++) {
                icell_src = iz + iy*nz + ix*ny*nz;
                for (ip = cell_bound_min[icell_src]; ip < cell_bound_max[icell_src]; ip++) {
                    if (is_dead[ip]) continue;
                    if (particle_cell_indices[ip] == icell_src) continue;
                    
                    ip_src = ip;
                    icell_dst = particle_cell_indices[ip_src];
                    idx_dst = sorted_idx[ip_src];

                    while (icell_dst != icell_src) {
                        for (ip_dst = cell_bound_min[icell_dst]; ip_dst < cell_bound_max[icell_dst]; ip_dst++) {
                            if (particle_cell_indices[ip_dst] != icell_dst || is_dead[ip_dst]) {
                                // Swap values
                                npy_intp tmp = particle_cell_indices[ip_dst];
                                particle_cell_indices[ip_dst] = icell_dst;
                                icell_dst = tmp;

                                tmp = sorted_idx[ip_dst];
                                sorted_idx[ip_dst] = idx_dst;
                                idx_dst = tmp;

                                ip_src = ip_dst;
                                ops++;
                                break;
                            }
                        }
                        if (is_dead[ip_dst]) break;
                    }
                    particle_cell_indices[ip] = icell_dst;
                    sorted_idx[ip] = idx_dst;
                    if (is_dead[ip_dst]) {
                        is_dead[ip] = 1;
                        is_dead[ip_dst] = 0;
                    }
                }
            }
        }
    }
    return ops;
}

static void sorted_cell_bound(
    npy_intp* grid_cell_count, npy_intp* cell_bound_min, npy_intp* cell_bound_max,
    npy_intp nx, npy_intp ny, npy_intp nz
) {
    npy_intp icell, icell_prev;
    npy_intp ncells = nx * ny * nz;
    
    cell_bound_min[0] = 0;
    for (icell = 1; icell < ncells; icell++) {
        icell_prev = icell - 1;
        cell_bound_min[icell] = cell_bound_min[icell_prev] + grid_cell_count[icell_prev];
        cell_bound_max[icell_prev] = cell_bound_min[icell];
    }
    cell_bound_max[ncells-1] = cell_bound_min[ncells-1] + grid_cell_count[ncells-1];
}

// Python wrappers
static PyObject* _calculate_cell_index(PyObject* self, PyObject* args) {
    PyObject *x, *y, *z, *is_dead, *particle_cell_indices, *grid_cell_count;
    npy_intp nx, ny, nz, npart;
    double dx, dy, dz, x0, y0, z0;

    if (!PyArg_ParseTuple(args, "OOOOnnnndddddOO", 
        &x, &y, &z, &is_dead, 
        &npart, &nx, &ny, &nz, 
        &dx, &dy, &dz, &x0, &y0, &z0,
        &particle_cell_indices, &grid_cell_count)) {
        return NULL;
    }

    calculate_cell_index(
        (double*)PyArray_DATA(x), (double*)PyArray_DATA(y), (double*)PyArray_DATA(z),
        (npy_bool*)PyArray_DATA(is_dead),
        npart, nx, ny, nz, dx, dy, dz, x0, y0, z0,
        (npy_intp*)PyArray_DATA(particle_cell_indices),
        (npy_intp*)PyArray_DATA(grid_cell_count)
    );
    Py_RETURN_NONE;
}

static PyObject* _cycle_sort(PyObject* self, PyObject* args) {
    PyArrayObject *cell_bound_min, *cell_bound_max, *particle_cell_indices, *is_dead, *sorted_idx;
    npy_intp nx, ny, nz;

    if (!PyArg_ParseTuple(args, "OOnnnOOO", 
        &cell_bound_min, &cell_bound_max,
        &nx, &ny, &nz,
        &particle_cell_indices, &is_dead, &sorted_idx)) {
        return NULL;
    }

    long ops = cycle_sort(
        (npy_intp*)PyArray_DATA(cell_bound_min),
        (npy_intp*)PyArray_DATA(cell_bound_max),
        nx, ny, nz,
        (npy_intp*)PyArray_DATA(particle_cell_indices),
        (npy_bool*)PyArray_DATA(is_dead),
        (npy_intp*)PyArray_DATA(sorted_idx)
    );
    return PyLong_FromLong(ops);
}

static PyObject* _sorted_cell_bound(PyObject* self, PyObject* args) {
    PyArrayObject *grid_cell_count, *cell_bound_min, *cell_bound_max;
    npy_intp nx, ny, nz;

    if (!PyArg_ParseTuple(args, "OOOnnn", 
        &grid_cell_count, &cell_bound_min, &cell_bound_max,
        &nx, &ny, &nz)) {
        return NULL;
    }

    sorted_cell_bound(
        (npy_intp*)PyArray_DATA(grid_cell_count),
        (npy_intp*)PyArray_DATA(cell_bound_min),
        (npy_intp*)PyArray_DATA(cell_bound_max),
        nx, ny, nz
    );
    Py_RETURN_NONE;
}

static PyObject* sort_particles_patches_3d(PyObject* self, PyObject* args) {
    PyObject *grid_cell_count_list, *cell_bound_min_list, *cell_bound_max_list;
    PyObject *x0s, *y0s, *z0s, *particle_cell_indices_list, *sorted_indices_list;
    PyObject *x_list, *y_list, *z_list, *is_dead_list, *attrs_list;
    npy_intp nx, ny, nz, npatches;
    double dx, dy, dz;

    if (!PyArg_ParseTuple(args, "OOOOOOnnndddnOOOOOOO", 
        &grid_cell_count_list, &cell_bound_min_list, &cell_bound_max_list,
        &x0s, &y0s, &z0s,
        &nx, &ny, &nz, &dx, &dy, &dz,
        &npatches,
        &particle_cell_indices_list, &sorted_indices_list,
        &x_list, &y_list, &z_list, &is_dead_list, &attrs_list)) {
        return NULL;
    }

    if (npatches <= 0) Py_RETURN_NONE;

    npy_intp nattrs = PyList_Size(attrs_list) / npatches;

    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        npy_intp* grid_cell_count = (npy_intp*)GetPatchArrayData(grid_cell_count_list, ipatch);
        npy_intp* cell_bound_min = (npy_intp*)GetPatchArrayData(cell_bound_min_list, ipatch);
        npy_intp* cell_bound_max = (npy_intp*)GetPatchArrayData(cell_bound_max_list, ipatch);
        
        double x0 = PyFloat_AsDouble(PyList_GetItem(x0s, ipatch));
        double y0 = PyFloat_AsDouble(PyList_GetItem(y0s, ipatch));
        double z0 = PyFloat_AsDouble(PyList_GetItem(z0s, ipatch));
        
        npy_intp* particle_cell_indices = (npy_intp*)GetPatchArrayData(particle_cell_indices_list, ipatch);
        npy_intp* sorted_indices = (npy_intp*)GetPatchArrayData(sorted_indices_list, ipatch);
        double* x = (double*)GetPatchArrayData(x_list, ipatch);
        double* y = (double*)GetPatchArrayData(y_list, ipatch);
        double* z = (double*)GetPatchArrayData(z_list, ipatch);
        npy_bool* is_dead = (npy_bool*)GetPatchArrayData(is_dead_list, ipatch);
        npy_intp npart = PyArray_DIM(PyList_GetItem(x_list, ipatch), 0);

        // Reset counts
        for (npy_intp icell = 0; icell < nx * ny * nz; icell++) {
            grid_cell_count[icell] = 0;
        }

        calculate_cell_index(
            x, y, z, is_dead, npart, nx, ny, nz, dx, dy, dz, x0, y0, z0,
            particle_cell_indices, grid_cell_count
        );

        sorted_cell_bound(grid_cell_count, cell_bound_min, cell_bound_max, nx, ny, nz);

        for (npy_intp ip = 0; ip < npart; ip++) {
            sorted_indices[ip] = ip;
        }
        
        cycle_sort(cell_bound_min, cell_bound_max, nx, ny, nz, 
                  particle_cell_indices, is_dead, sorted_indices);

        // Sort attributes
        double** attrs = (double**)malloc(nattrs * sizeof(double*));
        for (npy_intp iattr = 0; iattr < nattrs; iattr++) {
            attrs[iattr] = (double*)GetPatchArrayData(attrs_list, ipatch*nattrs + iattr);
        }
        double* buf = (double*)malloc(npart * sizeof(double));
        for (npy_intp iattr = 0; iattr < nattrs; iattr++) {
            double* attr = attrs[iattr];
            for (npy_intp ip = 0; ip < npart; ip++) {
                if (ip != sorted_indices[ip]) {
                    buf[ip] = attr[sorted_indices[ip]];
                }
            }
            for (npy_intp ip = 0; ip < npart; ip++) {
                if (ip != sorted_indices[ip]) {
                    attr[ip] = buf[ip];
                }
            }
        }
        free(buf);
        free(attrs);
    }

    Py_RETURN_NONE;
}

static PyMethodDef SortMethods[] = {
    {"sort_particles_patches_3d", sort_particles_patches_3d, METH_VARARGS, "Sort 3D particles"},
    {"_calculate_cell_index", _calculate_cell_index, METH_VARARGS, "Calculate 3D cell indices"},
    {"_sorted_cell_bound", _sorted_cell_bound, METH_VARARGS, "Calculate 3D cell bounds"},
    {"_cycle_sort", _cycle_sort, METH_VARARGS, "3D cycle sort"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef sortmodule = {
    PyModuleDef_HEAD_INIT,
    "cpu3d",
    NULL,
    -1,
    SortMethods
};

PyMODINIT_FUNC PyInit_cpu3d(void) {
    import_array();
    return PyModule_Create(&sortmodule);
}
