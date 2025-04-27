#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>

#define GetPatchArrayData(list, ipatch) PyArray_DATA((PyArrayObject*)PyList_GetItem(list, ipatch))
/**
 * Calculate the cell index for each particle based on its position.
 * 
 * @param x Pointer to the array of particle x-coordinates.
 * @param y Pointer to the array of particle y-coordinates.
 * @param is_dead Pointer to the array indicating if a particle is dead.
 * @param npart Number of particles.
 * @param nx Number of cells in the x-direction.
 * @param ny Number of cells in the y-direction.
 * @param dx Cell size in the x-direction.
 * @param dy Cell size in the y-direction.
 * @param x0 Origin x-coordinate.
 * @param y0 Origin y-coordinate.
 * @param particle_cell_indices Pointer to the array to store the cell indices.
 * @param grid_cell_count Pointer to the array to store the count of particles in each cell.
 */
static void calculate_bucket_index(
    double* x, double* y, npy_bool* is_dead, 
    npy_intp npart, npy_intp nx, npy_intp ny, 
    npy_intp nx_bucket, npy_intp ny_bucket,
    double dx, double dy, double x0, double y0,
    npy_intp* particle_bucket_indices, npy_intp* bucket_count
) {
    npy_intp ix, iy, ip, ix_bucket, iy_bucket, ibucket;
    npy_intp nx_buckets = (nx + nx_bucket - 1) / nx_bucket;
    npy_intp ny_buckets = (ny + ny_bucket - 1) / ny_bucket;

    for (ip = 0; ip < npart; ip++) {
        if (!is_dead[ip]) {
            ix = (npy_intp)floor((x[ip] - x0) / dx);
            iy = (npy_intp)floor((y[ip] - y0) / dy);
            
            if (0 <= ix && ix < nx && 0 <= iy && iy < ny) {
                ix_bucket = ix / nx_bucket;
                iy_bucket = iy / ny_bucket;
                ibucket = iy_bucket + ix_bucket * ny_buckets;
                
                if (ibucket < nx_buckets * ny_buckets) {
                    particle_bucket_indices[ip] = ibucket;
                    bucket_count[ibucket]++;
                    continue;
                }
            }
            particle_bucket_indices[ip] = -1;
        } else {
            particle_bucket_indices[ip] = -1;
        }
    }
}

static PyObject* _calculate_bucket_index(PyObject* self, PyObject* args) {
    PyObject *x, *y, *is_dead, *particle_bucket_indices, *bucket_count;
    npy_intp nx, ny, npart, nx_bucket, ny_bucket;
    double dx, dy, x0, y0;

    if (!PyArg_ParseTuple(args, "OOOnnnnnddddOO", 
        &x, &y, &is_dead, 
        &npart, &nx, &ny, &nx_bucket, &ny_bucket,
        &dx, &dy, &x0, &y0, 
        &particle_bucket_indices, &bucket_count)) {
        return NULL;  
    }

    calculate_bucket_index(
        (double*) PyArray_DATA(x), (double*) PyArray_DATA(y), 
        (npy_bool*) PyArray_DATA(is_dead), npart, nx, ny, 
        nx_bucket, ny_bucket,
        dx, dy, x0, y0,
        (npy_intp*) PyArray_DATA(particle_bucket_indices),
        (npy_intp*) PyArray_DATA(bucket_count)
    );
    Py_RETURN_NONE;
}

static void calculate_cell_index(
    double* x, double* y, npy_bool* is_dead, 
    npy_intp npart, npy_intp nx, npy_intp ny, double dx, double dy, double x0, double y0, 
    npy_intp* particle_cell_indices, npy_intp* grid_cell_count
) {
    npy_intp ix, iy, ip, icell;
    icell = 0;
    for (ip = 0; ip < npart; ip++) {
        if (!is_dead[ip]) {
            ix = (npy_intp) floor((x[ip] - x0) / dx);  // Calculate the x-index of the cell
            iy = (npy_intp) floor((y[ip] - y0) / dy);  // Calculate the y-index of the cell
            icell = iy + ix * ny;  // Calculate the cell index
            if (0 <= ix && ix < nx && 0 <= iy && iy < ny) {
                particle_cell_indices[ip] = icell;  // Store the cell index for the particle
                grid_cell_count[icell] += 1;  // Increment the count of particles in the cell
            }
        } else {
            particle_cell_indices[ip] = -1;  // Mark dead particles with a cell index of -1
        }
    }
}

/**
 * Python wrapper for the calculate_cell_index function.
 * 
 * @param self Python module object.
 * @param args Python arguments tuple.
 * @return None.
 */
static PyObject* _calculate_cell_index(PyObject* self, PyObject* args) {
    PyObject *x, *y, *is_dead, *particle_cell_indices, *grid_cell_count;
    npy_intp nx, ny, npart;
    double dx, dy, x0, y0;
    if (!PyArg_ParseTuple(args, "OOOnnnddddOO", 
        &x, &y, &is_dead, 
        &npart, &nx, &ny, &dx, &dy, &x0, &y0, 
        &particle_cell_indices, &grid_cell_count)) {
        return NULL;  
    }

    calculate_cell_index(
        (double*) PyArray_DATA(x), (double*) PyArray_DATA(y), (npy_bool*) PyArray_DATA(is_dead), 
        npart, nx, ny, dx, dy, x0, y0, 
        (npy_intp*) PyArray_DATA(particle_cell_indices), (npy_intp*) PyArray_DATA(grid_cell_count)
    );
    Py_RETURN_NONE;  
}

/**
 * Perform a cycle sort to sort particles within cells.
 * 
 * @param cell_bound_min Pointer to the array of minimum bounds for each cell.
 * @param cell_bound_max Pointer to the array of maximum bounds for each cell.
 * @param nx Number of cells in the x-direction.
 * @param ny Number of cells in the y-direction.
 * @param particle_cell_indices Pointer to the array of particle cell indices.
 * @param is_dead Pointer to the array indicating if a particle is dead.
 * @param sorted_idx Pointer to the array to store the sorted indices.
 * @return Number of operations performed.
 */
static npy_intp cycle_sort(
    npy_intp* cell_bound_min, npy_intp* cell_bound_max, 
    npy_intp nx, npy_intp ny, 
    npy_intp* particle_cell_indices, npy_bool* is_dead, npy_intp* sorted_idx
) {
    npy_intp ops = 0;
    npy_intp ix, iy, ip, ip_src, ip_dst, icell_src, icell_dst, idx_dst;
    
    for (ix = 0; ix < nx; ix++) {
        for (iy = 0; iy < ny; iy++) {
            icell_src = iy + ix * ny;  // Calculate the source cell index
            for (ip = cell_bound_min[icell_src]; ip < cell_bound_max[icell_src]; ip++) {
                if (is_dead[ip]) {
                    continue;  // Skip dead particles
                }
                if (particle_cell_indices[ip] == icell_src) {
                    continue;  // Skip particles already in the correct cell
                }
                ip_src = ip;
                icell_dst = particle_cell_indices[ip_src];  // Get the destination cell index
                idx_dst = sorted_idx[ip_src];  // Get the destination index

                while (icell_dst != icell_src) {
                    for (ip_dst = cell_bound_min[icell_dst]; ip_dst < cell_bound_max[icell_dst]; ip_dst++) {
                        if (particle_cell_indices[ip_dst] != icell_dst || is_dead[ip_dst]) {
                            // swap
                            npy_intp tmp = particle_cell_indices[ip_dst];
                            particle_cell_indices[ip_dst] = icell_dst;
                            icell_dst = tmp;

                            tmp = sorted_idx[ip_dst];
                            sorted_idx[ip_dst] = idx_dst;
                            idx_dst = tmp;

                            ip_src = ip_dst;  // Update source index
                            ops += 1;  // Increment operation count
                            break;
                        }
                    }
                    if (is_dead[ip_dst]) {
                        break;  // Break if the destination particle is dead
                    }
                }
                particle_cell_indices[ip] = icell_dst;  // Update particle cell index
                sorted_idx[ip] = idx_dst;  // Update sorted index
                if (is_dead[ip_dst]) {
                    is_dead[ip] = 1;  // Mark source particle as dead
                    is_dead[ip_dst] = 0;  // Mark destination particle as alive
                }
            }
        }
    }
    return ops;  // Return the number of operations performed
}

/**
 * Python wrapper for the cycle_sort function.
 * 
 * @param self Python module object.
 * @param args Python arguments tuple.
 * @return Number of operations performed.
 */
static PyObject* _cycle_sort(PyObject* self, PyObject* args) {
    PyArrayObject *cell_bound_min, *cell_bound_max, *particle_cell_indices, *is_dead, *sorted_idx;
    npy_intp nx, ny;
    if (!PyArg_ParseTuple(args, "OOnnOOO", 
        &cell_bound_min, &cell_bound_max, 
        &nx, &ny, 
        &particle_cell_indices, &is_dead, &sorted_idx)) {
        return NULL;  
    }
    npy_intp ops = cycle_sort(
        (npy_intp*) PyArray_DATA(cell_bound_min), (npy_intp*) PyArray_DATA(cell_bound_max), 
        nx, ny, 
        (npy_intp*) PyArray_DATA(particle_cell_indices), (npy_bool*) PyArray_DATA(is_dead), (npy_intp*) PyArray_DATA(sorted_idx)
    );
    return PyLong_FromLong(ops);  
}

/**
 * Calculate the bounds for each cell based on the particle counts.
 * 
 * @param grid_cell_count Pointer to the array of particle counts in each cell.
 * @param cell_bound_min Pointer to the array to store the minimum bounds.
 * @param cell_bound_max Pointer to the array to store the maximum bounds.
 * @param nx Number of cells in the x-direction.
 * @param ny Number of cells in the y-direction.
 */
static void sorted_cell_bound(
    npy_intp* grid_cell_count, npy_intp* cell_bound_min, npy_intp* cell_bound_max, 
    npy_intp nx, npy_intp ny
) {
    npy_intp icell, icell_prev;
    cell_bound_min[0] = 0;  // Initialize the minimum bound for the first cell

    for (icell = 1; icell < nx * ny; icell++) {
        icell_prev = icell - 1;  // Get the previous cell index
        cell_bound_min[icell] = cell_bound_min[icell_prev] + grid_cell_count[icell_prev];  // Calculate the minimum bound for the current cell
        cell_bound_max[icell_prev] = cell_bound_min[icell];  // Calculate the maximum bound for the previous cell
    }
    cell_bound_max[nx * ny - 1] = cell_bound_min[nx * ny - 1] + grid_cell_count[nx * ny - 1];  // Calculate the maximum bound for the last cell
}

/**
 * Python wrapper for sorted_cell_bound
 * 
 * @param self Python module object.
 * @param args Python arguments tuple.
 * @return None.
 */
static PyObject* _sorted_cell_bound(PyObject* self, PyObject* args) {
    PyArrayObject *grid_cell_count, *cell_bound_min, *cell_bound_max;
    npy_intp nx, ny;
    if (!PyArg_ParseTuple(args, "OOOnn", 
        &grid_cell_count, &cell_bound_min, &cell_bound_max, 
        &nx, &ny)) {
        return NULL;  
    }
    sorted_cell_bound(
        (npy_intp*) PyArray_DATA(grid_cell_count), (npy_intp*) PyArray_DATA(cell_bound_min), (npy_intp*) PyArray_DATA(cell_bound_max), 
        nx, ny
    );
    Py_RETURN_NONE;  
}


/**
 * Sort particles in patches using OpenMP for parallel execution.
 * 
 * @param self Python module object.
 * @param args Python arguments tuple.
 * @return None.
 */
static PyObject* sort_particles_patches_2d(PyObject* self, PyObject* args) {
    PyObject *grid_cell_count_list, *cell_bound_min_list, *cell_bound_max_list, *x0s, *y0s, *particle_cell_indices_list, *sorted_indices_list, *x_list, *y_list, *is_dead_list, *attrs_list;
    npy_intp nx, ny, npatches;
    double dx, dy;

    if (!PyArg_ParseTuple(args, "OOOOOnnddnOOOOOO", 
        &grid_cell_count_list, &cell_bound_min_list, &cell_bound_max_list, 
        &x0s, &y0s, 
        &nx, &ny, &dx, &dy, 
        &npatches, 
        &particle_cell_indices_list, &sorted_indices_list, &x_list, &y_list, &is_dead_list, &attrs_list)) {
        return NULL;  // Return NULL if argument parsing fails
    }

    if (npatches <= 0) {
        Py_RETURN_NONE;  // Return None if there are no patches
    }

    npy_intp nattrs = PyList_Size(attrs_list) / npatches;  

    #pragma omp parallel for  // Parallelize the loop using OpenMP
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        npy_intp* grid_cell_count = (npy_intp*) GetPatchArrayData(grid_cell_count_list, ipatch);  
        npy_intp* cell_bound_min = (npy_intp*) GetPatchArrayData(cell_bound_min_list, ipatch);
        npy_intp* cell_bound_max = (npy_intp*) GetPatchArrayData(cell_bound_max_list, ipatch);
        double x0 = PyFloat_AsDouble(PyList_GetItem(x0s, ipatch));  
        double y0 = PyFloat_AsDouble(PyList_GetItem(y0s, ipatch));  
        npy_intp* particle_cell_indices = (npy_intp*) GetPatchArrayData(particle_cell_indices_list, ipatch);  
        npy_intp* sorted_indices = (npy_intp*) GetPatchArrayData(sorted_indices_list, ipatch);  
        double* x = (double*) GetPatchArrayData(x_list, ipatch);  
        double* y = (double*) GetPatchArrayData(y_list, ipatch);  
        npy_bool* is_dead = (npy_bool*) GetPatchArrayData(is_dead_list, ipatch);  

        npy_intp npart = PyArray_DIM(PyList_GetItem(x_list, ipatch), 0);  

        for (npy_intp icell = 0; icell < nx * ny; icell++) {
            grid_cell_count[icell] = 0;  // Initialize grid cell count for the current patch
        }

        calculate_cell_index(
            x, y, is_dead,
            npart,
            nx, ny, dx, dy,
            x0, y0,
            particle_cell_indices, grid_cell_count
        );  // Calculate cell indices for the current patch

        sorted_cell_bound(
            grid_cell_count, 
            cell_bound_min, 
            cell_bound_max, 
            nx, ny
        );  // Calculate cell bounds for the current patch

        for (npy_intp ip = 0; ip < npart; ip++) {
            sorted_indices[ip] = ip;  // Initialize sorted indices for the current patch
        }

        cycle_sort(cell_bound_min, cell_bound_max, nx, ny, particle_cell_indices, is_dead, sorted_indices);  // Perform cycle sort for the current patch
        
        // attributes in ipatch
        double** attrs = (double**) malloc(nattrs * sizeof(double*));
        for (npy_intp iattr = 0; iattr < nattrs; iattr++) {
            attrs[iattr] = (double*) GetPatchArrayData(attrs_list, ipatch * nattrs + iattr);  
        }
        double* buf = (double*) malloc(npart * sizeof(double));  // Allocate memory for buffer
        for (npy_intp iattr = 0; iattr < nattrs; iattr++) {
            double* attr = attrs[iattr];
            for (npy_intp ip = 0; ip < npart; ip++) {
                if (ip != sorted_indices[ip]) {
                    buf[ip] = attr[sorted_indices[ip]];  // Copy attributes to buffer
                }
            }
            for (npy_intp ip = 0; ip < npart; ip++) {
                if (ip != sorted_indices[ip]) {
                    attr[ip] = buf[ip];  // Copy attributes from buffer
                }
            }
        }
        free(buf);  // Free buffer memory
    }

    Py_RETURN_NONE;
}

static PyMethodDef SortMethods[] = {
    {"sort_particles_patches_2d", sort_particles_patches_2d, METH_VARARGS, "Sort particles patches"},
    {"_calculate_cell_index", _calculate_cell_index, METH_VARARGS, "Calculate cell index"},
    {"_calculate_bucket_index", _calculate_bucket_index, METH_VARARGS, "Calculate bucket index"},
    {"_sorted_cell_bound", _sorted_cell_bound, METH_VARARGS, "Calculate sorted cell bound"},
    {"_cycle_sort", _cycle_sort, METH_VARARGS, "Cycle sort"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef sortmodule = {
    PyModuleDef_HEAD_INIT,
    "cpu2d",
    NULL,
    -1,
    SortMethods
};

PyMODINIT_FUNC PyInit_cpu2d(void) {
    import_array();
    return PyModule_Create(&sortmodule);
}
