#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>

#define GetPatchArrayData(list, ipatch) PyArray_DATA((PyArrayObject*)PyList_GetItem(list, ipatch))


static void calculate_bucket_index(
    double* x, double* y, npy_bool* is_dead, 
    npy_intp npart, npy_intp nx, npy_intp ny, double dx, double dy, double x0, double y0, 
    npy_intp* particle_cell_indices, npy_intp* grid_cell_count
) {
    npy_intp ix, iy, ip, icell;
    memset(grid_cell_count, 0, sizeof(npy_intp) * nx * ny);

    icell = 0;
    for (ip = 0; ip < npart; ip++) {
        if (!is_dead[ip]) {
            ix = (npy_intp) floor((x[ip] - x0) / dx);
            iy = (npy_intp) floor((y[ip] - y0) / dy);
            icell = iy + ix * ny;  // Calculate the cell index
            if (0 <= ix && ix < nx && 0 <= iy && iy < ny) {
                particle_cell_indices[ip] = icell;  // Store the cell index for the particle
                grid_cell_count[icell] += 1;  // Increment the count of particles in the cell
            } else {
                particle_cell_indices[ip] = nx * ny - 1; // out-of-bound particles to the last bucket
                grid_cell_count[nx * ny - 1] += 1;
            }
        } else {
            // dead stay in the same bucket with the previous particle
            particle_cell_indices[ip] = icell;
            grid_cell_count[icell] += 1;
        }
    }
}

static PyObject* _calculate_bucket_index(PyObject* self, PyObject* args) {
    PyArrayObject *x, *y, *is_dead, *particle_cell_indices, *grid_cell_count;
    npy_intp nx, ny, npart;
    double dx, dy, x0, y0;
    if (!PyArg_ParseTuple(args, "OOOnnnddddOO", 
        &x, &y, &is_dead, 
        &npart, &nx, &ny, &dx, &dy, &x0, &y0, 
        &particle_cell_indices, &grid_cell_count)) {
        return NULL;  
    }

    calculate_bucket_index(
        (double*) PyArray_DATA(x), (double*) PyArray_DATA(y), (npy_bool*) PyArray_DATA(is_dead), 
        npart, nx, ny, dx, dy, x0, y0, 
        (npy_intp*) PyArray_DATA(particle_cell_indices), (npy_intp*) PyArray_DATA(grid_cell_count)
    );
    Py_RETURN_NONE;  
}

static npy_intp bucket_sort(
    npy_intp* bin_count, npy_intp* bin_count_not, npy_intp* bin_start_counter, npy_intp nx, npy_intp ny, // length of nbin
    npy_intp* bucket_index, npy_intp* bucket_index_ref, npy_intp npart, // length of npart
    npy_intp* bucket_index_target, double * buf, // length of nbuf
    npy_bool* is_dead, double** attrs, npy_intp nattrs
) {
    npy_intp ip, ibuf, ibin, ix, iy, iattr, nbuf;
    npy_intp nbin = nx * ny;
    
    ibuf = 0;
    for (ix = 0; ix < nx; ix++) {
        for (iy = 0; iy < ny; iy++) {
            ibin = iy + ix * ny;
            for (ip = ibuf; ip < ibuf+bin_count[ibin]; ip++) {
                bucket_index_ref[ip] = ibin; // bucket_index_ref is what bucket_index should be after sorting
            }
            ibuf += bin_count[ibin];
        }
    }

    // count number of particles that need to be moved
    nbuf = 0;
    memset(bin_count_not, 0, sizeof(npy_intp) * nbin);
    for (ip = 0; ip < npart; ip++) {
        if (bucket_index[ip] != bucket_index_ref[ip]) {
            bin_count_not[bucket_index_ref[ip]] += 1;
            bucket_index_target[nbuf] = ip;
            nbuf += 1;
        }
    }

    if (nbuf == 0) {
        return nbuf;
    }

    // move particles
    for (iattr = 0; iattr < nattrs; iattr++) {
        memset(buf, 0, sizeof(double) * nbuf);
        // re-calculate counter for each attr
        bin_start_counter[0] = 0;
        for (ibin = 1; ibin < nbin; ibin++) {
            bin_start_counter[ibin] = bin_start_counter[ibin-1] + bin_count_not[ibin-1];
        }

        for (ip = 0; ip < npart; ip++) {
            if (bucket_index[ip] != bucket_index_ref[ip]) {
                buf[bin_start_counter[bucket_index[ip]]] = attrs[iattr][ip];
                bin_start_counter[bucket_index[ip]] += 1;
            }
        }
        
        // fill back
        for (ibuf = 0; ibuf < nbuf; ibuf++) {
            attrs[iattr][bucket_index_target[ibuf]] = buf[ibuf];
        }
    }

    // dead particles, same above
    npy_bool* is_dead_buf = (npy_bool*) buf;
    memset(is_dead_buf, 0, sizeof(npy_bool) * nbuf);
    // re-calculate counter for each attr
    bin_start_counter[0] = 0;
    for (ibin = 1; ibin < nbin; ibin++) {
        bin_start_counter[ibin] = bin_start_counter[ibin-1] + bin_count_not[ibin-1];
    }

    for (ip = 0; ip < npart; ip++) {
        if (bucket_index[ip] != bucket_index_ref[ip]) {
            is_dead_buf[bin_start_counter[bucket_index[ip]]] = is_dead[ip];
            is_dead[ip] = 1;
            bin_start_counter[bucket_index[ip]] += 1;
        }
    }
    
    // fill back
    for (ibuf = 0; ibuf < nbuf; ibuf++) {
        is_dead[bucket_index_target[ibuf]] = is_dead_buf[ibuf];
    }
    
    return nbuf;
}

static PyObject* _bucket_sort(PyObject* self, PyObject* args) {
    PyArrayObject *bin_count, *bin_count_not, *bin_start_counter;
    PyArrayObject *bucket_index, *bucket_index_ref;
    PyArrayObject *bucket_index_target, *buf;
    PyArrayObject *is_dead;
    PyObject *attrs;
    npy_intp nx, ny, npart, nattrs;
    if (!PyArg_ParseTuple(args, "OOOnnOOnOOOOn", 
        &bin_count, &bin_count_not, &bin_start_counter, &nx, &ny,
        &bucket_index, &bucket_index_ref, &npart,
        &bucket_index_target, &buf, 
        &is_dead, &attrs, &nattrs)) {
        return NULL;  
    }

    double** attrs_ = (double**) malloc(nattrs * sizeof(double*));
    for (npy_intp iattr = 0; iattr < nattrs; iattr++) {
        attrs_[iattr] = (double*) GetPatchArrayData(attrs, iattr);  
    }
    npy_intp nbuf = bucket_sort(
        (npy_intp*) PyArray_DATA(bin_count), (npy_intp*) PyArray_DATA(bin_count_not), (npy_intp*) PyArray_DATA(bin_start_counter), nx, ny, 
        (npy_intp*) PyArray_DATA(bucket_index), (npy_intp*) PyArray_DATA(bucket_index_ref), npart, 
        (npy_intp*) PyArray_DATA(bucket_index_target), (double*) PyArray_DATA(buf), 
        (npy_bool*) PyArray_DATA(is_dead), attrs_, nattrs
    );
    free(attrs_);
    return PyLong_FromLong(nbuf);
}

static PyObject* sort_particles_patches_2d(PyObject* self, PyObject* args) {
    PyObject *bin_count_list, *bin_count_not_list, *bin_start_counter_list;
    PyObject *bucket_index_list, *bucket_index_ref_list;
    PyObject *bucket_index_target_list, *buf_list;
    PyObject *x0s, *y0s;
    PyObject *x_list, *y_list, *is_dead_list, *attrs_list;

    npy_intp nx, ny, npatches;
    double dx, dy;

    if (!PyArg_ParseTuple(args, "OOOOOOnnddnOOOOOOO", 
        &x_list, &y_list, &is_dead_list, &attrs_list,
        &x0s, &y0s, 
        &nx, &ny, &dx, &dy, 
        &npatches, 
        // buffers
        &bin_count_list, &bin_count_not_list, &bin_start_counter_list, 
        &bucket_index_list, &bucket_index_ref_list,
        &bucket_index_target_list, &buf_list)) {
        return NULL;  // Return NULL if argument parsing fails
    }

    if (npatches <= 0) {
        Py_RETURN_NONE;  // Return None if there are no patches
    }

    npy_intp nattrs = PyList_Size(attrs_list) / npatches;  

    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        npy_int64* bin_count = (npy_int64*) GetPatchArrayData(bin_count_list, ipatch);  
        npy_int64* bin_count_not = (npy_int64*) GetPatchArrayData(bin_count_not_list, ipatch);  
        npy_int64* bin_start_counter = (npy_int64*) GetPatchArrayData(bin_start_counter_list, ipatch);  

        npy_int64* bucket_index = (npy_int64*) GetPatchArrayData(bucket_index_list, ipatch);
        npy_int64* bucket_index_ref = (npy_int64*) GetPatchArrayData(bucket_index_ref_list, ipatch);
        
        npy_int64* bucket_index_target = (npy_int64*) GetPatchArrayData(bucket_index_target_list, ipatch);
        double* buf = (double*) GetPatchArrayData(buf_list, ipatch);  
        
        double x0 = PyFloat_AsDouble(PyList_GetItem(x0s, ipatch));  
        double y0 = PyFloat_AsDouble(PyList_GetItem(y0s, ipatch));  
        double* x = (double*) GetPatchArrayData(x_list, ipatch);  
        double* y = (double*) GetPatchArrayData(y_list, ipatch);  
        npy_bool* is_dead = (npy_bool*) GetPatchArrayData(is_dead_list, ipatch);  

        npy_intp npart = PyArray_DIM((PyArrayObject*)PyList_GetItem(x_list, ipatch), 0);  

        
        calculate_bucket_index(
            x, y, is_dead,
            npart,
            nx, ny, dx, dy,
            x0, y0,
            bucket_index, bin_count
        );  // Calculate cell indices for the current patch

        // attributes in ipatch
        double** attrs = (double**) malloc(nattrs * sizeof(double*));
        for (npy_intp iattr = 0; iattr < nattrs; iattr++) {
            attrs[iattr] = (double*) GetPatchArrayData(attrs_list, ipatch * nattrs + iattr);  
        }

        bucket_sort(
            bin_count, bin_count_not, bin_start_counter, nx, ny, 
            bucket_index, bucket_index_ref, npart, 
            bucket_index_target, buf, 
            is_dead, attrs, nattrs
        );

        free(attrs);
    }

    Py_RETURN_NONE;
}

static PyMethodDef SortMethods[] = {
    {"sort_particles_patches_2d", sort_particles_patches_2d, METH_VARARGS, "Sort particles patches"},
    {"_calculate_bucket_index", _calculate_bucket_index, METH_VARARGS, "Calculate cell index"},
    {"_bucket_sort", _bucket_sort, METH_VARARGS, "Bucket sort"},
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
