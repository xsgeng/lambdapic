#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>

#define GetPatchArrayData(list, ipatch) PyArray_DATA((PyArrayObject*)PyList_GetItem(list, ipatch))


static void calculate_bucket_index(
    double* x, double* y, npy_bool* is_dead, 
    npy_intp npart, npy_intp nx, npy_intp ny, double dx, double dy, double x0, double y0, 
    npy_intp* particle_index, npy_intp* bucket_count
) {
    npy_intp ix, iy, ip, icell;
    memset(bucket_count, 0, sizeof(npy_intp) * nx * ny);

    icell = 0;
    for (ip = 0; ip < npart; ip++) {
        if (!is_dead[ip]) {
            ix = (npy_intp) floor((x[ip] - x0) / dx);
            iy = (npy_intp) floor((y[ip] - y0) / dy);
            icell = iy + ix * ny;  // Calculate the cell index
            if (0 <= ix && ix < nx && 0 <= iy && iy < ny) {
                particle_index[ip] = icell;  // Store the cell index for the particle
                bucket_count[icell] += 1;  // Increment the count of particles in the cell
            } else {
                particle_index[ip] = nx * ny - 1; // out-of-bound particles to the last bucket
                bucket_count[nx * ny - 1] += 1;
            }
        } else {
            // dead stay in the same bucket with the previous particle
            particle_index[ip] = icell;
            bucket_count[icell] += 1;
        }
    }
}

static PyObject* _calculate_bucket_index(PyObject* self, PyObject* args) {
    PyArrayObject *x, *y, *is_dead, *particle_index, *bucket_count;
    npy_intp nx, ny, npart;
    double dx, dy, x0, y0;
    if (!PyArg_ParseTuple(args, "OOOnnnddddOO", 
        &x, &y, &is_dead, 
        &npart, &nx, &ny, &dx, &dy, &x0, &y0, 
        &particle_index, &bucket_count)) {
        return NULL;  
    }

    calculate_bucket_index(
        (double*) PyArray_DATA(x), (double*) PyArray_DATA(y), (npy_bool*) PyArray_DATA(is_dead), 
        npart, nx, ny, dx, dy, x0, y0, 
        (npy_intp*) PyArray_DATA(particle_index), (npy_intp*) PyArray_DATA(bucket_count)
    );
    Py_RETURN_NONE;  
}

static void calculate_bucket_bound(
    npy_intp* bucket_count, npy_intp* bucket_bound_min, npy_intp* bucket_bound_max, 
    npy_intp nx, npy_intp ny
) {
    npy_intp icell, icell_prev;
    bucket_bound_min[0] = 0;  // Initialize the minimum bound for the first cell

    for (icell = 1; icell < nx * ny; icell++) {
        icell_prev = icell - 1;  // Get the previous cell index
        bucket_bound_min[icell] = bucket_bound_min[icell_prev] + bucket_count[icell_prev];  // Calculate the minimum bound for the current cell
        bucket_bound_max[icell_prev] = bucket_bound_min[icell];  // Calculate the maximum bound for the previous cell
    }
    bucket_bound_max[nx * ny - 1] = bucket_bound_min[nx * ny - 1] + bucket_count[nx * ny - 1];  // Calculate the maximum bound for the last cell
}

static PyObject* _calculate_bucket_bound(PyObject* self, PyObject* args) {
    PyArrayObject *bucket_count, *bucket_bound_min, *bucket_bound_max;
    npy_intp nx, ny;
    if (!PyArg_ParseTuple(args, "OOOnn", 
        &bucket_count, &bucket_bound_min, &bucket_bound_max, 
        &nx, &ny)) {
        return NULL;  
    }
    calculate_bucket_bound(
        (npy_intp*) PyArray_DATA(bucket_count), (npy_intp*) PyArray_DATA(bucket_bound_min), (npy_intp*) PyArray_DATA(bucket_bound_max), 
        nx, ny
    );
    Py_RETURN_NONE;  
}

static npy_intp bucket_sort(
    npy_intp* bucket_count, npy_intp* bucket_count_not, npy_intp* bucket_start_counter, npy_intp nx, npy_intp ny, // length of nbin
    npy_intp* particle_index, npy_intp* particle_index_ref, npy_intp npart, // length of npart
    npy_intp* particle_index_target, double * buf, // length of nbuf
    npy_bool* is_dead, double** attrs, npy_intp nattrs
) {
    npy_intp ip, ibuf, ibin, ix, iy, iattr, nbuf;
    npy_intp nbin = nx * ny;
    
    ibuf = 0;
    for (ix = 0; ix < nx; ix++) {
        for (iy = 0; iy < ny; iy++) {
            ibin = iy + ix * ny;
            for (ip = ibuf; ip < ibuf+bucket_count[ibin]; ip++) {
                particle_index_ref[ip] = ibin; // particle_index_ref is what particle_index should be after sorting
            }
            ibuf += bucket_count[ibin];
        }
    }

    // count number of particles that need to be moved
    nbuf = 0;
    memset(bucket_count_not, 0, sizeof(npy_intp) * nbin);
    for (ip = 0; ip < npart; ip++) {
        if (particle_index[ip] != particle_index_ref[ip]) {
            bucket_count_not[particle_index_ref[ip]] += 1;
            particle_index_target[nbuf] = ip;
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
        bucket_start_counter[0] = 0;
        for (ibin = 1; ibin < nbin; ibin++) {
            bucket_start_counter[ibin] = bucket_start_counter[ibin-1] + bucket_count_not[ibin-1];
        }

        for (ip = 0; ip < npart; ip++) {
            if (particle_index[ip] != particle_index_ref[ip]) {
                buf[bucket_start_counter[particle_index[ip]]] = attrs[iattr][ip];
                bucket_start_counter[particle_index[ip]] += 1;
            }
        }
        
        // fill back
        for (ibuf = 0; ibuf < nbuf; ibuf++) {
            attrs[iattr][particle_index_target[ibuf]] = buf[ibuf];
        }
    }

    // dead particles, same above
    npy_bool* is_dead_buf = (npy_bool*) buf;
    memset(is_dead_buf, 0, sizeof(npy_bool) * nbuf);
    // re-calculate counter for each attr
    bucket_start_counter[0] = 0;
    for (ibin = 1; ibin < nbin; ibin++) {
        bucket_start_counter[ibin] = bucket_start_counter[ibin-1] + bucket_count_not[ibin-1];
    }

    for (ip = 0; ip < npart; ip++) {
        if (particle_index[ip] != particle_index_ref[ip]) {
            is_dead_buf[bucket_start_counter[particle_index[ip]]] = is_dead[ip];
            is_dead[ip] = 1;
            bucket_start_counter[particle_index[ip]] += 1;
        }
    }
    
    // fill back
    for (ibuf = 0; ibuf < nbuf; ibuf++) {
        is_dead[particle_index_target[ibuf]] = is_dead_buf[ibuf];
    }
    
    return nbuf;
}

static PyObject* _bucket_sort(PyObject* self, PyObject* args) {
    PyArrayObject *bucket_count, *bucket_count_not, *bucket_start_counter;
    PyArrayObject *particle_index, *particle_index_ref;
    PyArrayObject *particle_index_target, *buf;
    PyArrayObject *is_dead;
    PyObject *attrs;
    npy_intp nx, ny, npart, nattrs;
    if (!PyArg_ParseTuple(args, "OOOnnOOnOOOOn", 
        &bucket_count, &bucket_count_not, &bucket_start_counter, &nx, &ny,
        &particle_index, &particle_index_ref, &npart,
        &particle_index_target, &buf, 
        &is_dead, &attrs, &nattrs)) {
        return NULL;  
    }

    double** attrs_ = (double**) malloc(nattrs * sizeof(double*));
    for (npy_intp iattr = 0; iattr < nattrs; iattr++) {
        attrs_[iattr] = (double*) GetPatchArrayData(attrs, iattr);  
    }
    npy_intp nbuf = bucket_sort(
        (npy_intp*) PyArray_DATA(bucket_count), (npy_intp*) PyArray_DATA(bucket_count_not), (npy_intp*) PyArray_DATA(bucket_start_counter), nx, ny, 
        (npy_intp*) PyArray_DATA(particle_index), (npy_intp*) PyArray_DATA(particle_index_ref), npart, 
        (npy_intp*) PyArray_DATA(particle_index_target), (double*) PyArray_DATA(buf), 
        (npy_bool*) PyArray_DATA(is_dead), attrs_, nattrs
    );
    free(attrs_);
    return PyLong_FromLong(nbuf);
}

static PyObject* sort_particles_patches_2d(PyObject* self, PyObject* args) {
    PyObject *bucket_count_list, *bucket_bound_min_list, *bucket_bound_max_list, 
             *bucket_count_not_list, *bin_start_counter_list; // length of nbucket
    PyObject *particle_index_list, *particle_index_ref_list; // length of npart
    PyObject *particle_index_target_list, *buf_list; // length of nbuf
    PyObject *x0s, *y0s;
    PyObject *x_list, *y_list, *is_dead_list, *attrs_list;

    npy_intp nx, ny, npatches;
    double dx, dy;

    if (!PyArg_ParseTuple(args, "OOOOOOnnddnOOOOOOOOO", 
        &x_list, &y_list, &is_dead_list, &attrs_list,
        &x0s, &y0s, 
        &nx, &ny, &dx, &dy, 
        &npatches, 
        // buffers
        &bucket_count_list, &bucket_bound_min_list, &bucket_bound_max_list, &bucket_count_not_list, &bin_start_counter_list, 
        &particle_index_list, &particle_index_ref_list,
        &particle_index_target_list, &buf_list)) {
        return NULL;  // Return NULL if argument parsing fails
    }

    if (npatches <= 0) {
        Py_RETURN_NONE;  // Return None if there are no patches
    }

    npy_intp nattrs = PyList_Size(attrs_list) / npatches;  

    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        npy_int64* bucket_count = (npy_int64*) GetPatchArrayData(bucket_count_list, ipatch);
        npy_int64* bucket_bound_min = (npy_int64*) GetPatchArrayData(bucket_bound_min_list, ipatch);
        npy_int64* bucket_bound_max = (npy_int64*) GetPatchArrayData(bucket_bound_max_list, ipatch);  
        npy_int64* bucket_count_not = (npy_int64*) GetPatchArrayData(bucket_count_not_list, ipatch);  
        npy_int64* bucket_start_counter = (npy_int64*) GetPatchArrayData(bin_start_counter_list, ipatch);  

        npy_int64* particle_index = (npy_int64*) GetPatchArrayData(particle_index_list, ipatch);
        npy_int64* particle_index_ref = (npy_int64*) GetPatchArrayData(particle_index_ref_list, ipatch);
        
        npy_int64* particle_index_target = (npy_int64*) GetPatchArrayData(particle_index_target_list, ipatch);
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
            particle_index, bucket_count
        );  // Calculate cell indices for the current patch

        calculate_bucket_bound(bucket_count, bucket_bound_min, bucket_bound_max, nx, ny);

        // attributes in ipatch
        double** attrs = (double**) malloc(nattrs * sizeof(double*));
        for (npy_intp iattr = 0; iattr < nattrs; iattr++) {
            attrs[iattr] = (double*) GetPatchArrayData(attrs_list, ipatch * nattrs + iattr);  
        }

        bucket_sort(
            bucket_count, bucket_count_not, bucket_start_counter, nx, ny, 
            particle_index, particle_index_ref, npart, 
            particle_index_target, buf, 
            is_dead, attrs, nattrs
        );

        free(attrs);
    }

    Py_RETURN_NONE;
}

static PyMethodDef SortMethods[] = {
    {"sort_particles_patches_2d", sort_particles_patches_2d, METH_VARARGS, "Sort particles patches"},
    {"_calculate_bucket_index", _calculate_bucket_index, METH_VARARGS, "Calculate cell index"},
    {"_calculate_bucket_bound", _calculate_bucket_bound, METH_VARARGS, "Calculate sorted cell bound"},
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
