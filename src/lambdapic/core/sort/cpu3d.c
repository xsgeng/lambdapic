#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>

#define GetPatchArrayData(list, ipatch) PyArray_DATA((PyArrayObject*)PyList_GetItem(list, ipatch))

static void calculate_cell_index(
    double* x, double* y, double* z, npy_bool* is_dead,
    npy_intp npart, npy_intp nx, npy_intp ny, npy_intp nz,
    double dx, double dy, double dz, double x0, double y0, double z0,
    npy_int64* particle_index, npy_int64* bucket_count
) {
    npy_intp ix, iy, iz, ip, icell;
    memset(bucket_count, 0, sizeof(npy_int64) * nx * ny * nz);

    icell = 0;
    for (ip = 0; ip < npart; ip++) {
        if (!is_dead[ip]) {
            ix = (npy_intp)floor((x[ip] - x0)/dx);
            iy = (npy_intp)floor((y[ip] - y0)/dy);
            iz = (npy_intp)floor((z[ip] - z0)/dz);
            icell = iz + iy*nz + ix*ny*nz;
            if (0 <= ix && ix < nx && 0 <= iy && iy < ny && 0 <= iz && iz < nz) {
                particle_index[ip] = icell;
                bucket_count[icell] += 1;
            } else {
                particle_index[ip] = nx * ny * nz - 1; // out-of-bound particles to the last bucket
                bucket_count[nx * ny * nz - 1] += 1;
            }
        } else {
            // dead stay in the same bucket with the previous particle
            particle_index[ip] = icell;
            bucket_count[icell] += 1;
        }
    }
}

static void calculate_bucket_bound(
    npy_intp* bucket_count, npy_intp* bucket_bound_min, npy_intp* bucket_bound_max, 
    npy_intp nx, npy_intp ny, npy_intp nz
) {
    npy_intp icell, icell_prev;
    bucket_bound_min[0] = 0;  // Initialize the minimum bound for the first cell

    for (icell = 1; icell < nx * ny * nz; icell++) {
        icell_prev = icell - 1;  // Get the previous cell index
        bucket_bound_min[icell] = bucket_bound_min[icell_prev] + bucket_count[icell_prev];  // Calculate the minimum bound for the current cell
        bucket_bound_max[icell_prev] = bucket_bound_min[icell];  // Calculate the maximum bound for the previous cell
    }
    bucket_bound_max[nx * ny * nz - 1] = bucket_bound_min[nx * ny * nz - 1] + bucket_count[nx * ny * nz - 1];  // Calculate the maximum bound for the last cell
}

static npy_intp bucket_sort_3d(
    npy_int64* bucket_count, npy_int64* bucket_count_not, npy_int64* bucket_start_counter,
    npy_intp nx, npy_intp ny, npy_intp nz, // length of nbin
    npy_int64* particle_index, npy_int64* particle_index_ref, npy_intp npart, // length of npart
    npy_int64* particle_index_target, double * buf, // length of nbuf
    npy_bool* is_dead, double** attrs, npy_intp nattrs
) {
    npy_intp ip, ibuf, ibin, ix, iy, iz, iattr, nbuf;
    npy_intp nbin = nx * ny * nz;
    
    ibuf = 0;
    for (ix = 0; ix < nx; ix++) {
        for (iy = 0; iy < ny; iy++) {
            for (iz = 0; iz < nz; iz++) {
                ibin = iz + iy*nz + ix*ny*nz;
                for (ip = ibuf; ip < ibuf+bucket_count[ibin]; ip++) {
                    particle_index_ref[ip] = ibin; // particle_index_ref is what particle_index should be after sorting
                }
                ibuf += bucket_count[ibin];
            }
        }
    }

    // count number of particles that need to be moved
    nbuf = 0;
    memset(bucket_count_not, 0, sizeof(npy_int64) * nbin);
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

// Python wrappers
static PyObject* _calculate_cell_index(PyObject* self, PyObject* args) {
    PyArrayObject *x, *y, *z, *is_dead, *particle_index, *bucket_count;
    npy_intp nx, ny, nz, npart;
    double dx, dy, dz, x0, y0, z0;

    if (!PyArg_ParseTuple(args, "OOOOnnnndddddOO", 
        &x, &y, &z, &is_dead, 
        &npart, &nx, &ny, &nz, 
        &dx, &dy, &dz, &x0, &y0, &z0,
        &particle_index, &bucket_count)) {
        return NULL;
    }

    calculate_cell_index(
        (double*)PyArray_DATA(x), (double*)PyArray_DATA(y), (double*)PyArray_DATA(z),
        (npy_bool*)PyArray_DATA(is_dead),
        npart, nx, ny, nz, dx, dy, dz, x0, y0, z0,
        (npy_int64*)PyArray_DATA(particle_index),
        (npy_int64*)PyArray_DATA(bucket_count)
    );
    Py_RETURN_NONE;
}

static PyObject* _bucket_sort_3d(PyObject* self, PyObject* args) {
    PyArrayObject *bucket_count, *bucket_count_not, *bucket_start_counter;
    PyArrayObject *particle_index, *particle_index_ref;
    PyArrayObject *particle_index_target, *buf;
    PyArrayObject *is_dead;
    PyObject *attrs;
    npy_intp nx, ny, nz, npart, nattrs;
    if (!PyArg_ParseTuple(args, "OOOnnnOOnOOOOn", 
        &bucket_count, &bucket_count_not, &bucket_start_counter, &nx, &ny, &nz,
        &particle_index, &particle_index_ref, &npart,
        &particle_index_target, &buf, 
        &is_dead, &attrs, &nattrs)) {
        return NULL;  
    }

    double** attrs_ = (double**) malloc(nattrs * sizeof(double*));
    for (npy_intp iattr = 0; iattr < nattrs; iattr++) {
        attrs_[iattr] = (double*) GetPatchArrayData(attrs, iattr);  
    }
    npy_intp nbuf = bucket_sort_3d(
        (npy_int64*) PyArray_DATA(bucket_count), (npy_int64*) PyArray_DATA(bucket_count_not), (npy_int64*) PyArray_DATA(bucket_start_counter), nx, ny, nz,
        (npy_int64*) PyArray_DATA(particle_index), (npy_int64*) PyArray_DATA(particle_index_ref), npart,
        (npy_int64*) PyArray_DATA(particle_index_target), (double*) PyArray_DATA(buf),
        (npy_bool*) PyArray_DATA(is_dead), attrs_, nattrs
    );
    free(attrs_);
    return PyLong_FromLong(nbuf);
}

static PyObject* sort_particles_patches_3d(PyObject* self, PyObject* args) {
    PyObject *bucket_count_list, *bucket_bound_min_list, *bucket_bound_max_list,
             *bucket_count_not_list, *bucket_start_counter_list; // length of nbucket
    PyObject *particle_index_list, *particle_index_ref_list; // length of npart
    PyObject *particle_index_target_list, *buf_list; // length of nbuf
    PyObject *x0s, *y0s, *z0s;
    PyObject *x_list, *y_list, *z_list, *is_dead_list, *attrs_list;

    npy_intp nx, ny, nz, npatches;
    double dx, dy, dz;

    if (!PyArg_ParseTuple(args, "OOOOOOOOnnndddnOOOOOOOOO", 
        &x_list, &y_list, &z_list, &is_dead_list, &attrs_list,
        &x0s, &y0s, &z0s, 
        &nx, &ny, &nz, &dx, &dy, &dz, 
        &npatches, 
        // buffers
        &bucket_count_list, &bucket_bound_min_list, &bucket_bound_max_list, &bucket_count_not_list, &bucket_start_counter_list, 
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
        npy_int64* bucket_start_counter = (npy_int64*) GetPatchArrayData(bucket_start_counter_list, ipatch);

        npy_int64* particle_index = (npy_int64*) GetPatchArrayData(particle_index_list, ipatch);
        npy_int64* particle_index_ref = (npy_int64*) GetPatchArrayData(particle_index_ref_list, ipatch);
        
        npy_int64* particle_index_target = (npy_int64*) GetPatchArrayData(particle_index_target_list, ipatch);
        double* buf = (double*) GetPatchArrayData(buf_list, ipatch);  
        
        double x0 = PyFloat_AsDouble(PyList_GetItem(x0s, ipatch));  
        double y0 = PyFloat_AsDouble(PyList_GetItem(y0s, ipatch));  
        double z0 = PyFloat_AsDouble(PyList_GetItem(z0s, ipatch));  
        double* x = (double*) GetPatchArrayData(x_list, ipatch);  
        double* y = (double*) GetPatchArrayData(y_list, ipatch);  
        double* z = (double*) GetPatchArrayData(z_list, ipatch);  
        npy_bool* is_dead = (npy_bool*) GetPatchArrayData(is_dead_list, ipatch);  

        npy_intp npart = PyArray_DIM((PyArrayObject*)PyList_GetItem(x_list, ipatch), 0);  

        
        calculate_cell_index(
            x, y, z, is_dead,
            npart,
            nx, ny, nz, dx, dy, dz,
            x0, y0, z0,
            particle_index, bucket_count
        );  // Calculate cell indices for the current patch

        calculate_bucket_bound(bucket_count, bucket_bound_min, bucket_bound_max, nx, ny, nz);

        // attributes in ipatch
        double** attrs = (double**) malloc(nattrs * sizeof(double*));
        for (npy_intp iattr = 0; iattr < nattrs; iattr++) {
            attrs[iattr] = (double*) GetPatchArrayData(attrs_list, ipatch * nattrs + iattr);  
        }

        bucket_sort_3d(
            bucket_count, bucket_count_not, bucket_start_counter, nx, ny, nz, 
            particle_index, particle_index_ref, npart, 
            particle_index_target, buf, 
            is_dead, attrs, nattrs
        );

        free(attrs);
    }

    Py_RETURN_NONE;
}

static PyMethodDef SortMethods[] = {
    {"sort_particles_patches_3d", sort_particles_patches_3d, METH_VARARGS, "Sort 3D particles"},
    {"_calculate_cell_index", _calculate_cell_index, METH_VARARGS, "Calculate 3D cell indices"},
    {"_bucket_sort_3d", _bucket_sort_3d, METH_VARARGS, "3D bucket sort"},
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
