#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>
#include "../utils/cutils.h"

// #ifdef USE_TCMALLOC
// #include "tcmalloc.h"
// #define malloc tc_malloc
// #define free tc_free
// #endif

enum Boundary2D {
    XMIN = 0,
    XMAX,
    YMIN,
    YMAX,
    XMINYMIN,
    XMAXYMIN,
    XMINYMAX,
    XMAXYMAX,
    NUM_BOUNDARIES
};

static const enum Boundary2D OPPOSITE_BOUNDARY[NUM_BOUNDARIES] = {
    XMAX,
    XMIN,
    YMAX,
    YMIN,
    XMAXYMAX,
    XMINYMAX,
    XMAXYMIN,
    XMINYMIN
};

// Implementation of count_outgoing_particles function
static void count_outgoing_particles(
    double* x, double* y, npy_bool* is_dead,
    double xmin, double xmax, double ymin, double ymax,
    npy_intp npart,
    npy_intp* npart_out
) {
    for (npy_intp ip = 0; ip < npart; ip++) {
        if (is_dead[ip]) continue;
        if (y[ip] < ymin) {
            if (x[ip] < xmin) {
                (npart_out[XMINYMIN])++;
                continue;
            }
            else if (x[ip] > xmax) {
                (npart_out[XMAXYMIN])++;
                continue;
            }
            else {
                (npart_out[YMIN])++;
                continue;
            }
        }
        else if (y[ip] > ymax) {
            if (x[ip] < xmin) {
                (npart_out[XMINYMAX])++;
                continue;
            }
            else if (x[ip] > xmax) {
                (npart_out[XMAXYMAX])++;
                continue;
            }
            else {
                (npart_out[YMAX])++;
                continue;
            }
        }
        else {
            if (x[ip] < xmin) {
                (npart_out[XMIN])++;
                continue;
            }
            else if (x[ip] > xmax) {
                (npart_out[XMAX])++;
                continue;
            }
        }
    }
}

#define SET_INCOMING_INDEX(BOUND) \
    ipatch = boundary_index[BOUND]; \
    if (ipatch < 0) continue; \
    ibound = OPPOSITE_BOUNDARY[BOUND]; \
    incoming_indices_list[ipatch][ibound][ibuff[BOUND]] = ip; \
    (ibuff[BOUND])++; \
    continue;

// Get indices of incoming particles from neighboring patches
static void get_incoming_index(
double* x, double* y, npy_bool* is_dead, npy_intp npart,
double xmin, double xmax, 
double ymin, double ymax, 
npy_intp* boundary_index,
// out
npy_intp*** incoming_indices_list
) {
    AUTOFREE npy_intp* ibuff = (npy_intp*)malloc(NUM_BOUNDARIES * sizeof(npy_intp));
    for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
        ibuff[ibound] = 0;
    }
    npy_intp ipatch, ibound;
    for (npy_intp ip = 0; ip < npart; ip++) {
        if (is_dead[ip]) continue;
        if (y[ip] < ymin) {
            if (x[ip] < xmin) {
                SET_INCOMING_INDEX(XMINYMIN)
            }
            else if (x[ip] > xmax) {
                SET_INCOMING_INDEX(XMAXYMIN)
            }
            else {
                SET_INCOMING_INDEX(YMIN)
            }
        }
        else if (y[ip] > ymax) {
            if (x[ip] < xmin) {
                SET_INCOMING_INDEX(XMINYMAX)
            }
            else if (x[ip] > xmax) {
                SET_INCOMING_INDEX(XMAXYMAX)
            }
            else {
                SET_INCOMING_INDEX(YMAX)
            }
        }
        else {
            if (x[ip] < xmin) {
                SET_INCOMING_INDEX(XMIN)
            }
            else if (x[ip] > xmax) {
                SET_INCOMING_INDEX(XMAX)
            }
        }
    }
}

// Fill buffer with boundary particles
static void fill_boundary_particles_to_buffer(
    double** attrs_list, npy_intp nattrs,
    npy_intp** incoming_indices,
    npy_intp* npart_incoming,
    npy_intp* boundary_index,
    double* buffer
) {
    for (npy_intp iattr = 0; iattr < nattrs; iattr++) {
        npy_intp ibuff = 0;

        for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            if (boundary_index[ibound] >= 0) {
                double* attr = attrs_list[boundary_index[ibound]*nattrs + iattr];
                for (npy_intp i = 0; i < npart_incoming[ibound]; i++) {
                    npy_intp idx = incoming_indices[ibound][i];
                    buffer[ibuff*nattrs+iattr] = attr[idx];
                    ibuff++;
                }
            }
        }
    }
}



// Mark out-of-bound particles as dead
static void mark_out_of_bound_as_dead(
    double *x, double *y, npy_bool *is_dead, npy_intp npart, 
    double xmin, double xmax, 
    double ymin, double ymax
) {
    for (npy_intp ipart = 0; ipart < npart; ipart++) {
        if (is_dead[ipart]) {
            // x[ipart] = NAN;
            // y[ipart] = NAN;
            continue;
        }
        if ((x[ipart] < xmin) || (x[ipart] > xmax) || (y[ipart] < ymin) || (y[ipart] > ymax)) {
            is_dead[ipart] = 1;
            x[ipart] = NAN;
            y[ipart] = NAN;
        }
    }
}

PyObject* get_npart_to_extend_2d(PyObject* self, PyObject* args) {
    // Parse input arguments
    PyObject* particles_list;
    PyObject* patch_list;
    double dx, dy;
    npy_intp npatches;

    if (!PyArg_ParseTuple(
            args, "OOndd", 
            &particles_list, 
            &patch_list,
            &npatches,
            &dx, &dy
        )
    ) {
        return NULL;
    }

    // Get attributes with cleanup attributes
    AUTOFREE double **x_list = get_attr_array_double(particles_list, npatches, "x");
    AUTOFREE double **y_list = get_attr_array_double(particles_list, npatches, "y");
    AUTOFREE npy_intp *npart_list = get_attr_int(particles_list, npatches, "npart");
    AUTOFREE npy_bool **is_dead_list = get_attr_array_bool(particles_list, npatches, "is_dead");
    
    AUTOFREE npy_intp **neighbor_index_list = get_attr_array_int(patch_list, npatches, "neighbor_ipatch");

    AUTOFREE double *xmin_list = get_attr_double(patch_list, npatches, "xmin");
    AUTOFREE double *xmax_list = get_attr_double(patch_list, npatches, "xmax");
    AUTOFREE double *ymin_list = get_attr_double(patch_list, npatches, "ymin");
    AUTOFREE double *ymax_list = get_attr_double(patch_list, npatches, "ymax");

    // Adjust particle boundaries
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        xmin_list[ipatch] -= 0.5 * dx;
        xmax_list[ipatch] += 0.5 * dx;
        ymin_list[ipatch] -= 0.5 * dy;
        ymax_list[ipatch] += 0.5 * dy;
    }

    // Allocate arrays for particle counts with cleanup attributes
    npy_intp dims = NUM_BOUNDARIES*npatches;
    PyArrayObject *npart_to_extend_array = (PyArrayObject*) PyArray_ZEROS(1, &npatches, NPY_INT64, 0);
    PyArrayObject *npart_incoming_array = (PyArrayObject*) PyArray_ZEROS(1, &npatches, NPY_INT64, 0);
    PyArrayObject *npart_outgoing_array = (PyArrayObject*) PyArray_ZEROS(1, &dims, NPY_INT64, 0);

    npy_intp *npart_to_extend = (npy_intp*) PyArray_DATA(npart_to_extend_array);
    npy_intp *npart_incoming = (npy_intp*) PyArray_DATA(npart_incoming_array);
    npy_intp *npart_outgoing = (npy_intp*) PyArray_DATA(npart_outgoing_array);

    // Count outgoing particles for each patch
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        double *x = x_list[ipatch];
        double *y = y_list[ipatch];
        npy_bool *is_dead = is_dead_list[ipatch];
        double xmin = xmin_list[ipatch];
        double xmax = xmax_list[ipatch];
        double ymin = ymin_list[ipatch];
        double ymax = ymax_list[ipatch];
        npy_intp npart = npart_list[ipatch];
        
        // Count particles going out of bounds
        npy_intp npart_out[NUM_BOUNDARIES] = {0};
        
        count_outgoing_particles(
            x, y, is_dead, xmin, xmax, ymin, ymax, npart,
            npart_out
        );
        
        // Store results in the outgoing array
        for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            npart_outgoing[ipatch * NUM_BOUNDARIES + ibound] = npart_out[ibound];
        }
    }
    
    // Calculate incoming particles for each patch
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        npy_bool *is_dead = is_dead_list[ipatch];
        npy_intp npart = npart_list[ipatch];
        
        // Count incoming particles
        npy_intp npart_new = 0;
        for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            npy_intp i = neighbor_index_list[ipatch][ibound];
            if (i >= 0) {
                npart_new += npart_outgoing[i*NUM_BOUNDARIES + OPPOSITE_BOUNDARY[ibound]];
            }
        }
        
        // Count dead particles
        npy_intp ndead = 0;
        for (npy_intp i = 0; i < npart; i++) {
            if (is_dead[i]) {
                ndead++;
            }
        }
        
        // Calculate number of particles to extend
        if ((npart_new - ndead) > 0) {
            // Reserve more space for new particles (25% extra)
            npart_to_extend[ipatch] = npart_new - ndead + (npy_intp)(npart * 0.25);
        }
        
        npart_incoming[ipatch] = npart_new;
    }
    Py_END_ALLOW_THREADS
    
    PyObject *ret = PyTuple_Pack(3, npart_to_extend_array, npart_incoming_array, npart_outgoing_array);
    return ret;
}

// Fill particles from boundary
PyObject* fill_particles_from_boundary_2d(PyObject* self, PyObject* args) {
    // Parse input arguments
    PyObject *particles_list, *patch_list, *attrs;
    PyArrayObject *npart_incoming_array, *npart_outgoing_array;
    double dx, dy;
    npy_intp npatches;

    if (!PyArg_ParseTuple(
            args, "OOOOnddO", 
            &particles_list, 
            &patch_list,
            &npart_incoming_array,
            &npart_outgoing_array,
            &npatches,
            &dx, &dy, &attrs
        )
    ) {
        return NULL;
    }

    // Get attributes with cleanup attributes
    AUTOFREE double **x_list = get_attr_array_double(particles_list, npatches, "x");
    AUTOFREE double **y_list = get_attr_array_double(particles_list, npatches, "y");
    AUTOFREE npy_intp *npart_list = get_attr_int(particles_list, npatches, "npart");
    AUTOFREE npy_bool **is_dead_list = get_attr_array_bool(particles_list, npatches, "is_dead");

    AUTOFREE npy_intp **neighbor_index_list = get_attr_array_int(patch_list, npatches, "neighbor_ipatch");

    AUTOFREE double *xmin_list = get_attr_double(patch_list, npatches, "xmin");
    AUTOFREE double *xmax_list = get_attr_double(patch_list, npatches, "xmax");
    AUTOFREE double *ymin_list = get_attr_double(patch_list, npatches, "ymin");
    AUTOFREE double *ymax_list = get_attr_double(patch_list, npatches, "ymax");

    AUTOFREE npy_intp *npart_incoming = PyArray_DATA(npart_incoming_array);
    AUTOFREE npy_intp *npart_outgoing = PyArray_DATA(npart_outgoing_array);

    npy_intp nattrs = PyList_Size(attrs);

    // Adjust particle boundaries
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        xmin_list[ipatch] -= 0.5 * dx;
        xmax_list[ipatch] += 0.5 * dx;
        ymin_list[ipatch] -= 0.5 * dy;
        ymax_list[ipatch] += 0.5 * dy;
    }
    
    // Create array of attribute arrays
    AUTOFREE double **attrs_list = malloc(nattrs * npatches * sizeof(double*));
    for (Py_ssize_t iattr = 0; iattr < nattrs; iattr++) {
        PyObject *attr_name = PyList_GetItem(attrs, iattr);
        
        for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
            PyObject *particle = PyList_GetItem(particles_list, ipatch);
            PyObject *attr_array = PyObject_GetAttr(particle, attr_name);
            attrs_list[ipatch*nattrs + iattr] = (double*) PyArray_DATA((PyArrayObject*)attr_array);
            Py_DecRef(attr_array);
        }
    }
    // Number of particles coming from each boundary
    AUTOFREE npy_intp** npart_incoming_boundary_list = (npy_intp**)malloc(npatches * sizeof(npy_intp**));
    // Indices of particles coming from boundary
    AUTOFREE npy_intp*** incoming_indices_list = (npy_intp***)malloc(npatches * sizeof(npy_intp**));
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        npart_incoming_boundary_list[ipatch] = (npy_intp*)malloc(NUM_BOUNDARIES * sizeof(npy_intp*));
        incoming_indices_list[ipatch] = (npy_intp**)malloc(NUM_BOUNDARIES * sizeof(npy_intp*));
        for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            incoming_indices_list[ipatch][ibound] = NULL;
            incoming_indices_list[ipatch][ibound] = 0;
        }
    }

    int num_threads = omp_get_max_threads();
    if (npatches < num_threads) num_threads = npatches;

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel num_threads(num_threads)
    {    // Number of particles coming from each boundary
        #pragma omp for
        for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
            npy_intp npart_new = npart_incoming[ipatch];
            if (npart_new <= 0) {
                continue;
            }
            
            for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
                npy_intp i = neighbor_index_list[ipatch][ibound];
                if (i >= 0) {
                    npart_incoming_boundary_list[ipatch][ibound] = npart_outgoing[i*NUM_BOUNDARIES + OPPOSITE_BOUNDARY[ibound]];
                    if (npart_incoming_boundary_list[ipatch][ibound] > 0) {
                        incoming_indices_list[ipatch][ibound] = (npy_intp*)malloc(npart_incoming_boundary_list[ipatch][ibound] * sizeof(npy_intp));
                    }
                }
            }
        }
        #pragma omp for
        for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
            
            // Get indices of incoming particles
            get_incoming_index(
                x_list[ipatch], y_list[ipatch], is_dead_list[ipatch], npart_list[ipatch],
                xmin_list[ipatch], xmax_list[ipatch], 
                ymin_list[ipatch], ymax_list[ipatch],
                // boundary indices
                neighbor_index_list[ipatch],
                // out
                incoming_indices_list
            );
        }
        #pragma omp for
        for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
            npy_intp npart_new = npart_incoming[ipatch];
            if (npart_new <= 0) {
                continue;
            }

            npy_bool* is_dead = is_dead_list[ipatch];
            npy_intp npart = npart_list[ipatch];
            // Allocate buffer for incoming particles with cleanup attribute
            AUTOFREE double* buffer = malloc(nattrs*npart_new * sizeof(double));                                                                                                                                                                                                                  
            // Fill buffer with boundary particles
            fill_boundary_particles_to_buffer(
                attrs_list, nattrs,
                incoming_indices_list[ipatch],
                npart_incoming_boundary_list[ipatch],
                neighbor_index_list[ipatch],
                buffer
            );
            
            // Fill particles from buffer
            npy_intp ibuff = 0;
            for (npy_intp ipart = 0; ipart < npart; ipart++) {
                if (ibuff >= npart_new) {
                    break;
                }
                if (is_dead[ipart]) {
                    for (npy_intp iattr = 0; iattr < nattrs; iattr++) {
                        attrs_list[ipatch*nattrs + iattr][ipart] = buffer[ibuff*nattrs+iattr];
                    }
                    is_dead[ipart] = 0; // Mark as alive
                    ibuff++;
                }
            }    
        }
        #pragma omp for
        for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
            mark_out_of_bound_as_dead(
                x_list[ipatch], y_list[ipatch], is_dead_list[ipatch], npart_list[ipatch],
                xmin_list[ipatch], xmax_list[ipatch],
                ymin_list[ipatch], ymax_list[ipatch]
            );
        }
    }
    Py_END_ALLOW_THREADS

    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            free(incoming_indices_list[ipatch][ibound]);
        }
        free(npart_incoming_boundary_list[ipatch]);
        free(incoming_indices_list[ipatch]);
    }

    Py_RETURN_NONE;
}

// Module method definitions
static PyMethodDef SyncParticlesMethods[] = {
    {"get_npart_to_extend_2d", get_npart_to_extend_2d, METH_VARARGS, "count the number of particles to be extended, and return the number of new particles"},
    {"fill_particles_from_boundary_2d", fill_particles_from_boundary_2d, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef syncparticlesmodule = {
    PyModuleDef_HEAD_INIT,
    "sync_particles_2d",
    NULL,
    -1,
    SyncParticlesMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_sync_particles_2d(void) {
    import_array();
    return PyModule_Create(&syncparticlesmodule);
}
