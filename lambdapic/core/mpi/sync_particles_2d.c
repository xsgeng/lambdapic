#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>
#include <mpi.h>

#include "../utils/cutils.h"

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

static void get_outgoing_indices(
    double* x, double* y, npy_bool* is_dead,
    double xmin, double xmax, double ymin, double ymax,
    npy_intp npart,
    npy_intp **outgoing_indices
) {
    npy_intp ibuff[NUM_BOUNDARIES] = {0};
    for (npy_intp ip = 0; ip < npart; ip++) {
        if (is_dead[ip]) continue;
        if (y[ip] < ymin) {
            if (x[ip] < xmin) {
                if (outgoing_indices[XMINYMIN] == NULL) continue;
                outgoing_indices[XMINYMIN][ibuff[XMINYMIN]] = ip;
                ibuff[XMINYMIN]++;
                continue;
            }
            else if (x[ip] > xmax) {
                if (outgoing_indices[XMAXYMIN] == NULL) continue;
                outgoing_indices[XMAXYMIN][ibuff[XMAXYMIN]] = ip;
                ibuff[XMAXYMIN]++;
                continue;
            }
            else {
                if (outgoing_indices[YMIN] == NULL) continue;
                outgoing_indices[YMIN][ibuff[YMIN]] = ip;
                ibuff[YMIN]++;
                continue;
            }
        }
        else if (y[ip] > ymax) {
            if (x[ip] < xmin) {
                if (outgoing_indices[XMINYMAX] == NULL) continue;
                outgoing_indices[XMINYMAX][ibuff[XMINYMAX]] = ip;
                ibuff[XMINYMAX]++;
                continue;
            }
            else if (x[ip] > xmax) {
                if (outgoing_indices[XMAXYMAX] == NULL) continue;
                outgoing_indices[XMAXYMAX][ibuff[XMAXYMAX]] = ip;
                ibuff[XMAXYMAX]++;
                continue;
            }
            else {
                if (outgoing_indices[YMAX] == NULL) continue;
                outgoing_indices[YMAX][ibuff[YMAX]] = ip;
                ibuff[YMAX]++;
                continue;
            }
        }
        else {
            if (x[ip] < xmin) {
                if (outgoing_indices[XMIN] == NULL) continue;
                outgoing_indices[XMIN][ibuff[XMIN]] = ip;
                ibuff[XMIN]++;
                continue;
            }
            else if (x[ip] > xmax) {
                if (outgoing_indices[XMAX] == NULL) continue;
                outgoing_indices[XMAX][ibuff[XMAX]] = ip;
                ibuff[XMAX]++;
                continue;
            }
        }
    }
}

PyObject* fill_particles_from_boundary_2d(PyObject* self, PyObject* args) {
    // Parse input arguments
    PyObject* particles_list;
    PyObject* patch_list;
    PyArrayObject* npart_incoming_array, *npart_outgoing_array;
    PyObject* comm_py;
    double dx, dy;
    npy_intp npatches;
    PyObject* attrs;

    if (!PyArg_ParseTuple(
            args, "OOOOOnddO", 
            &particles_list, 
            &patch_list,
            &npart_incoming_array,
            &npart_outgoing_array,
            &comm_py,
            &npatches,
            &dx, &dy,
            &attrs
        )
    ) {
        return NULL;
    }
    // Get MPI communicator from Python object
    PyObject *comm_handle = PyObject_GetAttrString(comm_py, "handle");
    if (!PyLong_Check(comm_handle)) {
        PyErr_SetString(PyExc_TypeError, "comm_py must be a long integer");
        return NULL;
    }
    MPI_Comm comm = PyLong_AsLong(comm_handle);
    Py_DecRef(comm_handle);
    
    // Get MPI rank and size
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Get attributes with cleanup attributes
    AUTOFREE double **x_list = get_attr_array_double(particles_list, npatches, "x");
    AUTOFREE double **y_list = get_attr_array_double(particles_list, npatches, "y");
    AUTOFREE npy_intp *npart_list = get_attr_int(particles_list, npatches, "npart");
    AUTOFREE npy_bool **is_dead_list = get_attr_array_bool(particles_list, npatches, "is_dead");
    
    AUTOFREE npy_intp **neighbor_index_list = get_attr_array_int(patch_list, npatches, "neighbor_index");
    AUTOFREE npy_intp **neighbor_rank_list = get_attr_array_int(patch_list, npatches, "neighbor_rank");
    AUTOFREE npy_intp *index_list = get_attr_int(patch_list, npatches, "index");

    AUTOFREE double *xmin_list = get_attr_double(patch_list, npatches, "xmin");
    AUTOFREE double *xmax_list = get_attr_double(patch_list, npatches, "xmax");
    AUTOFREE double *ymin_list = get_attr_double(patch_list, npatches, "ymin");
    AUTOFREE double *ymax_list = get_attr_double(patch_list, npatches, "ymax");

    int nattrs = PyList_Size(attrs);
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

    // Adjust particle boundaries
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        xmin_list[ipatch] -= 0.5 * dx;
        xmax_list[ipatch] += 0.5 * dx;
        ymin_list[ipatch] -= 0.5 * dy;
        ymax_list[ipatch] += 0.5 * dy;
    }

    // Get arrays for particle counts
    npy_intp *npart_incoming = (npy_intp*) PyArray_DATA(npart_incoming_array);
    npy_intp *npart_outgoing = (npy_intp*) PyArray_DATA(npart_outgoing_array);

    // send and recv buffers
    AUTOFREE double **attrs_send = (double **)malloc(npatches * NUM_BOUNDARIES * sizeof(double *));
    AUTOFREE double **attrs_recv = (double **)malloc(npatches * NUM_BOUNDARIES * sizeof(double *));

    // Allocate MPI request arrays
    AUTOFREE MPI_Request *send_requests = (MPI_Request*)malloc(npatches * NUM_BOUNDARIES * sizeof(MPI_Request));

    Py_BEGIN_ALLOW_THREADS

    // fill send buffers and send
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

        npy_intp *outgoing_indices[NUM_BOUNDARIES] = {NULL};
        
        for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            npy_intp i = ipatch * NUM_BOUNDARIES + ibound;
            
            if (npart_outgoing[i] == 0) {
                attrs_send[i] = NULL;
            } else {
                outgoing_indices[ibound] = (npy_intp*)malloc(npart_outgoing[i] * sizeof(npy_intp));
                attrs_send[i] = (double *)malloc(nattrs*npart_outgoing[i] * sizeof(double));
            }

            if (npart_incoming[i] == 0) {
                attrs_recv[i] = NULL;
            } else {
                attrs_recv[i] = (double *)malloc(nattrs*npart_incoming[i] * sizeof(double));
            }
        }

        get_outgoing_indices(x, y, is_dead, xmin, xmax, ymin, ymax, npart, outgoing_indices);
        for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            int neighbor_rank = neighbor_rank_list[ipatch][ibound];
            npy_intp npart_out = npart_outgoing[ipatch * NUM_BOUNDARIES + ibound];
            if (neighbor_rank < 0) {
                send_requests[ipatch * NUM_BOUNDARIES + ibound] = MPI_REQUEST_NULL;
                continue;
            }

            
            for (npy_intp iattr = 0; iattr < nattrs; iattr++) {
                for (npy_intp ip = 0; ip < npart_out; ip++) {
                    npy_intp idx = outgoing_indices[ibound][ip];
                    attrs_send[ipatch * NUM_BOUNDARIES + ibound][ip*nattrs+iattr] = attrs_list[ipatch*nattrs+iattr][idx];
                }
            }
            int index = index_list[ipatch];
            int send_tag = index*NUM_BOUNDARIES + ibound;

            MPI_Isend(
                attrs_send[ipatch * NUM_BOUNDARIES + ibound], nattrs*npart_outgoing[ipatch * NUM_BOUNDARIES + ibound], MPI_DOUBLE, 
                neighbor_rank, send_tag, comm, &send_requests[ipatch * NUM_BOUNDARIES + ibound]
            );

            // mark outgoing particles as dead
            for (npy_intp ip = 0; ip < npart_out; ip++) {
                npy_intp idx = outgoing_indices[ibound][ip];
                is_dead[idx] = 1;
            }
        }
        for (int ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            free(outgoing_indices[ibound]);
        }
    }

    // recv
    #pragma omp parallel for
    for (npy_intp i = 0; i < npatches*NUM_BOUNDARIES; i++) {
        int ipatch = i / NUM_BOUNDARIES;
        int ibound = i % NUM_BOUNDARIES;
        int neighbor_rank = neighbor_rank_list[ipatch][ibound];
        npy_intp npart_in = npart_incoming[ipatch * NUM_BOUNDARIES + ibound];
        if (neighbor_rank < 0) {
            continue;
        }

        int neighbor_index = neighbor_index_list[ipatch][ibound];
        // Tag based on patch index and boundary
        int recv_tag = neighbor_index*NUM_BOUNDARIES + OPPOSITE_BOUNDARY[ibound];
        MPI_Recv(
            attrs_recv[ipatch * NUM_BOUNDARIES + ibound], nattrs*npart_in, MPI_DOUBLE, 
            neighbor_rank, recv_tag, comm, MPI_STATUS_IGNORE
        );
    }

    // Fill particles from buffer
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        npy_bool *is_dead = is_dead_list[ipatch];
        
        npy_intp ipart = 0;
        for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            npy_intp npart_new = npart_incoming[ipatch * NUM_BOUNDARIES + ibound];
            if (npart_new <= 0) {
                continue;
            }
            double *buffer = attrs_recv[ipatch * NUM_BOUNDARIES + ibound];
            for (npy_intp ibuff = 0; ibuff < npart_new; ibuff++) {
                while (!is_dead[ipart]) {
                    ipart++;
                }
                for (npy_intp iattr = 0; iattr < nattrs; iattr++) {
                    attrs_list[ipatch*nattrs + iattr][ipart] = buffer[ibuff*nattrs + iattr];
                }
                is_dead[ipart] = 0; // Mark as alive
            }
        }
    }
    Py_END_ALLOW_THREADS

    MPI_Waitall(npatches * NUM_BOUNDARIES, send_requests, MPI_STATUSES_IGNORE);
    for (npy_intp i = 0; i < npatches*NUM_BOUNDARIES; i++) {
        free(attrs_send[i]);
        free(attrs_recv[i]);
    }

    Py_RETURN_NONE;
}

PyObject* get_npart_to_extend_2d(PyObject* self, PyObject* args) {
    // Parse input arguments
    PyObject* particles_list;
    PyObject* patch_list;
    PyObject* comm_py;
    double dx, dy;
    npy_intp npatches;

    if (!PyArg_ParseTuple(
            args, "OOOndd", 
            &particles_list, 
            &patch_list,
            &comm_py,
            &npatches,
            &dx, &dy
        )
    ) {
        return NULL;
    }
    // Get MPI communicator from Python object
    PyObject *comm_handle = PyObject_GetAttrString(comm_py, "handle");
    if (!PyLong_Check(comm_handle)) {
        PyErr_SetString(PyExc_TypeError, "comm_py must be a long integer");
        return NULL;
    }
    MPI_Comm comm = PyLong_AsLong(comm_handle);
    Py_DecRef(comm_handle);
    
    // Get MPI rank and size
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Get attributes with cleanup attributes
    AUTOFREE double **x_list = get_attr_array_double(particles_list, npatches, "x");
    AUTOFREE double **y_list = get_attr_array_double(particles_list, npatches, "y");
    AUTOFREE npy_intp *npart_list = get_attr_int(particles_list, npatches, "npart");
    AUTOFREE npy_bool **is_dead_list = get_attr_array_bool(particles_list, npatches, "is_dead");
    
    AUTOFREE npy_intp **neighbor_index_list = get_attr_array_int(patch_list, npatches, "neighbor_index");
    AUTOFREE npy_intp **neighbor_rank_list = get_attr_array_int(patch_list, npatches, "neighbor_rank");
    AUTOFREE npy_intp *index_list = get_attr_int(patch_list, npatches, "index");

    AUTOFREE double *xmin_list = get_attr_double(patch_list, npatches, "xmin");
    AUTOFREE double *xmax_list = get_attr_double(patch_list, npatches, "xmax");
    AUTOFREE double *ymin_list = get_attr_double(patch_list, npatches, "ymin");
    AUTOFREE double *ymax_list = get_attr_double(patch_list, npatches, "ymax");

    // Adjust particle boundaries
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        xmin_list[ipatch] -= 0.5 * dx;
        xmax_list[ipatch] += 0.5 * dx;
        ymin_list[ipatch] -= 0.5 * dy;
        ymax_list[ipatch] += 0.5 * dy;
    }

    // Allocate arrays for particle counts with cleanup attributes
    npy_intp dims[2] = {npatches, NUM_BOUNDARIES};
    PyArrayObject *npart_to_extend_array = (PyArrayObject*) PyArray_ZEROS(1, &npatches, NPY_INT64, 0);
    PyArrayObject *npart_incoming_array = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_INT64, 0);
    PyArrayObject *npart_outgoing_array = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_INT64, 0);

    npy_intp *npart_to_extend = (npy_intp*) PyArray_DATA(npart_to_extend_array);
    npy_intp *npart_incoming = (npy_intp*) PyArray_DATA(npart_incoming_array);
    npy_intp *npart_outgoing = (npy_intp*) PyArray_DATA(npart_outgoing_array);

    // Allocate MPI request arrays
    AUTOFREE MPI_Request *sendrecv_requests = (MPI_Request*)malloc(npatches * NUM_BOUNDARIES * sizeof(MPI_Request));

    Py_BEGIN_ALLOW_THREADS
    
    // Count outgoing particles for each patch
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
            int neighbor_rank = neighbor_rank_list[ipatch][ibound];
            if (neighbor_rank < 0) {
                continue;
            }
            npart_outgoing[ipatch * NUM_BOUNDARIES + ibound] = npart_out[ibound];
        }
    }
    
    // Post non-blocking sends and receives for particle counts
    #pragma omp parallel for
    for (npy_intp i = 0; i < npatches*NUM_BOUNDARIES; i++) {
        int ipatch = i / NUM_BOUNDARIES;
        int ibound = i % NUM_BOUNDARIES;
        int neighbor_rank = neighbor_rank_list[ipatch][ibound];
        
        // Skip if no neighbor in this direction
        if (neighbor_rank < 0){
            sendrecv_requests[i] = MPI_REQUEST_NULL;
            continue;
        } 
        
        int index = index_list[ipatch];
        int neighbor_index = neighbor_index_list[ipatch][ibound];
        // Tag based on patch index and boundary
        int send_tag = index*NUM_BOUNDARIES + ibound;
        int recv_tag = neighbor_index*NUM_BOUNDARIES + OPPOSITE_BOUNDARY[ibound];
        // printf("send_tag: %i", index*8 + ibound);
            // printf("rank: %d, ipatch: %d, ibound: %d, neighbor_rank: %d, index: %d, neighbor_index: %d, send_tag: %d, recv_tag: %d\n", rank, ipatch, ibound, neighbor_rank, index, neighbor_index, send_tag, recv_tag);
        
        // Thread-multiple access to MPI functions
        MPI_Isendrecv(&npart_outgoing[i], 1, MPI_LONG, neighbor_rank, send_tag,
                      &npart_incoming[i], 1, MPI_LONG, neighbor_rank, recv_tag,
                      comm, &sendrecv_requests[i]);
    }
    
    // Wait for all communications to complete
    MPI_Waitall(npatches*NUM_BOUNDARIES, sendrecv_requests, MPI_STATUSES_IGNORE);
    
    // Process received counts
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        npy_intp npart_new = 0;
        
        for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            int neighbor_rank = neighbor_rank_list[ipatch][ibound];
            if (neighbor_rank < 0) continue;
            
            npart_new += npart_incoming[ipatch * NUM_BOUNDARIES + ibound];
        }
        if (npart_new == 0) continue;
        
        // Count dead particles
        npy_bool *is_dead = is_dead_list[ipatch];
        npy_intp npart = npart_list[ipatch];
        npy_intp ndead = 0;
        
        for (npy_intp i = 0; i < npart; i++) {
            if (is_dead[i]) {
                ndead++;
            }
        }
        
        // Calculate number of particles to extend
        if ((npart_new - ndead) > 0) {
            // Reserve more space for new particles
            npart_to_extend[ipatch] = npart_new - ndead;
        }
    }
    
    // Cleanup
    
    // Synchronize all ranks
    MPI_Barrier(comm);
    
    Py_END_ALLOW_THREADS
    
    PyObject *ret = PyTuple_Pack(3, npart_to_extend_array, npart_incoming_array, npart_outgoing_array);
    return ret;
}

// Module method definitions
static PyMethodDef SyncParticlesMethods[] = {
    {"get_npart_to_extend_2d", get_npart_to_extend_2d, METH_VARARGS, "count the number of particles to be extended, and return the number of new particles"},
    // {"fill_particles_from_boundary_2d", fill_particles_from_boundary_2d, METH_VARARGS, ""},
    {"fill_particles_from_boundary_2d", fill_particles_from_boundary_2d, METH_VARARGS, "fill particles from boundary using MPI"},
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
