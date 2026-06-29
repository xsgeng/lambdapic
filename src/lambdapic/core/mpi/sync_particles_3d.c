#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>
#include <mpi.h>
#include <mpi4py/mpi4py.h>

#include "../utils/cutils.h"

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

// Implementation of count_outgoing_particles function
static void count_outgoing_particles(
    double* x, double* y, double* z, npy_bool* is_dead,
    double xmin, double xmax, double ymin, double ymax, double zmin, double zmax,
    npy_intp npart,
    npy_intp* npart_out
) {
    for (npy_intp ip = 0; ip < npart; ip++) {
        if (is_dead[ip]) continue;
        if (z[ip] < zmin) {
            if (y[ip] < ymin) {
                if (x[ip] < xmin) {
                    (npart_out[XMINYMINZMIN])++;
                    continue;
                } else if (x[ip] > xmax) {
                    (npart_out[XMAXYMINZMIN])++;
                    continue;
                } else {
                    (npart_out[YMINZMIN])++;
                    continue;
                }
            } else if (y[ip] > ymax) {
                if (x[ip] < xmin) {
                    (npart_out[XMINYMAXZMIN])++;
                    continue;
                } else if (x[ip] > xmax) {
                    (npart_out[XMAXYMAXZMIN])++;
                    continue;
                } else {
                    (npart_out[YMAXZMIN])++;
                    continue;
                }
            } else {
                if (x[ip] < xmin) {
                    (npart_out[XMINZMIN])++;
                    continue;
                } else if (x[ip] > xmax) {
                    (npart_out[XMAXZMIN])++;
                    continue;
                } else {
                    (npart_out[ZMIN])++;
                    continue;
                }
            }
        } else if (z[ip] > zmax) {
            if (y[ip] < ymin) {
                if (x[ip] < xmin) {
                    (npart_out[XMINYMINZMAX])++;
                    continue;
                } else if (x[ip] > xmax) {
                    (npart_out[XMAXYMINZMAX])++;
                    continue;
                } else {
                    (npart_out[YMINZMAX])++;
                    continue;
                }
            } else if (y[ip] > ymax) {
                if (x[ip] < xmin) {
                    (npart_out[XMINYMAXZMAX])++;
                    continue;
                } else if (x[ip] > xmax) {
                    (npart_out[XMAXYMAXZMAX])++;
                    continue;
                } else {
                    (npart_out[YMAXZMAX])++;
                    continue;
                }
            } else {
                if (x[ip] < xmin) {
                    (npart_out[XMINZMAX])++;
                    continue;
                } else if (x[ip] > xmax) {
                    (npart_out[XMAXZMAX])++;
                    continue;
                } else {
                    (npart_out[ZMAX])++;
                    continue;
                }
            }
        } else {
            if (y[ip] < ymin) {
                if (x[ip] < xmin) {
                    (npart_out[XMINYMIN])++;
                    continue;
                } else if (x[ip] > xmax) {
                    (npart_out[XMAXYMIN])++;
                    continue;
                } else {
                    (npart_out[YMIN])++;
                    continue;
                }
            } else if (y[ip] > ymax) {
                if (x[ip] < xmin) {
                    (npart_out[XMINYMAX])++;
                    continue;
                } else if (x[ip] > xmax) {
                    (npart_out[XMAXYMAX])++;
                    continue;
                } else {
                    (npart_out[YMAX])++;
                    continue;
                }
            } else {
                if (x[ip] < xmin) {
                    (npart_out[XMIN])++;
                    continue;
                } else if (x[ip] > xmax) {
                    (npart_out[XMAX])++;
                    continue;
                } else {
                    continue;
                }
            }
        }
    }
}
#define SET_INCOMING_INDEX(BOUND) \
    if (outgoing_indices[BOUND] == NULL) continue;\
    outgoing_indices[BOUND][ibuff[BOUND]] = ip;\
    ibuff[BOUND]++;\
    continue;

static void get_outgoing_indices(
    double* x, double* y, double* z, npy_bool* is_dead,
    double xmin, double xmax, double ymin, double ymax, double zmin, double zmax,
    npy_intp npart,
    npy_intp **outgoing_indices
) {
    npy_intp ibuff[NUM_BOUNDARIES] = {0};
    for (npy_intp ip = 0; ip < npart; ip++) {
        if (is_dead[ip]) continue;
        if (z[ip] < zmin) {
            if (y[ip] < ymin) {
                if (x[ip] < xmin) {
                    SET_INCOMING_INDEX(XMINYMINZMIN);
                } else if (x[ip] > xmax) {
                    SET_INCOMING_INDEX(XMAXYMINZMIN)
                } else {
                    SET_INCOMING_INDEX(YMINZMIN)
                }
            } else if (y[ip] > ymax) {
                if (x[ip] < xmin) {
                    SET_INCOMING_INDEX(XMINYMAXZMIN)
                } else if (x[ip] > xmax) {
                    SET_INCOMING_INDEX(XMAXYMAXZMIN)
                } else {
                    SET_INCOMING_INDEX(YMAXZMIN)
                }
            } else {
                if (x[ip] < xmin) {
                    SET_INCOMING_INDEX(XMINZMIN)
                } else if (x[ip] > xmax) {
                    SET_INCOMING_INDEX(XMAXZMIN)
                } else {
                    SET_INCOMING_INDEX(ZMIN)
                }
            }
        } else if (z[ip] > zmax) {
            if (y[ip] < ymin) {
                if (x[ip] < xmin) {
                    SET_INCOMING_INDEX(XMINYMINZMAX)
                } else if (x[ip] > xmax) {
                    SET_INCOMING_INDEX(XMAXYMINZMAX)
                } else {
                    SET_INCOMING_INDEX(YMINZMAX)
                }
            } else if (y[ip] > ymax) {
                if (x[ip] < xmin) {
                    SET_INCOMING_INDEX(XMINYMAXZMAX)
                } else if (x[ip] > xmax) {
                    SET_INCOMING_INDEX(XMAXYMAXZMAX)
                } else {
                    SET_INCOMING_INDEX(YMAXZMAX)
                }
            } else {
                if (x[ip] < xmin) {
                    SET_INCOMING_INDEX(XMINZMAX)
                } else if (x[ip] > xmax) {
                    SET_INCOMING_INDEX(XMAXZMAX)
                } else {
                    SET_INCOMING_INDEX(ZMAX)
                }
            }
        } else {
            if (y[ip] < ymin) {
                if (x[ip] < xmin) {
                    SET_INCOMING_INDEX(XMINYMIN)
                } else if (x[ip] > xmax) {
                    SET_INCOMING_INDEX(XMAXYMIN)
                } else {
                    SET_INCOMING_INDEX(YMIN)
                }
            } else if (y[ip] > ymax) {
                if (x[ip] < xmin) {
                    SET_INCOMING_INDEX(XMINYMAX)
                } else if (x[ip] > xmax) {
                    SET_INCOMING_INDEX(XMAXYMAX)
                } else {
                    SET_INCOMING_INDEX(YMAX)
                }
            } else {
                if (x[ip] < xmin) {
                    SET_INCOMING_INDEX(XMIN)
                } else if (x[ip] > xmax) {
                    SET_INCOMING_INDEX(XMAX)
                } else {
                }
            }
        }
    }
}

// Apply periodic boundary conditions for a single coordinate
static void handle_periodic(
    double* buffer, npy_intp ibuff,
    double min_global, double max_global, double L,
    double min_patch, double max_patch,
    double cell_size
) {
    double coord = buffer[ibuff];
    
    if (coord > max_global && fabs(min_patch - min_global) < cell_size) {
        buffer[ibuff] -= L;
    }
    if (coord < min_global && fabs(max_patch - max_global) < cell_size) {
        buffer[ibuff] += L;
    }
}


/* ------------------------------------------------------------------ */
/* Handle struct for split _start/_wait particle fill                  */
/* ------------------------------------------------------------------ */

typedef struct {
    MPI_Comm comm;
    MPI_Request *send_requests;
    MPI_Request *recv_requests;
    double **attrs_send;
    double **attrs_recv;
    double **attrs_list;
    npy_bool **is_dead_list;
    npy_intp *npart_incoming;
    npy_intp **neighbor_index_list;
    npy_intp **neighbor_rank_list;
    double *xmin_list, *xmax_list, *ymin_list, *ymax_list, *zmin_list, *zmax_list;
    int nattrs;
    int npatches;
    int iattr_x, iattr_y, iattr_z;
    double xmin_global, xmax_global, ymin_global, ymax_global, zmin_global, zmax_global;
    double Lx, Ly, Lz, dx, dy, dz;
    int finalized;
} ParticleSyncHandle;


/* ------------------------------------------------------------------ */
/* Capsule destructor (safety net if _wait not called)                */
/* ------------------------------------------------------------------ */

static void particle_sync_destructor(PyObject* capsule) {
    ParticleSyncHandle* h = (ParticleSyncHandle*)PyCapsule_GetPointer(capsule, "sync_particles_3d.fill");
    if (!h) { PyErr_Clear(); return; }
    if (h->finalized) { free(h); return; }

    int total = h->npatches * NUM_BOUNDARIES;
    if (h->send_requests) {
        Py_BEGIN_ALLOW_THREADS
        MPI_Waitall(total, h->send_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(total, h->recv_requests, MPI_STATUSES_IGNORE);
        Py_END_ALLOW_THREADS
        free(h->send_requests);
        free(h->recv_requests);
    }
    if (h->attrs_send) {
        for (int i = 0; i < total; i++) {
            free(h->attrs_send[i]);
            free(h->attrs_recv[i]);
        }
        free(h->attrs_send);
        free(h->attrs_recv);
    }
    free(h->attrs_list);
    free(h->is_dead_list);
    free(h->npart_incoming);
    free(h->neighbor_index_list);
    free(h->neighbor_rank_list);
    free(h->xmin_list); free(h->xmax_list);
    free(h->ymin_list); free(h->ymax_list);
    free(h->zmin_list); free(h->zmax_list);
    free(h);
}


/* ------------------------------------------------------------------ */
/* fill_particles_from_boundary_3d: _start / _wait / wrapper          */
/* ------------------------------------------------------------------ */

static PyObject* fill_particles_from_boundary_3d_start(PyObject* self, PyObject* args) {
    PyObject* particles_list;
    PyObject* patch_list;
    PyArrayObject* npart_incoming_array, *npart_outgoing_array;
    PyObject* comm_py;
    double dx, dy, dz;
    double xmin_global, xmax_global, ymin_global, ymax_global, zmin_global, zmax_global;
    npy_intp npatches;
    PyObject* attrs;

    if (!PyArg_ParseTuple(
            args, "OOOOOndddddddddO", 
            &particles_list, 
            &patch_list,
            &npart_incoming_array,
            &npart_outgoing_array,
            &comm_py,
            &npatches,
            &dx, &dy, &dz,
            &xmin_global, &xmax_global, &ymin_global, &ymax_global, &zmin_global, &zmax_global,
            &attrs
        )
    ) {
        return NULL;
    }
    MPI_Comm *comm_p = NULL;
    comm_p = PyMPIComm_Get(comm_py);
    MPI_Comm comm = *comm_p;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double Lx = xmax_global - xmin_global;
    double Ly = ymax_global - ymin_global;
    double Lz = zmax_global - zmin_global;

    double **x_list = get_attr_array_double(particles_list, npatches, "x");
    double **y_list = get_attr_array_double(particles_list, npatches, "y");
    double **z_list = get_attr_array_double(particles_list, npatches, "z");
    npy_intp *npart_list = get_attr_int(particles_list, npatches, "npart");
    npy_bool **is_dead_list = get_attr_array_bool(particles_list, npatches, "is_dead");
    
    npy_intp **neighbor_index_list = get_attr_array_int(patch_list, npatches, "neighbor_index");
    npy_intp **neighbor_rank_list = get_attr_array_int(patch_list, npatches, "neighbor_rank");
    npy_intp *index_list = get_attr_int(patch_list, npatches, "index");

    double *xmin_list = get_attr_double(patch_list, npatches, "xmin");
    double *xmax_list = get_attr_double(patch_list, npatches, "xmax");
    double *ymin_list = get_attr_double(patch_list, npatches, "ymin");
    double *ymax_list = get_attr_double(patch_list, npatches, "ymax");
    double *zmin_list = get_attr_double(patch_list, npatches, "zmin");
    double *zmax_list = get_attr_double(patch_list, npatches, "zmax");

    int nattrs = PyList_Size(attrs);
    Py_ssize_t iattr_x = -1, iattr_y = -1, iattr_z = -1;
    double **attrs_list = malloc(nattrs * npatches * sizeof(double*));
    for (Py_ssize_t iattr = 0; iattr < nattrs; iattr++) {
        PyObject *attr_name = PyList_GetItem(attrs, iattr);

        if (PyUnicode_CompareWithASCIIString(attr_name, "x") == 0) iattr_x = iattr;
        if (PyUnicode_CompareWithASCIIString(attr_name, "y") == 0) iattr_y = iattr;
        if (PyUnicode_CompareWithASCIIString(attr_name, "z") == 0) iattr_z = iattr;
        
        for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
            PyObject *particle = PyList_GetItem(particles_list, ipatch);
            PyObject *attr_array = PyObject_GetAttr(particle, attr_name);
            attrs_list[ipatch*nattrs + iattr] = (double*) PyArray_DATA((PyArrayObject*)attr_array);
            Py_DecRef(attr_array);
        }
    }

    if (iattr_x < 0 || iattr_y < 0 || iattr_z < 0) {
        PyErr_SetString(PyExc_ValueError, "attrs must contain 'x', 'y', and 'z'");
        return NULL;
    }

    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        xmin_list[ipatch] -= 0.5 * dx;
        xmax_list[ipatch] += 0.5 * dx;
        ymin_list[ipatch] -= 0.5 * dy;
        ymax_list[ipatch] += 0.5 * dy;
        zmin_list[ipatch] -= 0.5 * dz;
        zmax_list[ipatch] += 0.5 * dz;
    }

    npy_intp *npart_incoming_src = (npy_intp*) PyArray_DATA(npart_incoming_array);
    npy_intp *npart_outgoing = (npy_intp*) PyArray_DATA(npart_outgoing_array);

    int total = npatches * NUM_BOUNDARIES;
    npy_intp *npart_incoming = (npy_intp*)malloc(total * sizeof(npy_intp));
    memcpy(npart_incoming, npart_incoming_src, total * sizeof(npy_intp));
    double **attrs_send = (double **)malloc(total * sizeof(double *));
    double **attrs_recv = (double **)malloc(total * sizeof(double *));

    MPI_Request *send_requests = (MPI_Request*)malloc(total * sizeof(MPI_Request));
    MPI_Request *recv_requests = (MPI_Request*)malloc(total * sizeof(MPI_Request));

    Py_BEGIN_ALLOW_THREADS

    // fill send buffers and post sends
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        double *x = x_list[ipatch];
        double *y = y_list[ipatch];
        double *z = z_list[ipatch];
        npy_bool *is_dead = is_dead_list[ipatch];
        double xmin = xmin_list[ipatch];
        double xmax = xmax_list[ipatch];
        double ymin = ymin_list[ipatch];
        double ymax = ymax_list[ipatch];
        double zmin = zmin_list[ipatch];
        double zmax = zmax_list[ipatch];
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

        get_outgoing_indices(x, y, z, is_dead, xmin, xmax, ymin, ymax, zmin, zmax, npart, outgoing_indices);
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

    Py_END_ALLOW_THREADS

    free(x_list);
    free(y_list);
    free(z_list);
    free(npart_list);
    free(index_list);

    ParticleSyncHandle* handle = (ParticleSyncHandle*)malloc(sizeof(ParticleSyncHandle));
    handle->comm = comm;
    handle->send_requests = send_requests;
    handle->recv_requests = recv_requests;
    handle->attrs_send = attrs_send;
    handle->attrs_recv = attrs_recv;
    handle->attrs_list = attrs_list;
    handle->is_dead_list = is_dead_list;
    handle->npart_incoming = npart_incoming;
    handle->neighbor_index_list = neighbor_index_list;
    handle->neighbor_rank_list = neighbor_rank_list;
    handle->xmin_list = xmin_list;
    handle->xmax_list = xmax_list;
    handle->ymin_list = ymin_list;
    handle->ymax_list = ymax_list;
    handle->zmin_list = zmin_list;
    handle->zmax_list = zmax_list;
    handle->nattrs = nattrs;
    handle->npatches = npatches;
    handle->iattr_x = iattr_x;
    handle->iattr_y = iattr_y;
    handle->iattr_z = iattr_z;
    handle->xmin_global = xmin_global;
    handle->xmax_global = xmax_global;
    handle->ymin_global = ymin_global;
    handle->ymax_global = ymax_global;
    handle->zmin_global = zmin_global;
    handle->zmax_global = zmax_global;
    handle->Lx = Lx;
    handle->Ly = Ly;
    handle->Lz = Lz;
    handle->dx = dx;
    handle->dy = dy;
    handle->dz = dz;

    PyErr_Clear();
    return PyCapsule_New(handle, "sync_particles_3d.fill", particle_sync_destructor);
}

static PyObject* fill_particles_from_boundary_3d_wait(PyObject* self, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return NULL;
    }

    ParticleSyncHandle* h = (ParticleSyncHandle*)PyCapsule_GetPointer(capsule, "sync_particles_3d.fill");
    if (!h) {
        return NULL;
    }

    int total = h->npatches * NUM_BOUNDARIES;

    Py_BEGIN_ALLOW_THREADS

    // Post blocking receives (same as original monolithic code)
    #pragma omp parallel for
    for (npy_intp i = 0; i < (npy_intp)total; i++) {
        int ipatch = i / NUM_BOUNDARIES;
        int ibound = i % NUM_BOUNDARIES;
        int neighbor_rank = h->neighbor_rank_list[ipatch][ibound];
        npy_intp npart_in = h->npart_incoming[ipatch * NUM_BOUNDARIES + ibound];
        if (neighbor_rank < 0) {
            continue;
        }

        int neighbor_index = h->neighbor_index_list[ipatch][ibound];
        int recv_tag = neighbor_index*NUM_BOUNDARIES + OPPOSITE_BOUNDARY[ibound];
        MPI_Recv(
            h->attrs_recv[ipatch * NUM_BOUNDARIES + ibound], h->nattrs*npart_in, MPI_DOUBLE,
            neighbor_rank, recv_tag, h->comm, MPI_STATUS_IGNORE
        );
    }

    // Fill particles from received buffers
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < h->npatches; ipatch++) {
        npy_bool *is_dead = h->is_dead_list[ipatch];
        
        npy_intp ipart = 0;
        for (npy_intp ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            npy_intp npart_new = h->npart_incoming[ipatch * NUM_BOUNDARIES + ibound];
            if (npart_new <= 0) {
                continue;
            }
            double *buffer = h->attrs_recv[ipatch * NUM_BOUNDARIES + ibound];
            for (npy_intp ibuff = 0; ibuff < npart_new; ibuff++) {
                while (!is_dead[ipart]) {
                    ipart++;
                }
                for (npy_intp iattr = 0; iattr < h->nattrs; iattr++) {
                    if (iattr == h->iattr_x) {
                        handle_periodic(
                            buffer, ibuff*h->nattrs+iattr,
                            h->xmin_global, h->xmax_global, h->Lx,
                            h->xmin_list[ipatch], h->xmax_list[ipatch],
                            h->dx
                        );
                    }
                    if (iattr == h->iattr_y) {
                        handle_periodic(
                            buffer, ibuff*h->nattrs+iattr,
                            h->ymin_global, h->ymax_global, h->Ly,
                            h->ymin_list[ipatch], h->ymax_list[ipatch],
                            h->dy
                        );
                    }
                    if (iattr == h->iattr_z) {
                        handle_periodic(
                            buffer, ibuff*h->nattrs+iattr,
                            h->zmin_global, h->zmax_global, h->Lz,
                            h->zmin_list[ipatch], h->zmax_list[ipatch],
                            h->dz
                        );
                    }
                    h->attrs_list[ipatch*h->nattrs + iattr][ipart] = buffer[ibuff*h->nattrs + iattr];
                }
                is_dead[ipart] = 0;
            }
        }
    }

    // Wait for all sends to complete
    MPI_Waitall(total, h->send_requests, MPI_STATUSES_IGNORE);

    // Free buffers
    for (int i = 0; i < total; i++) {
        free(h->attrs_send[i]);
        free(h->attrs_recv[i]);
    }

    Py_END_ALLOW_THREADS

    free(h->send_requests);
    free(h->recv_requests);
    free(h->attrs_send);
    free(h->attrs_recv);
    free(h->attrs_list);
    free(h->is_dead_list);
    free(h->npart_incoming);
    free(h->neighbor_index_list);
    free(h->neighbor_rank_list);
    free(h->xmin_list); free(h->xmax_list);
    free(h->ymin_list); free(h->ymax_list);
    free(h->zmin_list); free(h->zmax_list);
    h->finalized = 1;
    Py_RETURN_NONE;
}

static PyObject* fill_particles_from_boundary_3d(PyObject* self, PyObject* args) {
    PyObject* capsule = fill_particles_from_boundary_3d_start(self, args);
    if (!capsule) return NULL;
    PyObject* wait_args = PyTuple_Pack(1, capsule);
    PyObject* result = fill_particles_from_boundary_3d_wait(self, wait_args);
    Py_DECREF(wait_args);
    Py_DECREF(capsule);
    return result;
}

PyObject* get_npart_to_extend_3d(PyObject* self, PyObject* args) {
    // Parse input arguments
    PyObject* particles_list;
    PyObject* patch_list;
    PyObject* comm_py;
    double dx, dy, dz;
    npy_intp npatches;

    if (!PyArg_ParseTuple(
            args, "OOOnddd", 
            &particles_list, 
            &patch_list,
            &comm_py,
            &npatches,
            &dx, &dy, &dz
        )
    ) {
        return NULL;
    }
    // Get MPI communicator from Python object
    MPI_Comm *comm_p = NULL;
    comm_p = PyMPIComm_Get(comm_py);
    MPI_Comm comm = *comm_p;
    
    // Get MPI rank and size
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Get attributes with cleanup attributes
    AUTOFREE double **x_list = get_attr_array_double(particles_list, npatches, "x");
    AUTOFREE double **y_list = get_attr_array_double(particles_list, npatches, "y");
    AUTOFREE double **z_list = get_attr_array_double(particles_list, npatches, "z");
    AUTOFREE npy_intp *npart_list = get_attr_int(particles_list, npatches, "npart");
    AUTOFREE npy_bool **is_dead_list = get_attr_array_bool(particles_list, npatches, "is_dead");
    
    AUTOFREE npy_intp **neighbor_index_list = get_attr_array_int(patch_list, npatches, "neighbor_index");
    AUTOFREE npy_intp **neighbor_rank_list = get_attr_array_int(patch_list, npatches, "neighbor_rank");
    AUTOFREE npy_intp *index_list = get_attr_int(patch_list, npatches, "index");

    AUTOFREE double *xmin_list = get_attr_double(patch_list, npatches, "xmin");
    AUTOFREE double *xmax_list = get_attr_double(patch_list, npatches, "xmax");
    AUTOFREE double *ymin_list = get_attr_double(patch_list, npatches, "ymin");
    AUTOFREE double *ymax_list = get_attr_double(patch_list, npatches, "ymax");
    AUTOFREE double *zmin_list = get_attr_double(patch_list, npatches, "zmin");
    AUTOFREE double *zmax_list = get_attr_double(patch_list, npatches, "zmax");

    // Adjust particle boundaries
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        xmin_list[ipatch] -= 0.5 * dx;
        xmax_list[ipatch] += 0.5 * dx;
        ymin_list[ipatch] -= 0.5 * dy;
        ymax_list[ipatch] += 0.5 * dy;
        zmin_list[ipatch] -= 0.5 * dz;
        zmax_list[ipatch] += 0.5 * dz;
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
    AUTOFREE MPI_Request *send_requests = (MPI_Request*)malloc(npatches * NUM_BOUNDARIES * sizeof(MPI_Request));
    AUTOFREE MPI_Request *recv_requests = (MPI_Request*)malloc(npatches * NUM_BOUNDARIES * sizeof(MPI_Request));

    Py_BEGIN_ALLOW_THREADS

    // Count outgoing particles for each patch
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        double *x = x_list[ipatch];
        double *y = y_list[ipatch];
        double *z = z_list[ipatch];
        npy_bool *is_dead = is_dead_list[ipatch];
        double xmin = xmin_list[ipatch];
        double xmax = xmax_list[ipatch];
        double ymin = ymin_list[ipatch];
        double ymax = ymax_list[ipatch];
        double zmin = zmin_list[ipatch];
        double zmax = zmax_list[ipatch];
        npy_intp npart = npart_list[ipatch];

        // Count particles going out of bounds
        npy_intp npart_out[NUM_BOUNDARIES] = {0};

        count_outgoing_particles(
            x, y, z, is_dead, xmin, xmax, ymin, ymax, zmin, zmax, npart,
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
            send_requests[i] = MPI_REQUEST_NULL;
            recv_requests[i] = MPI_REQUEST_NULL;
            continue;
        }

        int index = index_list[ipatch];
        int neighbor_index = neighbor_index_list[ipatch][ibound];
        // Tag based on patch index and boundary
        int send_tag = index*NUM_BOUNDARIES + ibound;
        int recv_tag = neighbor_index*NUM_BOUNDARIES + OPPOSITE_BOUNDARY[ibound];

        // Thread-multiple access to MPI functions
        MPI_Isend(&npart_outgoing[i], 1, MPI_LONG, neighbor_rank, send_tag,
                  comm, &send_requests[i]);
        MPI_Irecv(&npart_incoming[i], 1, MPI_LONG, neighbor_rank, recv_tag,
                  comm, &recv_requests[i]);
    }

    // Wait for all communications to complete
    MPI_Waitall(npatches*NUM_BOUNDARIES, send_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(npatches*NUM_BOUNDARIES, recv_requests, MPI_STATUSES_IGNORE);
    
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
    {"get_npart_to_extend_3d", get_npart_to_extend_3d, METH_VARARGS, "count the number of particles to be extended, and return the number of new particles"},
    {"fill_particles_from_boundary_3d", fill_particles_from_boundary_3d, METH_VARARGS, "fill particles from boundary using MPI"},
    {"fill_particles_from_boundary_3d_start", fill_particles_from_boundary_3d_start, METH_VARARGS, "start async fill particles from boundary using MPI"},
    {"fill_particles_from_boundary_3d_wait", fill_particles_from_boundary_3d_wait, METH_VARARGS, "wait and finalize async fill particles from boundary using MPI"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef syncparticlesmodule = {
    PyModuleDef_HEAD_INIT,
    "sync_particles_3d",
    NULL,
    -1,
    SyncParticlesMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_sync_particles_3d(void) {
    import_array();
    if (import_mpi4py() < 0) {
        return NULL;
    }
    return PyModule_Create(&syncparticlesmodule);
}
