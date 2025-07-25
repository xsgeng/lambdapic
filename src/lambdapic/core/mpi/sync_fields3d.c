#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <mpi.h>
#include "../utils/cutils.h"
#include <mpi4py/mpi4py.h>

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

static void fill_currents_buf(
    int ix_src, int nx, int NX,
    int iy_src, int ny, int NY,
    int iz_src, int nz, int NZ,
    double *jx, double *jy, double *jz, double *rho,
    double *buf
) {
    for (npy_intp ix = 0; ix < nx; ix++) {
        for (npy_intp iy = 0; iy < ny; iy++) {
            for (npy_intp iz = 0; iz < nz; iz++) {
                buf[ix*ny*nz + iy*nz + iz + 0*nx*ny*nz] = jx[INDEX3(ix_src+ix, iy_src+iy, iz_src+iz)];
                jx[INDEX3(ix_src+ix, iy_src+iy, iz_src+iz)] = 0;
            }
        }
    }
    for (npy_intp ix = 0; ix < nx; ix++) {
        for (npy_intp iy = 0; iy < ny; iy++) {
            for (npy_intp iz = 0; iz < nz; iz++) {
                buf[ix*ny*nz + iy*nz + iz + 1*nx*ny*nz] = jy[INDEX3(ix_src+ix, iy_src+iy, iz_src+iz)];
                jy[INDEX3(ix_src+ix, iy_src+iy, iz_src+iz)] = 0;
            }
        }
    }
    for (npy_intp ix = 0; ix < nx; ix++) {
        for (npy_intp iy = 0; iy < ny; iy++) {
            for (npy_intp iz = 0; iz < nz; iz++) {
                buf[ix*ny*nz + iy*nz + iz + 2*nx*ny*nz] = jz[INDEX3(ix_src+ix, iy_src+iy, iz_src+iz)];
                jz[INDEX3(ix_src+ix, iy_src+iy, iz_src+iz)] = 0;
            }
        }
    }
    for (npy_intp ix = 0; ix < nx; ix++) {
        for (npy_intp iy = 0; iy < ny; iy++) {
            for (npy_intp iz = 0; iz < nz; iz++) {
                buf[ix*ny*nz + iy*nz + iz + 3*nx*ny*nz] = rho[INDEX3(ix_src+ix, iy_src+iy, iz_src+iz)];
                rho[INDEX3(ix_src+ix, iy_src+iy, iz_src+iz)] = 0;
            }
        }
    }
}

static void sync_currents_buf(
    int ix_dst, int nx, int NX,
    int iy_dst, int ny, int NY,
    int iz_dst, int nz, int NZ,
    double *buf,
    double *jx, double *jy, double *jz, double *rho
) {
    for (npy_intp ix = 0; ix < nx; ix++) {
        for (npy_intp iy = 0; iy < ny; iy++) {
            for (npy_intp iz = 0; iz < nz; iz++) {
                jx[INDEX3(ix_dst+ix, iy_dst+iy, iz_dst+iz)] += buf[ix*ny*nz + iy*nz + iz + 0*nx*ny*nz];
            }
        }
    }
    for (npy_intp ix = 0; ix < nx; ix++) {
        for (npy_intp iy = 0; iy < ny; iy++) {
            for (npy_intp iz = 0; iz < nz; iz++) {
                jy[INDEX3(ix_dst+ix, iy_dst+iy, iz_dst+iz)] += buf[ix*ny*nz + iy*nz + iz + 1*nx*ny*nz];
            }
        }
    }
    for (npy_intp ix = 0; ix < nx; ix++) {
        for (npy_intp iy = 0; iy < ny; iy++) {
            for (npy_intp iz = 0; iz < nz; iz++) {
                jz[INDEX3(ix_dst+ix, iy_dst+iy, iz_dst+iz)] += buf[ix*ny*nz + iy*nz + iz + 2*nx*ny*nz];
            }
        }
    }
    for (npy_intp ix = 0; ix < nx; ix++) {
        for (npy_intp iy = 0; iy < ny; iy++) {
            for (npy_intp iz = 0; iz < nz; iz++) {
                rho[INDEX3(ix_dst+ix, iy_dst+iy, iz_dst+iz)] += buf[ix*ny*nz + iy*nz + iz + 3*nx*ny*nz];
            }
        }
    }
}

void get_boundary_currents(
    enum Boundary3D ibound,
    int nx, int ny, int nz, int ng,
    int *ix_dst, int *ix_src, int *nx_sync,
    int *iy_dst, int *iy_src, int *ny_sync,
    int *iz_dst, int *iz_src, int *nz_sync
) {
    switch (ibound) {
        case XMIN:
            *ix_dst = 0;     *ix_src = -ng; *nx_sync = ng;
            *iy_dst = 0;     *iy_src = 0;   *ny_sync = ny;
            *iz_dst = 0;     *iz_src = 0;   *nz_sync = nz;
            break;
        case XMAX:
            *ix_dst = nx-ng; *ix_src = nx;  *nx_sync = ng;
            *iy_dst = 0;     *iy_src = 0;   *ny_sync = ny;
            *iz_dst = 0;     *iz_src = 0;   *nz_sync = nz;
            break;
        case YMIN:
            *ix_dst = 0;     *ix_src = 0;   *nx_sync = nx;
            *iy_dst = 0;     *iy_src = -ng; *ny_sync = ng;
            *iz_dst = 0;     *iz_src = 0;   *nz_sync = nz;
            break;
        case YMAX:
            *ix_dst = 0;     *ix_src = 0;   *nx_sync = nx;
            *iy_dst = ny-ng; *iy_src = ny;  *ny_sync = ng;
            *iz_dst = 0;     *iz_src = 0;   *nz_sync = nz;
            break;
        case ZMIN:
            *ix_dst = 0;     *ix_src = 0;   *nx_sync = nx;
            *iy_dst = 0;     *iy_src = 0;   *ny_sync = ny;
            *iz_dst = 0;     *iz_src = -ng; *nz_sync = ng;
            break;
        case ZMAX:
            *ix_dst = 0;     *ix_src = 0;   *nx_sync = nx;
            *iy_dst = 0;     *iy_src = 0;   *ny_sync = ny;
            *iz_dst = nz-ng; *iz_src = nz;  *nz_sync = ng;
            break;
        case XMINYMIN:
            *ix_dst = 0;     *ix_src = -ng; *nx_sync = ng;
            *iy_dst = 0;     *iy_src = -ng; *ny_sync = ng;
            *iz_dst = 0;     *iz_src = 0;   *nz_sync = nz;
            break;
        case XMINYMAX:
            *ix_dst = 0;     *ix_src = -ng; *nx_sync = ng;
            *iy_dst = ny-ng; *iy_src = ny;  *ny_sync = ng;
            *iz_dst = 0;     *iz_src = 0;   *nz_sync = nz;
            break;
        case XMINZMIN:
            *ix_dst = 0;     *ix_src = -ng; *nx_sync = ng;
            *iy_dst = 0;     *iy_src = 0;   *ny_sync = ny;
            *iz_dst = 0;     *iz_src = -ng; *nz_sync = ng;
            break;
        case XMINZMAX:
            *ix_dst = 0;     *ix_src = -ng; *nx_sync = ng;
            *iy_dst = 0;     *iy_src = 0;   *ny_sync = ny;
            *iz_dst = nz-ng; *iz_src = nz;  *nz_sync = ng;
            break;
        case XMAXYMIN:
            *ix_dst = nx-ng; *ix_src = nx;  *nx_sync = ng;
            *iy_dst = 0;     *iy_src = -ng; *ny_sync = ng;
            *iz_dst = 0;     *iz_src = 0;   *nz_sync = nz;
            break;
        case XMAXYMAX:
            *ix_dst = nx-ng; *ix_src = nx;  *nx_sync = ng;
            *iy_dst = ny-ng; *iy_src = ny;  *ny_sync = ng;
            *iz_dst = 0;     *iz_src = 0;   *nz_sync = nz;
            break;
        case XMAXZMIN:
            *ix_dst = nx-ng; *ix_src = nx;  *nx_sync = ng;
            *iy_dst = 0;     *iy_src = 0;   *ny_sync = ny;
            *iz_dst = 0;     *iz_src = -ng; *nz_sync = ng;
            break;
        case XMAXZMAX:
            *ix_dst = nx-ng; *ix_src = nx;  *nx_sync = ng;
            *iy_dst = 0;     *iy_src = 0;   *ny_sync = ny;
            *iz_dst = nz-ng; *iz_src = nz;  *nz_sync = ng;
            break;
        case YMINZMIN:
            *ix_dst = 0;     *ix_src = 0;   *nx_sync = nx;
            *iy_dst = 0;     *iy_src = -ng; *ny_sync = ng;
            *iz_dst = 0;     *iz_src = -ng; *nz_sync = ng;
            break;
        case YMINZMAX:
            *ix_dst = 0;     *ix_src = 0;   *nx_sync = nx;
            *iy_dst = 0;     *iy_src = -ng; *ny_sync = ng;
            *iz_dst = nz-ng; *iz_src = nz;  *nz_sync = ng;
            break;
        case YMAXZMIN:
            *ix_dst = 0;     *ix_src = 0;   *nx_sync = nx;
            *iy_dst = ny-ng; *iy_src = ny;  *ny_sync = ng;
            *iz_dst = 0;     *iz_src = -ng; *nz_sync = ng;
            break;
        case YMAXZMAX:
            *ix_dst = 0;     *ix_src = 0;   *nx_sync = nx;
            *iy_dst = ny-ng; *iy_src = ny;  *ny_sync = ng;
            *iz_dst = nz-ng; *iz_src = nz;  *nz_sync = ng;
            break;
        case XMINYMINZMIN:
            *ix_dst = 0;     *ix_src = -ng; *nx_sync = ng;
            *iy_dst = 0;     *iy_src = -ng; *ny_sync = ng;
            *iz_dst = 0;     *iz_src = -ng; *nz_sync = ng;
            break;
        case XMINYMINZMAX:
            *ix_dst = 0;     *ix_src = -ng; *nx_sync = ng;
            *iy_dst = 0;     *iy_src = -ng; *ny_sync = ng;
            *iz_dst = nz-ng; *iz_src = nz;  *nz_sync = ng;
            break;
        case XMINYMAXZMIN:
            *ix_dst = 0;     *ix_src = -ng; *nx_sync = ng;
            *iy_dst = ny-ng; *iy_src = ny;  *ny_sync = ng;
            *iz_dst = 0;     *iz_src = -ng; *nz_sync = ng;
            break;
        case XMINYMAXZMAX:
            *ix_dst = 0;     *ix_src = -ng; *nx_sync = ng;
            *iy_dst = ny-ng; *iy_src = ny;  *ny_sync = ng;
            *iz_dst = nz-ng; *iz_src = nz;  *nz_sync = ng;
            break;
        case XMAXYMINZMIN:
            *ix_dst = nx-ng; *ix_src = nx;  *nx_sync = ng;
            *iy_dst = 0;     *iy_src = -ng; *ny_sync = ng;
            *iz_dst = 0;     *iz_src = -ng; *nz_sync = ng;
            break;
        case XMAXYMINZMAX:
            *ix_dst = nx-ng; *ix_src = nx;  *nx_sync = ng;
            *iy_dst = 0;     *iy_src = -ng; *ny_sync = ng;
            *iz_dst = nz-ng; *iz_src = nz;  *nz_sync = ng;
            break;
        case XMAXYMAXZMIN:
            *ix_dst = nx-ng; *ix_src = nx;  *nx_sync = ng;
            *iy_dst = ny-ng; *iy_src = ny;  *ny_sync = ng;
            *iz_dst = 0;     *iz_src = -ng; *nz_sync = ng;
            break;
        case XMAXYMAXZMAX:
            *ix_dst = nx-ng; *ix_src = nx;  *nx_sync = ng;
            *iy_dst = ny-ng; *iy_src = ny;  *ny_sync = ng;
            *iz_dst = nz-ng; *iz_src = nz;  *nz_sync = ng;
            break;
        case NUM_BOUNDARIES:
            break;
    }
}

void get_boundary_guards(
    enum Boundary3D ibound,
    int nx, int ny, int nz, int ng,
    int *ix_dst, int *ix_src, int *nx_sync,
    int *iy_dst, int *iy_src, int *ny_sync,
    int *iz_dst, int *iz_src, int *nz_sync
) {
    switch (ibound) {
        case XMIN:
            *ix_dst = -ng; *ix_src = 0;     *nx_sync = ng;
            *iy_dst = 0;   *iy_src = 0;     *ny_sync = ny;
            *iz_dst = 0;   *iz_src = 0;     *nz_sync = nz;
            break;
        case XMAX:
            *ix_dst = nx;  *ix_src = nx-ng; *nx_sync = ng;
            *iy_dst = 0;   *iy_src = 0;     *ny_sync = ny;
            *iz_dst = 0;   *iz_src = 0;     *nz_sync = nz;
            break;
        case YMIN:
            *ix_dst = 0;   *ix_src = 0;     *nx_sync = nx;
            *iy_dst = -ng; *iy_src = 0;     *ny_sync = ng;
            *iz_dst = 0;   *iz_src = 0;     *nz_sync = nz;
            break;
        case YMAX:
            *ix_dst = 0;   *ix_src = 0;     *nx_sync = nx;
            *iy_dst = ny;  *iy_src = ny-ng; *ny_sync = ng;
            *iz_dst = 0;   *iz_src = 0;     *nz_sync = nz;
            break;
        case ZMIN:
            *ix_dst = 0;   *ix_src = 0;     *nx_sync = nx;
            *iy_dst = 0;   *iy_src = 0;     *ny_sync = ny;
            *iz_dst = -ng; *iz_src = 0;     *nz_sync = ng;
            break;
        case ZMAX:
            *ix_dst = 0;   *ix_src = 0;     *nx_sync = nx;
            *iy_dst = 0;   *iy_src = 0;     *ny_sync = ny;
            *iz_dst = nz;  *iz_src = nz-ng; *nz_sync = ng;
            break;
        case XMINYMIN:
            *ix_dst = -ng; *ix_src = 0;     *nx_sync = ng;
            *iy_dst = -ng; *iy_src = 0;     *ny_sync = ng;
            *iz_dst = 0;   *iz_src = 0;     *nz_sync = nz;
            break;
        case XMINYMAX:
            *ix_dst = -ng; *ix_src = 0;     *nx_sync = ng;
            *iy_dst = ny;  *iy_src = ny-ng; *ny_sync = ng;
            *iz_dst = 0;   *iz_src = 0;     *nz_sync = nz;
            break;
        case XMINZMIN:
            *ix_dst = -ng; *ix_src = 0;     *nx_sync = ng;
            *iy_dst = 0;   *iy_src = 0;     *ny_sync = ny;
            *iz_dst = -ng; *iz_src = 0;     *nz_sync = ng;
            break;
        case XMINZMAX:
            *ix_dst = -ng; *ix_src = 0;     *nx_sync = ng;
            *iy_dst = 0;   *iy_src = 0;     *ny_sync = ny;
            *iz_dst = nz;  *iz_src = nz-ng; *nz_sync = ng;
            break;
        case XMAXYMIN:
            *ix_dst = nx;  *ix_src = nx-ng; *nx_sync = ng;
            *iy_dst = -ng; *iy_src = 0;     *ny_sync = ng;
            *iz_dst = 0;   *iz_src = 0;     *nz_sync = nz;
            break;
        case XMAXYMAX:
            *ix_dst = nx;  *ix_src = nx-ng; *nx_sync = ng;
            *iy_dst = ny;  *iy_src = ny-ng; *ny_sync = ng;
            *iz_dst = 0;   *iz_src = 0;     *nz_sync = nz;
            break;
        case XMAXZMIN:
            *ix_dst = nx;  *ix_src = nx-ng; *nx_sync = ng;
            *iy_dst = 0;   *iy_src = 0;     *ny_sync = ny;
            *iz_dst = -ng; *iz_src = 0;     *nz_sync = ng;
            break;
        case XMAXZMAX:
            *ix_dst = nx;  *ix_src = nx-ng; *nx_sync = ng;
            *iy_dst = 0;   *iy_src = 0;     *ny_sync = ny;
            *iz_dst = nz;  *iz_src = nz-ng; *nz_sync = ng;
            break;
        case YMINZMIN:
            *ix_dst = 0;   *ix_src = 0;     *nx_sync = nx;
            *iy_dst = -ng; *iy_src = 0;     *ny_sync = ng;
            *iz_dst = -ng; *iz_src = 0;     *nz_sync = ng;
            break;
        case YMINZMAX:
            *ix_dst = 0;   *ix_src = 0;     *nx_sync = nx;
            *iy_dst = -ng; *iy_src = 0;     *ny_sync = ng;
            *iz_dst = nz;  *iz_src = nz-ng; *nz_sync = ng;
            break;
        case YMAXZMIN:
            *ix_dst = 0;   *ix_src = 0;     *nx_sync = nx;
            *iy_dst = ny;  *iy_src = ny-ng; *ny_sync = ng;
            *iz_dst = -ng; *iz_src = 0;     *nz_sync = ng;
            break;
        case YMAXZMAX:
            *ix_dst = 0;   *ix_src = 0;     *nx_sync = nx;
            *iy_dst = ny;  *iy_src = ny-ng; *ny_sync = ng;
            *iz_dst = nz;  *iz_src = nz-ng; *nz_sync = ng;
            break;
        case XMINYMINZMIN:
            *ix_dst = -ng; *ix_src = 0;     *nx_sync = ng;
            *iy_dst = -ng; *iy_src = 0;     *ny_sync = ng;
            *iz_dst = -ng; *iz_src = 0;     *nz_sync = ng;
            break;
        case XMINYMINZMAX:
            *ix_dst = -ng; *ix_src = 0;     *nx_sync = ng;
            *iy_dst = -ng; *iy_src = 0;     *ny_sync = ng;
            *iz_dst = nz;  *iz_src = nz-ng; *nz_sync = ng;
            break;
        case XMINYMAXZMIN:
            *ix_dst = -ng; *ix_src = 0;     *nx_sync = ng;
            *iy_dst = ny;  *iy_src = ny-ng; *ny_sync = ng;
            *iz_dst = -ng; *iz_src = 0;     *nz_sync = ng;
            break;
        case XMINYMAXZMAX:
            *ix_dst = -ng; *ix_src = 0;     *nx_sync = ng;
            *iy_dst = ny;  *iy_src = ny-ng; *ny_sync = ng;
            *iz_dst = nz;  *iz_src = nz-ng; *nz_sync = ng;
            break;
        case XMAXYMINZMIN:
            *ix_dst = nx;  *ix_src = nx-ng; *nx_sync = ng;
            *iy_dst = -ng; *iy_src = 0;     *ny_sync = ng;
            *iz_dst = -ng; *iz_src = 0;     *nz_sync = ng;
            break;
        case XMAXYMINZMAX:
            *ix_dst = nx;  *ix_src = nx-ng; *nx_sync = ng;
            *iy_dst = -ng; *iy_src = 0;     *ny_sync = ng;
            *iz_dst = nz;  *iz_src = nz-ng; *nz_sync = ng;
            break;
        case XMAXYMAXZMIN:
            *ix_dst = nx;  *ix_src = nx-ng; *nx_sync = ng;
            *iy_dst = ny;  *iy_src = ny-ng; *ny_sync = ng;
            *iz_dst = -ng; *iz_src = 0;     *nz_sync = ng;
            break;
        case XMAXYMAXZMAX:
            *ix_dst = nx;  *ix_src = nx-ng; *nx_sync = ng;
            *iy_dst = ny;  *iy_src = ny-ng; *ny_sync = ng;
            *iz_dst = nz;  *iz_src = nz-ng; *nz_sync = ng;
            break;
        case NUM_BOUNDARIES:
            break;
    }
}

void init_boundary_types(int nx, int ny, int nz, int ng, MPI_Datatype *mpi_types_bound, MPI_Datatype *mpi_types_guard) {
    int size[3] = {nx, ny, nz};
    int subsize[3] = {ng, ng, ng};
    int start_bound[3] = {0, 0, 0};
    int start_guard[3] = {0, 0, 0};

    for (int ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
        switch (ibound) {
            case XMIN:
                subsize[0]     = ng;      subsize[1]     = ny-2*ng; subsize[2]     = nz-2*ng;
                start_bound[0] = 0;       start_bound[1] = 0;       start_bound[2] = 0;
                start_guard[0] = nx-ng;   start_guard[1] = 0;       start_guard[2] = 0;
                break;
            case XMAX:
                subsize[0]     = ng;      subsize[1]     = ny-2*ng; subsize[2]     = nz-2*ng;
                start_bound[0] = nx-3*ng; start_bound[1] = 0;       start_bound[2] = 0;
                start_guard[0] = nx-2*ng; start_guard[1] = 0;       start_guard[2] = 0;
                break;
            case YMIN:
                subsize[0]     = nx-2*ng; subsize[1]     = ng;      subsize[2]     = nz-2*ng;
                start_bound[0] = 0;       start_bound[1] = 0;       start_bound[2] = 0;
                start_guard[0] = 0;       start_guard[1] = ny-ng;   start_guard[2] = 0;
                break;
            case YMAX:
                subsize[0]     = nx-2*ng; subsize[1]     = ng;      subsize[2]     = nz-2*ng;
                start_bound[0] = 0;       start_bound[1] = ny-3*ng; start_bound[2] = 0;
                start_guard[0] = 0;       start_guard[1] = ny-2*ng; start_guard[2] = 0;
                break;
            case ZMIN:
                subsize[0]     = nx-2*ng; subsize[1]     = ny-2*ng; subsize[2]     = ng;
                start_bound[0] = 0;       start_bound[1] = 0;       start_bound[2] = 0;
                start_guard[0] = 0;       start_guard[1] = 0;       start_guard[2] = nz-ng;
                break;
            case ZMAX:
                subsize[0]     = nx-2*ng; subsize[1]     = ny-2*ng; subsize[2]     = ng;
                start_bound[0] = 0;       start_bound[1] = 0;       start_bound[2] = nz-3*ng;
                start_guard[0] = 0;       start_guard[1] = 0;       start_guard[2] = nz-2*ng;
                break;
            case XMINYMIN:
                subsize[0]     = ng;      subsize[1]     = ng;      subsize[2]     = nz-2*ng;
                start_bound[0] = 0;       start_bound[1] = 0;       start_bound[2] = 0;
                start_guard[0] = nx-ng;   start_guard[1] = ny-ng;   start_guard[2] = 0;
                break;
            case XMINYMAX:
                subsize[0]     = ng;      subsize[1]     = ng;      subsize[2]     = nz-2*ng;
                start_bound[0] = 0;       start_bound[1] = ny-3*ng; start_bound[2] = 0;
                start_guard[0] = nx-ng;   start_guard[1] = ny-2*ng; start_guard[2] = 0;
                break;
            case XMINZMIN:
                subsize[0]     = ng;      subsize[1]     = ny-2*ng; subsize[2]     = ng;
                start_bound[0] = 0;       start_bound[1] = 0;       start_bound[2] = 0;
                start_guard[0] = nx-ng;   start_guard[1] = 0;       start_guard[2] = nz-ng;
                break;
            case XMINZMAX:
                subsize[0]     = ng;      subsize[1]     = ny-2*ng; subsize[2]     = ng;
                start_bound[0] = 0;       start_bound[1] = 0;       start_bound[2] = nz-3*ng;
                start_guard[0] = nx-ng;   start_guard[1] = 0;       start_guard[2] = nz-2*ng;
                break;
            case XMAXYMIN:
                subsize[0]     = ng;      subsize[1]     = ng;      subsize[2]     = nz-2*ng;
                start_bound[0] = nx-3*ng; start_bound[1] = 0;       start_bound[2] = 0;
                start_guard[0] = nx-2*ng; start_guard[1] = ny-ng;   start_guard[2] = 0;
                break;
            case XMAXYMAX:
                subsize[0]     = ng;      subsize[1]     = ng;      subsize[2]     = nz-2*ng;
                start_bound[0] = nx-3*ng; start_bound[1] = ny-3*ng; start_bound[2] = 0;
                start_guard[0] = nx-2*ng; start_guard[1] = ny-2*ng; start_guard[2] = 0;
                break;
            case XMAXZMIN:
                subsize[0]     = ng;      subsize[1]     = ny-2*ng; subsize[2]     = ng;
                start_bound[0] = nx-3*ng; start_bound[1] = 0;       start_bound[2] = 0;
                start_guard[0] = nx-2*ng; start_guard[1] = 0;       start_guard[2] = nz-ng;
                break;
            case XMAXZMAX:
                subsize[0]     = ng;      subsize[1]     = ny-2*ng; subsize[2]     = ng;
                start_bound[0] = nx-3*ng; start_bound[1] = 0;       start_bound[2] = nz-3*ng;
                start_guard[0] = nx-2*ng; start_guard[1] = 0;       start_guard[2] = nz-2*ng;
                break;
            case YMINZMIN:
                subsize[0]     = nx-2*ng; subsize[1]     = ng;      subsize[2]     = ng;
                start_bound[0] = 0;       start_bound[1] = 0;       start_bound[2] = 0;
                start_guard[0] = 0;       start_guard[1] = ny-ng;   start_guard[2] = nz-ng;
                break;
            case YMINZMAX:
                subsize[0]     = nx-2*ng; subsize[1]     = ng;      subsize[2]     = ng;
                start_bound[0] = 0;       start_bound[1] = 0;       start_bound[2] = nz-3*ng;
                start_guard[0] = 0;       start_guard[1] = ny-ng;   start_guard[2] = nz-2*ng;
                break;
            case YMAXZMIN:
                subsize[0]     = nx-2*ng; subsize[1]     = ng;      subsize[2]     = ng;
                start_bound[0] = 0;       start_bound[1] = ny-3*ng; start_bound[2] = 0;
                start_guard[0] = 0;       start_guard[1] = ny-2*ng; start_guard[2] = nz-ng;
                break;
            case YMAXZMAX:
                subsize[0]     = nx-2*ng; subsize[1]     = ng;      subsize[2]     = ng;
                start_bound[0] = 0;       start_bound[1] = ny-3*ng; start_bound[2] = nz-3*ng;
                start_guard[0] = 0;       start_guard[1] = ny-2*ng; start_guard[2] = nz-2*ng;
                break;
            case XMINYMINZMIN:
                subsize[0]     = ng;      subsize[1]     = ng;      subsize[2]     = ng;
                start_bound[0] = 0;       start_bound[1] = 0;       start_bound[2] = 0;
                start_guard[0] = nx-ng;   start_guard[1] = ny-ng;   start_guard[2] = nz-ng;
                break;
            case XMINYMINZMAX:
                subsize[0]     = ng;      subsize[1]     = ng;      subsize[2]     = ng;
                start_bound[0] = 0;       start_bound[1] = 0;       start_bound[2] = nz-3*ng;
                start_guard[0] = nx-ng;   start_guard[1] = ny-ng;   start_guard[2] = nz-2*ng;
                break;
            case XMINYMAXZMIN:
                subsize[0]     = ng;      subsize[1]     = ng;      subsize[2]     = ng;
                start_bound[0] = 0;       start_bound[1] = ny-3*ng; start_bound[2] = 0;
                start_guard[0] = nx-ng;   start_guard[1] = ny-2*ng; start_guard[2] = nz-ng;
                break;
            case XMINYMAXZMAX:
                subsize[0]     = ng;      subsize[1]     = ng;      subsize[2]     = ng;
                start_bound[0] = 0;       start_bound[1] = ny-3*ng; start_bound[2] = nz-3*ng;
                start_guard[0] = nx-ng;   start_guard[1] = ny-2*ng; start_guard[2] = nz-2*ng;
                break;
            case XMAXYMINZMIN:
                subsize[0]     = ng;      subsize[1]     = ng;      subsize[2]     = ng;
                start_bound[0] = nx-3*ng; start_bound[1] = 0;       start_bound[2] = 0;
                start_guard[0] = nx-2*ng; start_guard[1] = ny-ng;   start_guard[2] = nz-ng;
                break;
            case XMAXYMINZMAX:
                subsize[0]     = ng;      subsize[1]     = ng;      subsize[2]     = ng;
                start_bound[0] = nx-3*ng; start_bound[1] = 0;       start_bound[2] = nz-3*ng;
                start_guard[0] = nx-2*ng; start_guard[1] = ny-ng;   start_guard[2] = nz-2*ng;
                break;
            case XMAXYMAXZMIN:
                subsize[0]     = ng;      subsize[1]     = ng;      subsize[2]     = ng;
                start_bound[0] = nx-3*ng; start_bound[1] = ny-3*ng; start_bound[2] = 0;
                start_guard[0] = nx-2*ng; start_guard[1] = ny-2*ng; start_guard[2] = nz-ng;
                break;
            case XMAXYMAXZMAX:
                subsize[0]     = ng;      subsize[1]     = ng;      subsize[2]     = ng;
                start_bound[0] = nx-3*ng; start_bound[1] = ny-3*ng; start_bound[2] = nz-3*ng;
                start_guard[0] = nx-2*ng; start_guard[1] = ny-2*ng; start_guard[2] = nz-2*ng;
                break;
            case NUM_BOUNDARIES:
                break;
        }
        MPI_Type_create_subarray(3, size, subsize, start_bound, MPI_ORDER_C, MPI_DOUBLE, &mpi_types_bound[ibound]);
        MPI_Type_create_subarray(3, size, subsize, start_guard, MPI_ORDER_C, MPI_DOUBLE, &mpi_types_guard[ibound]);
        MPI_Type_commit(&mpi_types_bound[ibound]);
        MPI_Type_commit(&mpi_types_guard[ibound]);
    }
}

void free_boundary_types(MPI_Datatype *mpi_types_bound, MPI_Datatype *mpi_types_guard) {
    for (int ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
        MPI_Type_free(&mpi_types_bound[ibound]);
        MPI_Type_free(&mpi_types_guard[ibound]);
    }
}

static PyObject* sync_currents_3d(PyObject* self, PyObject* args) {
    PyObject *fields_list, *patches_list;
    PyObject* comm_py;
    npy_intp npatches, nx, ny, nz, ng;

    if (!PyArg_ParseTuple(args, "OOOnnnnn", 
        &fields_list, &patches_list,
        &comm_py,
        &npatches, &nx, &ny, &nz, &ng)) {
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

    npy_intp NX = nx+2*ng;
    npy_intp NY = ny+2*ng;
    npy_intp NZ = nz+2*ng;
    AUTOFREE double **jx = get_attr_array_double(fields_list, npatches, "jx");
    AUTOFREE double **jy = get_attr_array_double(fields_list, npatches, "jy"); 
    AUTOFREE double **jz = get_attr_array_double(fields_list, npatches, "jz");
    AUTOFREE double **rho = get_attr_array_double(fields_list, npatches, "rho");

    AUTOFREE npy_intp **neighbor_index_list = get_attr_array_int(patches_list, npatches, "neighbor_index");
    AUTOFREE npy_intp **neighbor_rank_list = get_attr_array_int(patches_list, npatches, "neighbor_rank");
    AUTOFREE npy_intp *index_list = get_attr_int(patches_list, npatches, "index");

    // Allocate MPI request arrays
    AUTOFREE MPI_Request *sendrecv_requests = (MPI_Request*)malloc(npatches * NUM_BOUNDARIES * sizeof(MPI_Request));

    // buffers
    AUTOFREE double **buf = (double**) malloc(npatches * NUM_BOUNDARIES * sizeof(double*));

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for collapse(2)
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        for (int ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            const npy_intp* neighbor_index = neighbor_index_list[ipatch];
            const npy_intp index = index_list[ipatch];
            int neighbor_rank = neighbor_rank_list[ipatch][ibound];
            if (neighbor_rank < 0) {
                buf[ipatch*NUM_BOUNDARIES + ibound] = NULL;
                continue;
            }
            int ix_src, ix_dst, nx_sync, 
                iy_src, iy_dst, ny_sync,
                iz_src, iz_dst, nz_sync;
            get_boundary_currents(
                ibound, 
                nx, ny, nz, ng, 
                &ix_dst, &ix_src, &nx_sync, 
                &iy_dst, &iy_src, &ny_sync,
                &iz_dst, &iz_src, &nz_sync
            );
            // store attrs into one buffer
            buf[ipatch*NUM_BOUNDARIES + ibound] = (double*) malloc(4*sizeof(double) * nx_sync * ny_sync * nz_sync);
            int send_tag = index*NUM_BOUNDARIES + ibound;
            int recv_tag = neighbor_index[ibound]*NUM_BOUNDARIES + OPPOSITE_BOUNDARY[ibound];
            fill_currents_buf(
                ix_src, nx_sync, NX,
                iy_src, ny_sync, NY,
                iz_src, nz_sync, NZ,
                jx[ipatch], jy[ipatch], jz[ipatch], rho[ipatch], 
                buf[ipatch*NUM_BOUNDARIES + ibound]
            );
            MPI_Isendrecv_replace(
                buf[ipatch*NUM_BOUNDARIES + ibound], 4*nx_sync*ny_sync*nz_sync, MPI_DOUBLE, 
                neighbor_rank, send_tag, 
                neighbor_rank, recv_tag, 
                comm, &sendrecv_requests[ipatch*NUM_BOUNDARIES + ibound]
            );
        }
    }

    // read buffers
    #pragma omp parallel for collapse(2)
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        for (int ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
            int neighbor_rank = neighbor_rank_list[ipatch][ibound];
            if (neighbor_rank < 0) {
                continue;
            }
            int ix_src, ix_dst, nx_sync, 
                iy_src, iy_dst, ny_sync,
                iz_src, iz_dst, nz_sync;
            get_boundary_currents(
                ibound, 
                nx, ny, nz, ng, 
                &ix_dst, &ix_src, &nx_sync, 
                &iy_dst, &iy_src, &ny_sync,
                &iz_dst, &iz_src, &nz_sync
            );
            MPI_Wait(&sendrecv_requests[ipatch*NUM_BOUNDARIES + ibound], MPI_STATUS_IGNORE);
            sync_currents_buf(
                ix_dst, nx_sync, NX,
                iy_dst, ny_sync, NY,
                iz_dst, nz_sync, NZ,
                buf[ipatch*NUM_BOUNDARIES + ibound],
                jx[ipatch], jy[ipatch], jz[ipatch], rho[ipatch]
            );
            free(buf[ipatch*NUM_BOUNDARIES + ibound]);
        }
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

static PyObject* sync_guard_fields_3d(PyObject* self, PyObject* args) {
    PyObject *fields_list, *patches_list, *attrs;
    PyObject* comm_py;
    npy_intp npatches, nx, ny, nz, ng;

    if (!PyArg_ParseTuple(args, "OOOOnnnnn", 
        &fields_list, &patches_list, 
        &comm_py,
        &attrs,
        &npatches, &nx, &ny, &nz, &ng)) {
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

    npy_intp NX = nx+2*ng;
    npy_intp NY = ny+2*ng;
    npy_intp NZ = nz+2*ng;

    int nattrs = PyList_Size(attrs);
    AUTOFREE double ***attrs_list = malloc(nattrs * sizeof(double**));
    for (int i = 0; i < nattrs; i++) {
        attrs_list[i] = get_attr_array_double(fields_list, npatches, PyUnicode_AsUTF8(PyList_GetItem(attrs, i)));
    }

    AUTOFREE npy_intp **neighbor_index_list = get_attr_array_int(patches_list, npatches, "neighbor_index");
    AUTOFREE npy_intp **neighbor_rank_list = get_attr_array_int(patches_list, npatches, "neighbor_rank");
    AUTOFREE npy_intp *index_list = get_attr_int(patches_list, npatches, "index");

    // Initialize boundary types for MPI communication
    MPI_Datatype mpi_types_bound[NUM_BOUNDARIES];
    MPI_Datatype mpi_types_guard[NUM_BOUNDARIES];
    init_boundary_types(NX, NY, NZ, ng, mpi_types_bound, mpi_types_guard);

    // Allocate MPI request arrays - 2 requests per boundary (send and recv)
    AUTOFREE MPI_Request *send_requests = (MPI_Request*)malloc(npatches * NUM_BOUNDARIES * nattrs * sizeof(MPI_Request));
    AUTOFREE MPI_Request *recv_requests = (MPI_Request*)malloc(npatches * NUM_BOUNDARIES * nattrs * sizeof(MPI_Request));

    Py_BEGIN_ALLOW_THREADS
    // Post non-blocking receives and sends for all boundaries
    #pragma omp parallel for collapse(2)
    for (int iattr = 0; iattr < nattrs; iattr++) {
        for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
            const npy_intp index = index_list[ipatch];
            const npy_intp* neighbor_index = neighbor_index_list[ipatch];                                                 
            for (int ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
                int neighbor_rank = neighbor_rank_list[ipatch][ibound];
                if (neighbor_rank < 0) {
                    send_requests[ipatch * NUM_BOUNDARIES * nattrs + ibound * nattrs + iattr] = MPI_REQUEST_NULL;
                    recv_requests[ipatch * NUM_BOUNDARIES * nattrs + ibound * nattrs + iattr] = MPI_REQUEST_NULL;
                    continue;
                }
                int send_tag = index * NUM_BOUNDARIES * nattrs + ibound * nattrs + iattr;
                int recv_tag = neighbor_index[ibound] * NUM_BOUNDARIES * nattrs + OPPOSITE_BOUNDARY[ibound] * nattrs + iattr;
                // Post receive for guard cells
                MPI_Isend(
                    attrs_list[iattr][ipatch], 1, mpi_types_bound[ibound], neighbor_rank, send_tag,
                    comm, &send_requests[ipatch * NUM_BOUNDARIES * nattrs + ibound * nattrs + iattr]
                );
                MPI_Irecv(
                    attrs_list[iattr][ipatch], 1, mpi_types_guard[ibound], neighbor_rank, recv_tag, 
                    comm, &recv_requests[ipatch * NUM_BOUNDARIES * nattrs + ibound * nattrs + iattr]
                );
            }
        }
    }
    #pragma omp parallel for collapse(3)
    for (int iattr = 0; iattr < nattrs; iattr++) {
        for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
            for (int ibound = 0; ibound < NUM_BOUNDARIES; ibound++) {
                int neighbor_rank = neighbor_rank_list[ipatch][ibound];
                if (neighbor_rank < 0) {
                    continue;
                }
                MPI_Wait(&send_requests[ipatch * NUM_BOUNDARIES * nattrs + ibound * nattrs + iattr], MPI_STATUS_IGNORE);
                MPI_Wait(&recv_requests[ipatch * NUM_BOUNDARIES * nattrs + ibound * nattrs + iattr], MPI_STATUS_IGNORE);
            }
        }
    }
    Py_END_ALLOW_THREADS

    free_boundary_types(mpi_types_bound, mpi_types_guard);
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
    if (import_mpi4py() < 0) {
        return NULL;
    }
    return PyModule_Create(&module);
}
