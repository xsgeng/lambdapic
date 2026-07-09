#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>

#include "../../utils/cutils.h"
#include "../../current/current_deposit.h"


__attribute__((always_inline)) inline static void boris(
    double* restrict ux, double* restrict uy, double* restrict uz, double* restrict inv_gamma,
    double Ex, double Ey, double Ez,
    double Bx, double By, double Bz,
    double efactor, double bfactor
) {

    // E field half acceleration
    double ux_minus = *ux + efactor * Ex;
    double uy_minus = *uy + efactor * Ey;
    double uz_minus = *uz + efactor * Ez;

    // B field rotation
    *inv_gamma = 1.0 / sqrt(1 + ux_minus * ux_minus + uy_minus * uy_minus + uz_minus * uz_minus);
    double Tx = bfactor * Bx * (*inv_gamma);
    double Ty = bfactor * By * (*inv_gamma);
    double Tz = bfactor * Bz * (*inv_gamma);

    double ux_prime = ux_minus + uy_minus * Tz - uz_minus * Ty;
    double uy_prime = uy_minus + uz_minus * Tx - ux_minus * Tz;
    double uz_prime = uz_minus + ux_minus * Ty - uy_minus * Tx;

    double Tfactor = 2.0 / (1 + Tx * Tx + Ty * Ty + Tz * Tz);
    double Sx = Tfactor * Tx;
    double Sy = Tfactor * Ty;
    double Sz = Tfactor * Tz;

    double ux_plus = ux_minus + uy_prime * Sz - uz_prime * Sy;
    double uy_plus = uy_minus + uz_prime * Sx - ux_prime * Sz;
    double uz_plus = uz_minus + ux_prime * Sy - uy_prime * Sx;

    // E field half acceleration
    *ux = ux_plus + efactor * Ex;
    *uy = uy_plus + efactor * Ey;
    *uz = uz_plus + efactor * Ez;
    *inv_gamma = 1.0 / sqrt(1 + (*ux) * (*ux) + (*uy) * (*uy) + (*uz) * (*uz));
}

__attribute__((always_inline)) inline static void push_position_3d(
    double* restrict x, double* restrict y, double* restrict z,
    double ux, double uy, double uz,
    double inv_gamma,
    double cdt
) {
    *x += cdt * inv_gamma * ux;
    *y += cdt * inv_gamma * uy;
    *z += cdt * inv_gamma * uz;
}


__attribute__((always_inline)) inline static void get_gx(double delta, double* gx) {
    double delta2 = delta * delta;
    gx[0] = 0.5 * (0.25 + delta2 + delta);
    gx[1] = 0.75 - delta2;
    gx[2] = 0.5 * (0.25 + delta2 - delta);
}

// Slow path: uses INDEX3 with periodic boundary checks
__attribute__((always_inline)) inline static double interp_field_safe_3d(
    double* restrict field,
    double* restrict facx, double* restrict facy, double* restrict facz,
    npy_intp ix, npy_intp iy, npy_intp iz,
    npy_intp nx, npy_intp ny, npy_intp nz
) {
    double field_part =
          facz[0] * (facy[0] * (facx[0] * field[INDEX3(ix-1, iy-1, iz-1)]
        +                       facx[1] * field[INDEX3(ix  , iy-1, iz-1)]
        +                       facx[2] * field[INDEX3(ix+1, iy-1, iz-1)])
        +            facy[1] * (facx[0] * field[INDEX3(ix-1, iy  , iz-1)]
        +                       facx[1] * field[INDEX3(ix  , iy  , iz-1)]
        +                       facx[2] * field[INDEX3(ix+1, iy  , iz-1)])
        +            facy[2] * (facx[0] * field[INDEX3(ix-1, iy+1, iz-1)]
        +                       facx[1] * field[INDEX3(ix  , iy+1, iz-1)]
        +                       facx[2] * field[INDEX3(ix+1, iy+1, iz-1)]))
        + facz[1] * (facy[0] * (facx[0] * field[INDEX3(ix-1, iy-1, iz  )]
        +                       facx[1] * field[INDEX3(ix  , iy-1, iz  )]
        +                       facx[2] * field[INDEX3(ix+1, iy-1, iz  )])
        +            facy[1] * (facx[0] * field[INDEX3(ix-1, iy  , iz  )]
        +                       facx[1] * field[INDEX3(ix  , iy  , iz  )]
        +                       facx[2] * field[INDEX3(ix+1, iy  , iz  )])
        +            facy[2] * (facx[0] * field[INDEX3(ix-1, iy+1, iz  )]
        +                       facx[1] * field[INDEX3(ix  , iy+1, iz  )]
        +                       facx[2] * field[INDEX3(ix+1, iy+1, iz  )]))
        + facz[2] * (facy[0] * (facx[0] * field[INDEX3(ix-1, iy-1, iz+1)]
        +                       facx[1] * field[INDEX3(ix  , iy-1, iz+1)]
        +                       facx[2] * field[INDEX3(ix+1, iy-1, iz+1)])
        +            facy[1] * (facx[0] * field[INDEX3(ix-1, iy  , iz+1)]
        +                       facx[1] * field[INDEX3(ix  , iy  , iz+1)]
        +                       facx[2] * field[INDEX3(ix+1, iy  , iz+1)])
        +            facy[2] * (facx[0] * field[INDEX3(ix-1, iy+1, iz+1)]
        +                       facx[1] * field[INDEX3(ix  , iy+1, iz+1)]
        +                       facx[2] * field[INDEX3(ix+1, iy+1, iz+1)]));
    return field_part;
}

__attribute__((always_inline)) inline static double interp_field_fast_3d(
    double* restrict field,
    double* restrict facx, double* restrict facy, double* restrict facz,
    int ix, int iy, int iz,
    int ny, int nz
) {
    int ny_nz = ny * nz;
    int im1 = (ix - 1) * ny_nz, i0 = ix * ny_nz, ip1 = (ix + 1) * ny_nz;
    int jm1 = (iy - 1) * nz, j0 = iy * nz, jp1 = (iy + 1) * nz;
    int km1 = iz - 1, k0 = iz, kp1 = iz + 1;
    return facz[0] * (facy[0] * (facx[0] * field[im1 + jm1 + km1]
                              + facx[1] * field[i0  + jm1 + km1]
                              + facx[2] * field[ip1 + jm1 + km1])
                    + facy[1] * (facx[0] * field[im1 + j0  + km1]
                              + facx[1] * field[i0  + j0  + km1]
                              + facx[2] * field[ip1 + j0  + km1])
                    + facy[2] * (facx[0] * field[im1 + jp1 + km1]
                              + facx[1] * field[i0  + jp1 + km1]
                              + facx[2] * field[ip1 + jp1 + km1]))
         + facz[1] * (facy[0] * (facx[0] * field[im1 + jm1 + k0]
                              + facx[1] * field[i0  + jm1 + k0]
                              + facx[2] * field[ip1 + jm1 + k0])
                    + facy[1] * (facx[0] * field[im1 + j0  + k0]
                              + facx[1] * field[i0  + j0  + k0]
                              + facx[2] * field[ip1 + j0  + k0])
                    + facy[2] * (facx[0] * field[im1 + jp1 + k0]
                              + facx[1] * field[i0  + jp1 + k0]
                              + facx[2] * field[ip1 + jp1 + k0]))
         + facz[2] * (facy[0] * (facx[0] * field[im1 + jm1 + kp1]
                              + facx[1] * field[i0  + jm1 + kp1]
                              + facx[2] * field[ip1 + jm1 + kp1])
                    + facy[1] * (facx[0] * field[im1 + j0  + kp1]
                              + facx[1] * field[i0  + j0  + kp1]
                              + facx[2] * field[ip1 + j0  + kp1])
                    + facy[2] * (facx[0] * field[im1 + jp1 + kp1]
                              + facx[1] * field[i0  + jp1 + kp1]
                              + facx[2] * field[ip1 + jp1 + kp1]));
}

__attribute__((always_inline)) inline static void interpolation_3d(
    double x, double y, double z,
    double* restrict ex_part, double* restrict ey_part, double* restrict ez_part,
    double* restrict bx_part, double* restrict by_part, double* restrict bz_part,
    double* restrict ex, double* restrict ey, double* restrict ez,
    double* restrict bx, double* restrict by, double* restrict bz,
    double inv_dx, double inv_dy, double inv_dz, double x0, double y0, double z0,
    npy_intp nx, npy_intp ny, npy_intp nz
) {

    double gx[3];
    double gy[3];
    double gz[3];
    double hx[3];
    double hy[3];
    double hz[3];

    double x_over_dx = (x - x0) * inv_dx;
    double y_over_dy = (y - y0) * inv_dy;
    double z_over_dz = (z - z0) * inv_dz;

    // Fast pre-computation with (int) - safe if result passes bounds check
    int ix1_fast = (int)(x_over_dx + 0.5);
    int ix2_fast = (int)x_over_dx;
    int iy1_fast = (int)(y_over_dy + 0.5);
    int iy2_fast = (int)y_over_dy;
    int iz1_fast = (int)(z_over_dz + 0.5);
    int iz2_fast = (int)z_over_dz;

    int use_fast_path = (ix2_fast > 0 && ix2_fast < nx - 1 && ix1_fast > 0 && ix1_fast < nx - 1 &&
                         iy2_fast > 0 && iy2_fast < ny - 1 && iy1_fast > 0 && iy1_fast < ny - 1 &&
                         iz2_fast > 0 && iz2_fast < nz - 1 && iz1_fast > 0 && iz1_fast < nz - 1);

    if (use_fast_path) {
        get_gx(ix1_fast - x_over_dx, gx);
        get_gx(ix2_fast - x_over_dx + 0.5, hx);
        get_gx(iy1_fast - y_over_dy, gy);
        get_gx(iy2_fast - y_over_dy + 0.5, hy);
        get_gx(iz1_fast - z_over_dz, gz);
        get_gx(iz2_fast - z_over_dz + 0.5, hz);

        *ex_part = interp_field_fast_3d(ex, hx, gy, gz, ix2_fast, iy1_fast, iz1_fast, ny, nz);
        *ey_part = interp_field_fast_3d(ey, gx, hy, gz, ix1_fast, iy2_fast, iz1_fast, ny, nz);
        *ez_part = interp_field_fast_3d(ez, gx, gy, hz, ix1_fast, iy1_fast, iz2_fast, ny, nz);
        *bx_part = interp_field_fast_3d(bx, gx, hy, hz, ix1_fast, iy2_fast, iz2_fast, ny, nz);
        *by_part = interp_field_fast_3d(by, hx, gy, hz, ix2_fast, iy1_fast, iz2_fast, ny, nz);
        *bz_part = interp_field_fast_3d(bz, hx, hy, gz, ix2_fast, iy2_fast, iz1_fast, ny, nz);
    } else {
        npy_intp ix1 = (int)floor(x_over_dx + 0.5);
        get_gx(ix1 - x_over_dx, gx);
        npy_intp ix2 = (int)floor(x_over_dx);
        get_gx(ix2 - x_over_dx + 0.5, hx);
        npy_intp iy1 = (int)floor(y_over_dy + 0.5);
        get_gx(iy1 - y_over_dy, gy);
        npy_intp iy2 = (int)floor(y_over_dy);
        get_gx(iy2 - y_over_dy + 0.5, hy);
        npy_intp iz1 = (int)floor(z_over_dz + 0.5);
        get_gx(iz1 - z_over_dz, gz);
        npy_intp iz2 = (int)floor(z_over_dz);
        get_gx(iz2 - z_over_dz + 0.5, hz);

        *ex_part = interp_field_safe_3d(ex, hx, gy, gz, ix2, iy1, iz1, nx, ny, nz);
        *ey_part = interp_field_safe_3d(ey, gx, hy, gz, ix1, iy2, iz1, nx, ny, nz);
        *ez_part = interp_field_safe_3d(ez, gx, gy, hz, ix1, iy1, iz2, nx, ny, nz);
        *bx_part = interp_field_safe_3d(bx, gx, hy, hz, ix1, iy2, iz2, nx, ny, nz);
        *by_part = interp_field_safe_3d(by, hx, gy, hz, ix2, iy1, iz2, nx, ny, nz);
        *bz_part = interp_field_safe_3d(bz, hx, hy, gz, ix2, iy2, iz1, nx, ny, nz);
    }
}

static PyObject* unified_boris_pusher_cpu_3d(PyObject* self, PyObject* args) {
    PyObject *fields_list, *particles_list;
    npy_intp npatches;
    double dt, q, m;

    if (!PyArg_ParseTuple(args, "OOnddd",
        &particles_list, &fields_list,
        &npatches, &dt, &q, &m)) {
        return NULL;
    }

    if (npatches <= 0) {
        Py_RETURN_NONE;
    }

    npy_intp nx = PyLong_AsLong(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "nx"));
    npy_intp ny = PyLong_AsLong(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "ny"));
    npy_intp nz = PyLong_AsLong(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "nz"));
    npy_intp n_guard = PyLong_AsLong(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "n_guard"));
    nx += 2*n_guard;
    ny += 2*n_guard;
    nz += 2*n_guard;
    double dx = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "dx"));
    double dy = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "dy"));
    double dz = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "dz"));

    // fields
    AUTOFREE double **ex         = get_attr_array_double(fields_list, npatches, "ex");
    AUTOFREE double **ey         = get_attr_array_double(fields_list, npatches, "ey");
    AUTOFREE double **ez         = get_attr_array_double(fields_list, npatches, "ez");
    AUTOFREE double **bx         = get_attr_array_double(fields_list, npatches, "bx");
    AUTOFREE double **by         = get_attr_array_double(fields_list, npatches, "by");
    AUTOFREE double **bz         = get_attr_array_double(fields_list, npatches, "bz");
    AUTOFREE double **rho        = get_attr_array_double(fields_list, npatches, "rho");
    AUTOFREE double **jx         = get_attr_array_double(fields_list, npatches, "jx");
    AUTOFREE double **jy         = get_attr_array_double(fields_list, npatches, "jy");
    AUTOFREE double **jz         = get_attr_array_double(fields_list, npatches, "jz");
    AUTOFREE double *x0          = get_attr_double(fields_list, npatches, "x0");
    AUTOFREE double *y0          = get_attr_double(fields_list, npatches, "y0");
    AUTOFREE double *z0          = get_attr_double(fields_list, npatches, "z0");

    // particles
    AUTOFREE double **x          = get_attr_array_double(particles_list, npatches, "x");
    AUTOFREE double **y          = get_attr_array_double(particles_list, npatches, "y");
    AUTOFREE double **z          = get_attr_array_double(particles_list, npatches, "z");
    AUTOFREE double **ux         = get_attr_array_double(particles_list, npatches, "ux");
    AUTOFREE double **uy         = get_attr_array_double(particles_list, npatches, "uy");
    AUTOFREE double **uz         = get_attr_array_double(particles_list, npatches, "uz");
    AUTOFREE double **inv_gamma  = get_attr_array_double(particles_list, npatches, "inv_gamma");
    AUTOFREE double **ex_part    = get_attr_array_double(particles_list, npatches, "ex_part");
    AUTOFREE double **ey_part    = get_attr_array_double(particles_list, npatches, "ey_part");
    AUTOFREE double **ez_part    = get_attr_array_double(particles_list, npatches, "ez_part");
    AUTOFREE double **bx_part    = get_attr_array_double(particles_list, npatches, "bx_part");
    AUTOFREE double **by_part    = get_attr_array_double(particles_list, npatches, "by_part");
    AUTOFREE double **bz_part    = get_attr_array_double(particles_list, npatches, "bz_part");
    AUTOFREE npy_bool **is_dead  = get_attr_array_bool(particles_list, npatches, "is_dead");
    AUTOFREE double **w          = get_attr_array_double(particles_list, npatches, "w");
    AUTOFREE npy_intp *npart     = get_attr_int(particles_list, npatches, "npart");

    // release GIL
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        // Cache patch-local pointers to avoid double indirection
        double* x_patch = x[ipatch];
        double* y_patch = y[ipatch];
        double* z_patch = z[ipatch];
        double* ux_patch = ux[ipatch];
        double* uy_patch = uy[ipatch];
        double* uz_patch = uz[ipatch];
        double* inv_gamma_patch = inv_gamma[ipatch];
        double* ex_part_patch = ex_part[ipatch];
        double* ey_part_patch = ey_part[ipatch];
        double* ez_part_patch = ez_part[ipatch];
        double* bx_part_patch = bx_part[ipatch];
        double* by_part_patch = by_part[ipatch];
        double* bz_part_patch = bz_part[ipatch];
        npy_bool* is_dead_patch = is_dead[ipatch];
        double* w_patch = w[ipatch];

        double* ex_field = ex[ipatch];
        double* ey_field = ey[ipatch];
        double* ez_field = ez[ipatch];
        double* bx_field = bx[ipatch];
        double* by_field = by[ipatch];
        double* bz_field = bz[ipatch];
        double* rho_field = rho[ipatch];
        double* jx_field = jx[ipatch];
        double* jy_field = jy[ipatch];
        double* jz_field = jz[ipatch];

        double x0_patch = x0[ipatch];
        double y0_patch = y0[ipatch];
        double z0_patch = z0[ipatch];
        npy_intp npart_patch = npart[ipatch];

        const double efactor = q * dt / (2 * m * LIGHT_SPEED);
        const double bfactor = q * dt / (2 * m);
        const double cdt_half = LIGHT_SPEED * 0.5 * dt;
        const double q_over_dx_dy_dz = q / (dx * dy * dz);
        const double q_over_dy_dz_dt = q / (dy * dz * dt);
        const double q_over_dx_dz_dt = q / (dx * dz * dt);
        const double q_over_dx_dy_dt = q / (dx * dy * dt);
        const double inv_dx = 1.0 / dx;
        const double inv_dy = 1.0 / dy;
        const double inv_dz = 1.0 / dz;

        // Phase 1: position push, interpolation, Boris push, position push
        for (npy_intp ip = 0; ip < npart_patch; ip++) {
            if (is_dead_patch[ip]) continue;
            if (isnan(x_patch[ip]) || isnan(y_patch[ip]) || isnan(z_patch[ip])) continue;
            push_position_3d(
                &x_patch[ip], &y_patch[ip], &z_patch[ip],
                ux_patch[ip], uy_patch[ip], uz_patch[ip], inv_gamma_patch[ip],
                cdt_half
            );
            interpolation_3d(
                x_patch[ip], y_patch[ip], z_patch[ip],
                &ex_part_patch[ip], &ey_part_patch[ip], &ez_part_patch[ip],
                &bx_part_patch[ip], &by_part_patch[ip], &bz_part_patch[ip],
                ex_field, ey_field, ez_field,
                bx_field, by_field, bz_field,
                inv_dx, inv_dy, inv_dz, x0_patch, y0_patch, z0_patch,
                nx, ny, nz
            );
            boris(
                &ux_patch[ip], &uy_patch[ip], &uz_patch[ip], &inv_gamma_patch[ip],
                ex_part_patch[ip], ey_part_patch[ip], ez_part_patch[ip],
                bx_part_patch[ip], by_part_patch[ip], bz_part_patch[ip],
                efactor, bfactor
            );
            push_position_3d(
                &x_patch[ip], &y_patch[ip], &z_patch[ip],
                ux_patch[ip], uy_patch[ip], uz_patch[ip], inv_gamma_patch[ip],
                cdt_half
            );
        }

        // Phase 2: current deposit
        for (npy_intp ip = 0; ip < npart_patch; ip++) {
            if (is_dead_patch[ip]) continue;
            if (isnan(x_patch[ip]) || isnan(y_patch[ip]) || isnan(z_patch[ip])) continue;
            current_deposit_3d_fast(
                rho_field, jx_field, jy_field, jz_field,
                x_patch[ip], y_patch[ip], z_patch[ip],
                ux_patch[ip], uy_patch[ip], uz_patch[ip], inv_gamma_patch[ip],
                nx, ny, nz,
                dx, dy, dz, x0_patch, y0_patch, z0_patch, dt, w_patch[ip],
                q_over_dx_dy_dz, q_over_dy_dz_dt, q_over_dx_dz_dt, q_over_dx_dy_dt
            );
        }
    }
    // acquire GIL
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

static PyMethodDef CpuMethods[] = {
    {"unified_boris_pusher_cpu_3d", unified_boris_pusher_cpu_3d, METH_VARARGS, "Unified Boris Pusher"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cpumodule = {
    PyModuleDef_HEAD_INIT,
    "unified_pusher_3d",
    NULL,
    -1,
    CpuMethods
};

PyMODINIT_FUNC PyInit_unified_pusher_3d(void) {
    import_array();
    return PyModule_Create(&cpumodule);
}
