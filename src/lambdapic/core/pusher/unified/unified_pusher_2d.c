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

__attribute__((always_inline)) inline static void push_position_2d(
    double* restrict x, double* restrict y,
    double ux, double uy,
    double inv_gamma,
    double cdt
) {
    *x += cdt * inv_gamma * ux;
    *y += cdt * inv_gamma * uy;
}


__attribute__((always_inline)) inline static void get_gx(double delta, double* gx) {
    double delta2 = delta * delta;
    gx[0] = 0.5 * (0.25 + delta2 + delta);
    gx[1] = 0.75 - delta2;
    gx[2] = 0.5 * (0.25 + delta2 - delta);
}

// Slow path: uses INDEX2 with periodic boundary checks
__attribute__((always_inline)) inline static double interp_field_safe(double* restrict field, double* restrict fac1, double* restrict fac2, npy_intp ix, npy_intp iy, npy_intp nx, npy_intp ny) {
    double field_part =
          fac2[0] * (fac1[0] * field[INDEX2(ix-1, iy-1)]
        +            fac1[1] * field[INDEX2(ix,   iy-1)]
        +            fac1[2] * field[INDEX2(ix+1, iy-1)])
        + fac2[1] * (fac1[0] * field[INDEX2(ix-1, iy  )]
        +            fac1[1] * field[INDEX2(ix,   iy  )]
        +            fac1[2] * field[INDEX2(ix+1, iy  )])
        + fac2[2] * (fac1[0] * field[INDEX2(ix-1, iy+1)]
        +            fac1[1] * field[INDEX2(ix,   iy+1)]
        +            fac1[2] * field[INDEX2(ix+1, iy+1)]);
    return field_part;
}

__attribute__((always_inline)) inline static double interp_field_fast(
    double* restrict field, double* restrict fac1, double* restrict fac2,
    int ix, int iy, int ny
) {
    int im1 = (ix - 1) * ny, i0 = ix * ny, ip1 = (ix + 1) * ny;
    int jm1 = iy - 1, j0 = iy, jp1 = iy + 1;
    return fac2[0] * (fac1[0] * field[im1 + jm1] + fac1[1] * field[i0 + jm1] + fac1[2] * field[ip1 + jm1])
         + fac2[1] * (fac1[0] * field[im1 + j0]  + fac1[1] * field[i0 + j0]  + fac1[2] * field[ip1 + j0])
         + fac2[2] * (fac1[0] * field[im1 + jp1] + fac1[1] * field[i0 + jp1] + fac1[2] * field[ip1 + jp1]);
}

__attribute__((always_inline)) inline static void interpolation_2d(
    double x, double y,
    double* restrict ex_part, double* restrict ey_part, double* restrict ez_part,
    double* restrict bx_part, double* restrict by_part, double* restrict bz_part,
    double* restrict ex, double* restrict ey, double* restrict ez,
    double* restrict bx, double* restrict by, double* restrict bz,
    double inv_dx, double inv_dy, double x0, double y0,
    npy_intp nx, npy_intp ny
) {

    double gx[3];
    double gy[3];
    double hx[3];
    double hy[3];

    double x_over_dx = (x - x0) * inv_dx;
    double y_over_dy = (y - y0) * inv_dy;

    // Fast pre-computation with (int) - safe if result passes bounds check
    int ix1_fast = (int)(x_over_dx + 0.5);
    int ix2_fast = (int)x_over_dx;
    int iy1_fast = (int)(y_over_dy + 0.5);
    int iy2_fast = (int)y_over_dy;

    int use_fast_path = (ix2_fast > 0 && ix2_fast < nx - 1 && ix1_fast > 0 && ix1_fast < nx - 1 &&
                         iy2_fast > 0 && iy2_fast < ny - 1 && iy1_fast > 0 && iy1_fast < ny - 1);

    if (use_fast_path) {
        get_gx(ix1_fast - x_over_dx, gx);
        get_gx(ix2_fast - x_over_dx + 0.5, hx);
        get_gx(iy1_fast - y_over_dy, gy);
        get_gx(iy2_fast - y_over_dy + 0.5, hy);

        *ex_part = interp_field_fast(ex, hx, gy, ix2_fast, iy1_fast, ny);
        *ey_part = interp_field_fast(ey, gx, hy, ix1_fast, iy2_fast, ny);
        *ez_part = interp_field_fast(ez, gx, gy, ix1_fast, iy1_fast, ny);
        *bx_part = interp_field_fast(bx, gx, hy, ix1_fast, iy2_fast, ny);
        *by_part = interp_field_fast(by, hx, gy, ix2_fast, iy1_fast, ny);
        *bz_part = interp_field_fast(bz, hx, hy, ix2_fast, iy2_fast, ny);
    } else {
        npy_intp ix1 = (int)floor(x_over_dx + 0.5);
        get_gx(ix1 - x_over_dx, gx);
        npy_intp ix2 = (int)floor(x_over_dx);
        get_gx(ix2 - x_over_dx + 0.5, hx);
        npy_intp iy1 = (int)floor(y_over_dy + 0.5);
        get_gx(iy1 - y_over_dy, gy);
        npy_intp iy2 = (int)floor(y_over_dy);
        get_gx(iy2 - y_over_dy + 0.5, hy);

        *ex_part = interp_field_safe(ex, hx, gy, ix2, iy1, nx, ny);
        *ey_part = interp_field_safe(ey, gx, hy, ix1, iy2, nx, ny);
        *ez_part = interp_field_safe(ez, gx, gy, ix1, iy1, nx, ny);

        *bx_part = interp_field_safe(bx, gx, hy, ix1, iy2, nx, ny);
        *by_part = interp_field_safe(by, hx, gy, ix2, iy1, nx, ny);
        *bz_part = interp_field_safe(bz, hx, hy, ix2, iy2, nx, ny);
    }
}


static PyObject* unified_boris_pusher_cpu_2d(PyObject* self, PyObject* args) {
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
    npy_intp n_guard = PyLong_AsLong(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "n_guard"));
    nx += 2*n_guard;
    ny += 2*n_guard;
    double dx = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "dx"));
    double dy = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "dy"));

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

    // particles
    AUTOFREE double **x          = get_attr_array_double(particles_list, npatches, "x");
    AUTOFREE double **y          = get_attr_array_double(particles_list, npatches, "y");
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
        npy_intp npart_patch = npart[ipatch];
        
        const double efactor = q * dt / (2 * m * LIGHT_SPEED);
        const double bfactor = q * dt / (2 * m);
        const double cdt_half = LIGHT_SPEED * 0.5 * dt;
        const double q_over_dx_dy = q / (dx * dy);
        const double q_over_dy_dt = q / (dy * dt);
        const double q_over_dx_dt = q / (dx * dt);
        const double inv_dx = 1.0 / dx;
        const double inv_dy = 1.0 / dy;

        // Phase 1: position push, interpolation, Boris push, position push
        for (npy_intp ip = 0; ip < npart_patch; ip++) {
            if (is_dead_patch[ip]) continue;
            if (isnan(x_patch[ip]) || isnan(y_patch[ip])) continue;
            push_position_2d(
                &x_patch[ip], &y_patch[ip],
                ux_patch[ip], uy_patch[ip], inv_gamma_patch[ip],
                cdt_half
            );
            interpolation_2d(
                x_patch[ip], y_patch[ip],
                &ex_part_patch[ip], &ey_part_patch[ip], &ez_part_patch[ip],
                &bx_part_patch[ip], &by_part_patch[ip], &bz_part_patch[ip],
                ex_field, ey_field, ez_field,
                bx_field, by_field, bz_field,
                inv_dx, inv_dy, x0_patch, y0_patch,
                nx, ny
            );
            boris(
                &ux_patch[ip], &uy_patch[ip], &uz_patch[ip], &inv_gamma_patch[ip],
                ex_part_patch[ip], ey_part_patch[ip], ez_part_patch[ip],
                bx_part_patch[ip], by_part_patch[ip], bz_part_patch[ip],
                efactor, bfactor
            );
            push_position_2d(
                &x_patch[ip], &y_patch[ip],
                ux_patch[ip], uy_patch[ip], inv_gamma_patch[ip],
                cdt_half
            );
        }

        // Phase 2: current deposit
        for (npy_intp ip = 0; ip < npart_patch; ip++) {
            if (is_dead_patch[ip]) continue;
            if (isnan(x_patch[ip]) || isnan(y_patch[ip])) continue;
            current_deposit_2d_fast(
                rho_field, jx_field, jy_field, jz_field,
                x_patch[ip], y_patch[ip],
                ux_patch[ip], uy_patch[ip], uz_patch[ip], inv_gamma_patch[ip],
                nx, ny,
                dx, dy, x0_patch, y0_patch, dt, w_patch[ip],
                q_over_dx_dy, q_over_dy_dt, q_over_dx_dt
            );
        }
    }
    // acquire GIL
    Py_END_ALLOW_THREADS
    
    Py_RETURN_NONE;
}

static PyMethodDef CpuMethods[] = {
    {"unified_boris_pusher_cpu_2d", unified_boris_pusher_cpu_2d, METH_VARARGS, "Unified Boris Pusher"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cpumodule = {
    PyModuleDef_HEAD_INIT,
    "unified_pusher_2d",
    NULL,
    -1,
    CpuMethods
};

PyMODINIT_FUNC PyInit_unified_pusher_2d(void) {
    import_array();
    return PyModule_Create(&cpumodule);
}
