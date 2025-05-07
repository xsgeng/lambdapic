#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>
#include "../utils/cutils.h"

static void calculate_S(double delta, int shift, double* S) {
    double delta2 = delta * delta;

    double delta_minus = 0.5 * (delta2 + delta + 0.25);
    double delta_mid = 0.75 - delta2;
    double delta_positive = 0.5 * (delta2 - delta + 0.25);

    int minus = shift == -1;
    int mid = shift == 0;
    int positive = shift == 1;

    S[0] = minus * delta_minus;
    S[1] = minus * delta_mid + mid * delta_minus;
    S[2] = minus * delta_positive + mid * delta_mid + positive * delta_minus;
    S[3] = mid * delta_positive + positive * delta_mid;
    S[4] = positive * delta_positive;
}

inline static void current_deposit_3d(
    double* rho, double* jx, double* jy, double* jz, 
    double x, double y, double z,
    double ux, double uy, double uz, double inv_gamma,
    npy_intp nx, npy_intp ny, npy_intp nz,
    double dx, double dy, double dz, double x0, double y0, double z0, double dt, double w, double q
) {
    double vx = ux * LIGHT_SPEED * inv_gamma;
    double vy = uy * LIGHT_SPEED * inv_gamma;
    double vz = uz * LIGHT_SPEED * inv_gamma;
    double x_old = x - vx * 0.5 * dt - x0;
    double x_adv = x + vx * 0.5 * dt - x0;
    double y_old = y - vy * 0.5 * dt - y0;
    double y_adv = y + vy * 0.5 * dt - y0;
    double z_old = z - vz * 0.5 * dt - z0;
    double z_adv = z + vz * 0.5 * dt - z0;

    double x_over_dx0 = x_old / dx;
    int ix0 = (int)floor(x_over_dx0 + 0.5);
    double y_over_dy0 = y_old / dy;
    int iy0 = (int)floor(y_over_dy0 + 0.5);
    double z_over_dz0 = z_old / dz;
    int iz0 = (int)floor(z_over_dz0 + 0.5);

    double S0x[5], S0y[5], S0z[5];
    calculate_S(ix0 - x_over_dx0, 0, S0x);
    calculate_S(iy0 - y_over_dy0, 0, S0y);
    calculate_S(iz0 - z_over_dz0, 0, S0z);

    double x_over_dx1 = x_adv / dx;
    int ix1 = (int)floor(x_over_dx1 + 0.5);
    int dcell_x = ix1 - ix0;

    double y_over_dy1 = y_adv / dy;
    int iy1 = (int)floor(y_over_dy1 + 0.5);
    int dcell_y = iy1 - iy0;

    double z_over_dz1 = z_adv / dz;
    int iz1 = (int)floor(z_over_dz1 + 0.5);
    int dcell_z = iz1 - iz0;
    
    double S1x[5], S1y[5], S1z[5], DSx[5], DSy[5], DSz[5];
    calculate_S(ix1 - x_over_dx1, dcell_x, S1x);
    calculate_S(iy1 - y_over_dy1, dcell_y, S1y);
    calculate_S(iz1 - z_over_dz1, dcell_z, S1z);

    for (int i = 0; i < 5; i++) {
        DSx[i] = S1x[i] - S0x[i];
        DSy[i] = S1y[i] - S0y[i];
        DSz[i] = S1z[i] - S0z[i];
    }

    double charge_density = q * w / (dx * dy * dz);
    double factor = charge_density / dt;

    double jx_buff[5][5] = {0};
    for (int i = fmin(1, 1 + dcell_x); i < fmax(4, 4 + dcell_x); i++) {
        int ix = ix0 + (i - 2);
        
        double jy_buff[5] = {0};
        for (int j = fmin(1, 1 + dcell_y); j < fmax(4, 4 + dcell_y); j++) {
            int iy = iy0 + (j - 2);
            
            double jz_buff = 0;
            for (int k = fmin(1, 1 + dcell_z); k < fmax(4, 4 + dcell_z); k++) {
                int iz = iz0 + (k - 2);

                double wx = DSx[i] * (S0y[j]*S0z[k] + 0.5 * DSy[j]*S0z[k] + 0.5*S0y[j]*DSz[k] + one_third*DSy[j]*DSz[k]);
                double wy = DSy[j] * (S0x[i]*S0z[k] + 0.5 * DSx[i]*S0z[k] + 0.5*S0x[i]*DSz[k] + one_third*DSx[i]*DSz[k]);
                double wz = DSz[k] * (S0x[i]*S0y[j] + 0.5 * DSx[i]*S0y[j] + 0.5*S0x[i]*DSy[j] + one_third*DSx[i]*DSy[j]);

                jx_buff[k][j] -= factor * dx * wx;
                jy_buff[k] -= factor * dy * wy;
                jz_buff -= factor * dz * wz;

                jx[INDEX3(ix, iy, iz)] += jx_buff[k][j];
                jy[INDEX3(ix, iy, iz)] += jy_buff[k];
                jz[INDEX3(ix, iy, iz)] += jz_buff;
                rho[INDEX3(ix, iy, iz)] += charge_density * S1x[i] * S1y[j] * S1z[k];
            }
        }
    }
}

static PyObject* current_deposition_cpu_3d(PyObject* self, PyObject* args) {
    PyObject *fields_list, *particles_list;
    npy_intp npatches;
    double dt, q;

    if (!PyArg_ParseTuple(args, "OOndd", 
        &fields_list, &particles_list,
        &npatches, &dt, &q)) {
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
    AUTOFREE npy_bool **is_dead  = get_attr_array_bool(particles_list, npatches, "is_dead");
    AUTOFREE double **w          = get_attr_array_double(particles_list, npatches, "w");
    AUTOFREE npy_intp *npart     = get_attr_int(particles_list, npatches, "npart");

    // release GIL
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        for (npy_intp ip = 0; ip < npart[ipatch]; ip++) {
            if (is_dead[ipatch][ip]) continue;
            if (isnan(x[ipatch][ip]) || isnan(y[ipatch][ip]) || isnan(z[ipatch][ip])) continue;
            current_deposit_3d(
                rho[ipatch], jx[ipatch], jy[ipatch], jz[ipatch], 
                x[ipatch][ip], y[ipatch][ip], z[ipatch][ip], 
                ux[ipatch][ip], uy[ipatch][ip], uz[ipatch][ip], inv_gamma[ipatch][ip],
                nx, ny, nz,
                dx, dy, dz, x0[ipatch], y0[ipatch], z0[ipatch], dt, w[ipatch][ip], q
            );
        }
    }
    // reacquire GIL
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyMethodDef CpuMethods[] = {
    {"current_deposition_cpu_3d", current_deposition_cpu_3d, METH_VARARGS, "Current deposition on CPU"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cpumodule = {
    PyModuleDef_HEAD_INIT,
    "cpu3d",
    NULL,
    -1,
    CpuMethods
};

PyMODINIT_FUNC PyInit_cpu3d(void) {
    import_array();
    return PyModule_Create(&cpumodule);
}
