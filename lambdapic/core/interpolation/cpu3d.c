#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>
#include "../utils/cutils.h"

inline static void get_gx(double delta, double* gx) {
    double delta2 = delta * delta;
    gx[0] = 0.5 * (0.25 + delta2 + delta);
    gx[1] = 0.75 - delta2;
    gx[2] = 0.5 * (0.25 + delta2 - delta);
}

inline static double interp_field(
    double* field, 
    double* facx, double* facy, double* facz, 
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

inline static void interpolation_3d(
    double x, double y, double z,
    double* ex_part, double* ey_part, double* ez_part, 
    double* bx_part, double* by_part, double* bz_part, 
    double* ex, double* ey, double* ez, 
    double* bx, double* by, double* bz, 
    double dx, double dy, double dz, double x0, double y0, double z0, 
    npy_intp nx, npy_intp ny, npy_intp nz
) {
    
    double gx[3];
    double gy[3];
    double gz[3];
    double hx[3];
    double hy[3];
    double hz[3];
    
    double x_over_dx = (x - x0) / dx;
    double y_over_dy = (y - y0) / dy;
    double z_over_dz = (z - z0) / dz;

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

    *ex_part = interp_field(ex, hx, gy, gz, ix2, iy1, iz1, nx, ny, nz);
    *ey_part = interp_field(ey, gx, hy, gz, ix1, iy2, iz1, nx, ny, nz);
    *ez_part = interp_field(ez, gx, gy, hz, ix1, iy1, iz2, nx, ny, nz);

    *bx_part = interp_field(bx, gx, hy, hz, ix1, iy2, iz2, nx, ny, nz);
    *by_part = interp_field(by, hx, gy, hz, ix2, iy1, iz2, nx, ny, nz);
    *bz_part = interp_field(bz, hx, hy, gz, ix2, iy2, iz1, nx, ny, nz);
}

static PyObject* interpolation_patches_3d(PyObject* self, PyObject* args) {
    PyObject *fields_list, *particles_list;
    npy_intp npatches;
    if (!PyArg_ParseTuple(args, "OOn", 
        &particles_list, &fields_list, 
        &npatches)) {
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
    AUTOFREE double *x0          = get_attr_double(fields_list, npatches, "x0");
    AUTOFREE double *y0          = get_attr_double(fields_list, npatches, "y0");
    AUTOFREE double *z0          = get_attr_double(fields_list, npatches, "z0");

    // particles
    AUTOFREE double **x          = get_attr_array_double(particles_list, npatches, "x");
    AUTOFREE double **y          = get_attr_array_double(particles_list, npatches, "y");
    AUTOFREE double **z          = get_attr_array_double(particles_list, npatches, "z");
    AUTOFREE double **ex_part    = get_attr_array_double(particles_list, npatches, "ex_part");
    AUTOFREE double **ey_part    = get_attr_array_double(particles_list, npatches, "ey_part");
    AUTOFREE double **ez_part    = get_attr_array_double(particles_list, npatches, "ez_part");
    AUTOFREE double **bx_part    = get_attr_array_double(particles_list, npatches, "bx_part");
    AUTOFREE double **by_part    = get_attr_array_double(particles_list, npatches, "by_part");
    AUTOFREE double **bz_part    = get_attr_array_double(particles_list, npatches, "bz_part");
    AUTOFREE npy_bool **is_dead  = get_attr_array_bool(particles_list, npatches, "is_dead");
    AUTOFREE npy_intp *npart     = get_attr_int(particles_list, npatches, "npart");

    // release GIL
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        for (npy_intp ip = 0; ip < npart[ipatch]; ip++) {
            if (is_dead[ipatch][ip]) continue;
            if (isnan(x[ipatch][ip]) || isnan(y[ipatch][ip]) || isnan(z[ipatch][ip])) continue;
            interpolation_3d(
                x[ipatch][ip], y[ipatch][ip], z[ipatch][ip], 
                &ex_part[ipatch][ip], &ey_part[ipatch][ip], &ez_part[ipatch][ip], 
                &bx_part[ipatch][ip], &by_part[ipatch][ip], &bz_part[ipatch][ip], 
                ex[ipatch], ey[ipatch], ez[ipatch], 
                bx[ipatch], by[ipatch], bz[ipatch], 
                dx, dy, dz, x0[ipatch], y0[ipatch], z0[ipatch],
                nx, ny, nz
            );
        }
    }
    // reacquire GIL
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

static PyMethodDef InterpolationMethods[] = {
    {"interpolation_patches_3d", interpolation_patches_3d, METH_VARARGS, "2D interpolation on CPU"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef interpolationmodule = {
    PyModuleDef_HEAD_INIT,
    "cpu3d",
    NULL,
    -1,
    InterpolationMethods
};

PyMODINIT_FUNC PyInit_cpu3d(void) {
    import_array();
    return PyModule_Create(&interpolationmodule);
}
