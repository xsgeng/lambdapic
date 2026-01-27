#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include "../utils/cutils.h"
#include "current_deposit.h"

static void reset_current_2d(double* jx, double* jy, double* jz, double* rho, npy_intp nx, npy_intp ny) {
    npy_intp n_elements = nx * ny;
    memset(jx,  0, n_elements * sizeof(double));
    memset(jy,  0, n_elements * sizeof(double));
    memset(jz,  0, n_elements * sizeof(double));
    memset(rho, 0, n_elements * sizeof(double));
}



static PyObject* reset_current_cpu_2d(PyObject* self, PyObject* args) {
    PyObject *fields_list;
    npy_intp npatches;

    if (!PyArg_ParseTuple(args, "On", &fields_list, &npatches)) {
        return NULL;
    }

    if (npatches <= 0) {
        Py_RETURN_NONE;
    }
    
    // Get the first field to get nx, ny, and n_guard. All fields have the same dimensions and layout.
    npy_intp nx = PyLong_AsLong(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "nx"));
    npy_intp ny = PyLong_AsLong(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "ny"));
    npy_intp n_guard = PyLong_AsLong(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "n_guard"));
    nx += 2*n_guard;
    ny += 2*n_guard;

    AUTOFREE double **rho        = (double**) malloc(npatches * sizeof(double*));
    AUTOFREE double **jx         = (double**) malloc(npatches * sizeof(double*));
    AUTOFREE double **jy         = (double**) malloc(npatches * sizeof(double*));
    AUTOFREE double **jz         = (double**) malloc(npatches * sizeof(double*));

    // prestore the data in the list
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        PyObject *fields = PyList_GetItem(fields_list, ipatch);

        PyObject *rho_npy = PyObject_GetAttrString(fields, "rho");
        PyObject *jx_npy  = PyObject_GetAttrString(fields, "jx");
        PyObject *jy_npy  = PyObject_GetAttrString(fields, "jy");
        PyObject *jz_npy  = PyObject_GetAttrString(fields, "jz");

        rho[ipatch] = (double*) PyArray_DATA((PyArrayObject*) rho_npy);
        jx[ipatch]  = (double*) PyArray_DATA((PyArrayObject*) jx_npy);
        jy[ipatch]  = (double*) PyArray_DATA((PyArrayObject*) jy_npy);
        jz[ipatch]  = (double*) PyArray_DATA((PyArrayObject*) jz_npy);

        Py_DecRef(rho_npy);
        Py_DecRef(jx_npy);
        Py_DecRef(jy_npy);
        Py_DecRef(jz_npy);
    }

    // release GIL
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        reset_current_2d(jx[ipatch], jy[ipatch], jz[ipatch], rho[ipatch], nx, ny);
    }
    // reacquire GIL
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyObject* current_deposition_cpu_2d(PyObject* self, PyObject* args) {
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
    npy_intp n_guard = PyLong_AsLong(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "n_guard"));
    nx += 2*n_guard;
    ny += 2*n_guard;
    double dx = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "dx"));
    double dy = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GET_ITEM(fields_list, 0), "dy"));

    AUTOFREE double **rho        = (double**) malloc(npatches * sizeof(double*));
    AUTOFREE double **jx         = (double**) malloc(npatches * sizeof(double*));
    AUTOFREE double **jy         = (double**) malloc(npatches * sizeof(double*));
    AUTOFREE double **jz         = (double**) malloc(npatches * sizeof(double*));
    AUTOFREE double *x0          = (double*)  malloc(npatches * sizeof(double));
    AUTOFREE double *y0          = (double*)  malloc(npatches * sizeof(double));

    AUTOFREE double **x          = (double**) malloc(npatches * sizeof(double*));
    AUTOFREE double **y          = (double**) malloc(npatches * sizeof(double*));
    AUTOFREE double **ux         = (double**) malloc(npatches * sizeof(double*));
    AUTOFREE double **uy         = (double**) malloc(npatches * sizeof(double*));
    AUTOFREE double **uz         = (double**) malloc(npatches * sizeof(double*));
    AUTOFREE double **inv_gamma  = (double**) malloc(npatches * sizeof(double*));
    AUTOFREE npy_bool **is_dead  = (npy_bool**) malloc(npatches * sizeof(npy_bool*));
    AUTOFREE double **w          = (double**) malloc(npatches * sizeof(double*));
    AUTOFREE npy_intp *npart     = (npy_intp*) malloc(npatches * sizeof(npy_intp));

    // prestore the data in the list
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        PyObject *fields = PyList_GetItem(fields_list, ipatch);
        PyObject *particles = PyList_GetItem(particles_list, ipatch);

        PyObject *rho_npy = PyObject_GetAttrString(fields, "rho");
        PyObject *jx_npy  = PyObject_GetAttrString(fields, "jx");
        PyObject *jy_npy  = PyObject_GetAttrString(fields, "jy");
        PyObject *jz_npy  = PyObject_GetAttrString(fields, "jz");
        x0[ipatch]        = PyFloat_AsDouble(PyObject_GetAttrString(fields, "x0"));
        y0[ipatch]        = PyFloat_AsDouble(PyObject_GetAttrString(fields, "y0"));

        PyObject *x_npy          = PyObject_GetAttrString(particles, "x");
        PyObject *y_npy          = PyObject_GetAttrString(particles, "y");
        PyObject *ux_npy         = PyObject_GetAttrString(particles, "ux");
        PyObject *uy_npy         = PyObject_GetAttrString(particles, "uy");
        PyObject *uz_npy         = PyObject_GetAttrString(particles, "uz");
        PyObject *inv_gamma_npy  = PyObject_GetAttrString(particles, "inv_gamma");
        PyObject *is_dead_npy    = PyObject_GetAttrString(particles, "is_dead");
        PyObject *w_npy          = PyObject_GetAttrString(particles, "w");

        rho[ipatch] = (double*) PyArray_DATA((PyArrayObject*) rho_npy);
        jx[ipatch]  = (double*) PyArray_DATA((PyArrayObject*) jx_npy);
        jy[ipatch]  = (double*) PyArray_DATA((PyArrayObject*) jy_npy);
        jz[ipatch]  = (double*) PyArray_DATA((PyArrayObject*) jz_npy);
        
        x[ipatch]         = (double*) PyArray_DATA((PyArrayObject*) x_npy);
        y[ipatch]         = (double*) PyArray_DATA((PyArrayObject*) y_npy);
        ux[ipatch]        = (double*) PyArray_DATA((PyArrayObject*) ux_npy);
        uy[ipatch]        = (double*) PyArray_DATA((PyArrayObject*) uy_npy);
        uz[ipatch]        = (double*) PyArray_DATA((PyArrayObject*) uz_npy);
        inv_gamma[ipatch] = (double*) PyArray_DATA((PyArrayObject*) inv_gamma_npy);
        is_dead[ipatch]   = (npy_bool*) PyArray_DATA((PyArrayObject*) is_dead_npy);
        w[ipatch]         = (double*) PyArray_DATA((PyArrayObject*) w_npy);

        npart[ipatch]     = PyArray_DIM((PyArrayObject*) w_npy, 0);

        Py_DecRef(rho_npy);
        Py_DecRef(jx_npy);
        Py_DecRef(jy_npy);
        Py_DecRef(jz_npy);
        Py_DecRef(x_npy);
        Py_DecRef(y_npy);
        Py_DecRef(ux_npy);
        Py_DecRef(uy_npy);
        Py_DecRef(uz_npy);
        Py_DecRef(inv_gamma_npy);
        Py_DecRef(is_dead_npy);
        Py_DecRef(w_npy);
    }

    // release GIL
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        for (npy_intp ip = 0; ip < npart[ipatch]; ip++) {
            if (is_dead[ipatch][ip]) continue;
            if (isnan(x[ipatch][ip]) || isnan(y[ipatch][ip])) continue;
            current_deposit_2d(
                rho[ipatch], jx[ipatch], jy[ipatch], jz[ipatch], 
                x[ipatch][ip], y[ipatch][ip], 
                ux[ipatch][ip], uy[ipatch][ip], uz[ipatch][ip], inv_gamma[ipatch][ip],
                nx, ny,
                dx, dy, x0[ipatch], y0[ipatch], dt, w[ipatch][ip], q
            );
        }
    }
    // reacquire GIL
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyMethodDef CpuMethods[] = {
    {"current_deposition_cpu_2d", current_deposition_cpu_2d, METH_VARARGS, "Current deposition on CPU"},
    {"reset_current_cpu_2d", reset_current_cpu_2d, METH_VARARGS, "Reset current and charge density arrays to zero"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cpumodule = {
    PyModuleDef_HEAD_INIT,
    "cpu_2d",
    NULL,
    -1,
    CpuMethods
};

PyMODINIT_FUNC PyInit_cpu2d(void) {
    import_array();
    return PyModule_Create(&cpumodule);
}
