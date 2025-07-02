#pragma once

#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>

// Type-independent cleanup function using void*
static void cleanup_ptr(void* p) {
    void** ptr = (void**)p;
    if (*ptr) free(*ptr);
    *ptr = NULL;
}
#define AUTOFREE __attribute__((cleanup(cleanup_ptr)))

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define LIGHT_SPEED 299792458.0
#define one_third 0.3333333333333333
#define INDEX2(i, j) \
    ((j) >= 0 ? (j) : (j) + (ny)) + \
    ((i) >= 0 ? (i) : (i) + (nx)) * (ny)

#define INDEX3(i, j, k) \
    ((k) >= 0 ? (k) : (k) + (nz)) + \
    ((j) >= 0 ? (j) : (j) + (ny)) * (nz) + \
    ((i) >= 0 ? (i) : (i) + (nx)) * (ny) * (nz)

/*
    Get attribute from a list of objects.
    Returns a pointer to an array of pointers to the attribute of each patch
*/
static inline double** get_attr_array_double(
    PyObject* list, 
    npy_intp npatches, 
    const char* attr
) {
    double **data = malloc(npatches * sizeof(double*));
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        PyObject *npy = PyObject_GetAttrString(PyList_GET_ITEM(list, ipatch), attr);
        data[ipatch] = (double*) PyArray_DATA((PyArrayObject*) npy);
        Py_DecRef(npy);
    }
    return data;
}

static inline npy_intp** get_attr_array_int(
    PyObject* list, 
    npy_intp npatches, 
    const char* attr
) {
    npy_intp **data = malloc(npatches * sizeof(npy_intp*));
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        PyObject *npy = PyObject_GetAttrString(PyList_GET_ITEM(list, ipatch), attr);
        data[ipatch] = (npy_intp*) PyArray_DATA((PyArrayObject*) npy);
        Py_DecRef(npy);
    }
    return data;
}

static inline npy_bool** get_attr_array_bool(
    PyObject* list, 
    npy_intp npatches, 
    const char* attr
) {
    npy_bool **data = malloc(npatches * sizeof(npy_bool*));
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        PyObject *npy = PyObject_GetAttrString(PyList_GET_ITEM(list, ipatch), attr);
        data[ipatch] = (npy_bool*) PyArray_DATA((PyArrayObject*) npy);
        Py_DecRef(npy);
    }
    return data;
}

static inline double* get_attr_double(
    PyObject* list, 
    npy_intp npatches, 
    const char* attr
) {
    double *data = malloc(npatches * sizeof(double));
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        PyObject *npy = PyObject_GetAttrString(PyList_GET_ITEM(list, ipatch), attr);
        data[ipatch] = PyFloat_AsDouble(npy);
        Py_DecRef(npy);
    }
    return data;
}

static inline npy_intp* get_attr_int(
    PyObject* list, 
    npy_intp npatches, 
    const char* attr
) {
    npy_intp *data = malloc(npatches * sizeof(npy_intp));
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        PyObject *npy = PyObject_GetAttrString(PyList_GET_ITEM(list, ipatch), attr);
        data[ipatch] = PyLong_AsLong(npy);
        Py_DecRef(npy);
    }
    return data;
}