/*
 * Shared C helpers for the binary-collision kernels.
 *
 * Included by collision/cpu.c.
 * Everything is static (inline): each extension module compiles its own copy,
 * so the .so files stay independent while the source has a single owner here.
 *
 * Contents: physical constants, PCG32 RNG (one stream per patch), the raw
 * particle-data view, COM-frame collision data, pairing iterators, and
 * Python argument helpers.
 */
#ifndef LAMBDAPIC_COLLISION_UTILS_H
#define LAMBDAPIC_COLLISION_UTILS_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* physical constants (scipy.constants values) */
static const double C_LIGHT = 299792458.0;
static const double EPSILON_0 = 8.8541878128e-12;
static const double PI = 3.141592653589793;
static const double PLANCK = 6.62607015e-34;
static const double M_ELECTRON = 9.1093837015e-31; /* electron mass (kg) */
static const double E_CHARGE = 1.602176634e-19;   /* elementary charge (C) */
static const double M_U = 1.66053906892e-27;      /* atomic mass unit (kg) */

#define GetPatchArrayData(list, ipatch) PyArray_DATA((PyArrayObject*)PyList_GetItem(list, ipatch))

/* NaN-free max: compiles to a single maxsd. Plain fmax stays a libm call
 * (gcc won't relax its NaN semantics without -ffast-math) and cost ~15% of
 * the collision kernel via per-pair PLT calls. Identical to fmax for
 * non-NaN inputs (numba's max() has the same ternary semantics). */
static inline double dmax(double a, double b) {
    return a > b ? a : b;
}

/* ------------------------------------------------------------------ */
/* PCG32 RNG (http://pcg-random.org), one stream per patch             */
/* ------------------------------------------------------------------ */

/* XSH-RR output (official pcg_basic.c): ((old >> 18) ^ old) >> 27.
 * The 02 port had ((old>>18) ^ (old>>27)) >> 27 - only 19 significant
 * bits, which skewed outputs toward zero (74% of draws < 0.01 deep in
 * the stream) and inflated collision event rates 25x. */
static inline uint32_t pcg32_next(uint64_t *rng) {
    uint64_t oldstate = rng[0];
    rng[0] = oldstate * 6364136223846793005ULL + rng[1];
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

/* uniform double in [0, 1), single 24-bit draw: scatter angles do not need
 * more resolution, and one draw per double halves RNG cost per pair */
static inline double pcg32_double(uint64_t *rng) {
    return (double)(pcg32_next(rng) >> 8) * (1.0 / 16777216.0);
}

/* uniform double in [lo, hi) */
static inline double pcg32_uniform(uint64_t *rng, double lo, double hi) {
    return lo + (hi - lo) * pcg32_double(rng);
}

/* unbiased bounded integer via rejection (PCG reference) */
static inline uint32_t pcg32_bounded(uint64_t *rng, uint32_t bound) {
    uint32_t threshold = (uint32_t)(-bound) % bound;
    for (;;) {
        uint32_t r = pcg32_next(rng);
        if (r >= threshold) return r % bound;
    }
}

/* Fisher-Yates shuffle of n int64 values (mirrors gen.shuffle).
 * state/inc are cached in locals across the loop: gcc must otherwise reload
 * them from memory on every draw (stores through the rng pointer may alias),
 * which cost ~13% of the collision kernel */
static inline void pcg32_shuffle(uint64_t *rng, npy_int64 *idx, npy_int64 n) {
    uint64_t state = rng[0];
    const uint64_t inc = rng[1];
    for (npy_int64 i = n - 1; i > 0; i--) {
        /* inline pcg32_bounded with the cached state */
        uint32_t bound = (uint32_t)(i + 1);
        uint32_t threshold = (uint32_t)(-bound) % bound;
        uint32_t r;
        do {
            uint64_t oldstate = state;
            state = oldstate * 6364136223846793005ULL + inc;
            uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
            uint32_t rot = (uint32_t)(oldstate >> 59u);
            r = (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
        } while (r < threshold);
        npy_int64 j = (npy_int64)(r % bound);
        npy_int64 tmp = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp;
    }
    rng[0] = state;
}

/* official pcg32_srandom_r seeding: initstate, initseq */
static inline void pcg32_seed(uint64_t *rng, uint64_t initstate, uint64_t initseq) {
    rng[0] = 0ULL;
    rng[1] = (initseq << 1u) | 1u;
    (void)pcg32_next(rng);
    rng[0] += initstate;
    (void)pcg32_next(rng);
}

/* ------------------------------------------------------------------ */
/* particle data (raw pointers into the numpy arrays)                  */
/* ------------------------------------------------------------------ */

typedef struct {
    double *x, *y, *z; /* NULL for collision (momentum-only) use */
    double *ux, *uy, *uz, *inv_gamma, *w;
    npy_bool *is_dead;
    npy_int64 npart; /* buffer length (allocated slots incl. dead) */
    double m, q;
} ParticleDataC;

typedef struct {
    double px1, py1, pz1, gamma1, vx1, vy1, vz1, m1, q1, w1;
    double px2, py2, pz2, gamma2, vx2, vy2, vz2, m2, q2, w2;
    double vx_com, vy_com, vz_com, gamma_com, v_com_square;
    double px1_com, py1_com, pz1_com, p1_com, p_perp;
    double gamma1_com, gamma2_com;
} CollisionDataC;

/* ------------------------------------------------------------------ */
/* COM-frame collision data                                            */
/* ------------------------------------------------------------------ */

static inline void collision_data_ww(
    ParticleDataC part1, npy_int64 ip1, double w1,
    ParticleDataC part2, npy_int64 ip2, double w2,
    CollisionDataC *d
) {
    const double m1 = part1.m, m2 = part2.m;
    const double c = C_LIGHT;

    d->m1 = m1; d->q1 = part1.q; d->w1 = w1;
    d->m2 = m2; d->q2 = part2.q; d->w2 = w2;

    d->px1 = part1.ux[ip1] * m1 * c;
    d->py1 = part1.uy[ip1] * m1 * c;
    d->pz1 = part1.uz[ip1] * m1 * c;
    d->gamma1 = 1.0 / part1.inv_gamma[ip1];

    d->px2 = part2.ux[ip2] * m2 * c;
    d->py2 = part2.uy[ip2] * m2 * c;
    d->pz2 = part2.uz[ip2] * m2 * c;
    d->gamma2 = 1.0 / part2.inv_gamma[ip2];

    d->vx1 = part1.ux[ip1] * part1.inv_gamma[ip1] * c;
    d->vy1 = part1.uy[ip1] * part1.inv_gamma[ip1] * c;
    d->vz1 = part1.uz[ip1] * part1.inv_gamma[ip1] * c;

    d->vx2 = part2.ux[ip2] * part2.inv_gamma[ip2] * c;
    d->vy2 = part2.uy[ip2] * part2.inv_gamma[ip2] * c;
    d->vz2 = part2.uz[ip2] * part2.inv_gamma[ip2] * c;

    d->vx_com = (d->px1 + d->px2) / (d->gamma1 * m1 + d->gamma2 * m2);
    d->vy_com = (d->py1 + d->py2) / (d->gamma1 * m1 + d->gamma2 * m2);
    d->vz_com = (d->pz1 + d->pz2) / (d->gamma1 * m1 + d->gamma2 * m2);
    d->v_com_square = d->vx_com * d->vx_com + d->vy_com * d->vy_com + d->vz_com * d->vz_com;
    d->gamma_com = 1.0 / sqrt(1.0 - d->v_com_square / (c * c));

    double fac = d->v_com_square > 0 ? (d->gamma_com - 1.0) / d->v_com_square : 0.0;
    double v1_dot_vcom = d->vx1 * d->vx_com + d->vy1 * d->vy_com + d->vz1 * d->vz_com;
    double v2_dot_vcom = d->vx2 * d->vx_com + d->vy2 * d->vy_com + d->vz2 * d->vz_com;

    d->px1_com = d->px1 + (fac * v1_dot_vcom - d->gamma_com) * m1 * d->gamma1 * d->vx_com;
    d->py1_com = d->py1 + (fac * v1_dot_vcom - d->gamma_com) * m1 * d->gamma1 * d->vy_com;
    d->pz1_com = d->pz1 + (fac * v1_dot_vcom - d->gamma_com) * m1 * d->gamma1 * d->vz_com;
    d->p1_com = sqrt(d->px1_com * d->px1_com + d->py1_com * d->py1_com + d->pz1_com * d->pz1_com);

    /* p2_com = -p1_com */

    d->gamma1_com = (1.0 - v1_dot_vcom / (c * c)) * d->gamma_com * d->gamma1;
    d->gamma2_com = (1.0 - v2_dot_vcom / (c * c)) * d->gamma_com * d->gamma2;

    d->p_perp = sqrt(d->px1_com * d->px1_com + d->py1_com * d->py1_com);
}

static inline void boost_to_lab(
    double px_com, double py_com, double pz_com,
    double gamma_com_particle, double m,
    double vx_com, double vy_com, double vz_com,
    double v_com_square, double gamma_com,
    double *out
) {
    double vcom_dot_p = vx_com * px_com + vy_com * py_com + vz_com * pz_com;
    double fac = v_com_square > 0 ? (gamma_com - 1.0) / v_com_square : 0.0;
    double e = fac * vcom_dot_p + m * gamma_com_particle * gamma_com;
    out[0] = px_com + vx_com * e;
    out[1] = py_com + vy_com * e;
    out[2] = pz_com + vz_com * e;
}

/* ------------------------------------------------------------------ */
/* pairing iterators (port of the numba jitclasses)                    */
/* ------------------------------------------------------------------ */

/* per-thread scratch buffer, reused across cells/patches to avoid a
 * malloc/free per cell */
typedef struct {
    npy_int64 *data;
    npy_int64 cap;
} IdxBuf;

static inline int idxbuf_ensure(IdxBuf *buf, npy_int64 n) {
    if (buf->cap < n) {
        free(buf->data);
        buf->data = (npy_int64*)malloc(n * sizeof(npy_int64));
        buf->cap = buf->data ? n : 0;
    }
    return buf->data ? 0 : -1;
}

typedef struct {
    npy_int64 *idx; /* absolute indices, shuffled (borrowed from IdxBuf) */
    npy_int64 ip_start, ip_end, nbuf;
    npy_int64 npart, npairs, dt_corr;
    int even;
    npy_int64 ip1, ip2, ipair;
} IntraPairing;

static inline int intra_pairing_init(IntraPairing *it, const npy_bool *dead,
                                     npy_int64 ip_start, npy_int64 ip_end, uint64_t *rng,
                                     IdxBuf *buf) {
    it->ip_start = ip_start;
    it->ip_end = ip_end;
    it->nbuf = ip_end - ip_start;

    npy_int64 npart = 0;
    for (npy_int64 ip = ip_start; ip < ip_end; ip++) {
        if (!dead[ip]) npart++;
    }
    it->npart = npart;

    if (npart >= 2) {
        it->npairs = (npart + 1) / 2;
        it->dt_corr = 2 * it->npairs - 1;
        it->even = (npart % 2) == 0;
    } else {
        it->npairs = 0;
        it->dt_corr = 0;
        it->even = 1;
    }

    npy_int64 nalloc = it->nbuf > 0 ? it->nbuf : 1;
    if (idxbuf_ensure(buf, nalloc) != 0) return -1;
    it->idx = buf->data;
    for (npy_int64 i = 0; i < nalloc; i++) it->idx[i] = ip_start + i;
    pcg32_shuffle(rng, it->idx, nalloc);

    it->ip1 = -1;
    it->ip2 = -1;
    it->ipair = 0;
    return 0;
}

/* returns 1 when a pair is produced, 0 when exhausted */
static inline int intra_pairing_next(IntraPairing *it, const npy_bool *dead,
                                     npy_int64 *ip1_abs, npy_int64 *ip2_abs,
                                     double *w_corr, double *dt_corr) {
    if (it->ipair >= it->npairs) return 0;

    /* find ip1 (first live after previous ip2) */
    for (npy_int64 j = it->ip2 + 1; j < it->nbuf; j++) {
        if (!dead[it->idx[j]]) { it->ip1 = j; break; }
    }
    /* find ip2 */
    if (it->even) {
        for (npy_int64 j = it->ip1 + 1; j < it->nbuf; j++) {
            if (!dead[it->idx[j]]) { it->ip2 = j; break; }
        }
    } else if (it->ipair < it->npairs - 1) {
        for (npy_int64 j = it->ip1 + 1; j < it->nbuf; j++) {
            if (!dead[it->idx[j]]) { it->ip2 = j; break; }
        }
    } else {
        /* last pair wraps to the first live particle */
        for (npy_int64 j = 0; j < it->ip1; j++) {
            if (!dead[it->idx[j]]) { it->ip2 = j; break; }
        }
    }

    *w_corr = 1.0;
    if (!it->even && (it->ipair == 0 || it->ipair == it->npairs - 1)) {
        *w_corr = 0.5;
    }

    *ip1_abs = it->idx[it->ip1];
    *ip2_abs = it->idx[it->ip2];
    *dt_corr = (double)it->dt_corr;
    it->ipair++;
    return 1;
}

typedef struct {
    npy_int64 *shuffled_idx; /* absolute indices of the shuffled side (borrowed from IdxBuf) */
    npy_int64 ip_start1, ip_end1, ip_start2, ip_end2;
    npy_int64 nbuf1, nbuf2;
    npy_int64 npart1, npart2, npairs, npairs_not_repeated, dt_corr;
    npy_int64 ip1, ip2, ipair;
} InterPairing;

static inline int inter_pairing_init(InterPairing *it, const npy_bool *dead1,
                                     npy_int64 ip_start1, npy_int64 ip_end1,
                                     const npy_bool *dead2,
                                     npy_int64 ip_start2, npy_int64 ip_end2,
                                     uint64_t *rng, IdxBuf *buf) {
    it->ip_start1 = ip_start1; it->ip_end1 = ip_end1;
    it->ip_start2 = ip_start2; it->ip_end2 = ip_end2;
    it->nbuf1 = ip_end1 - ip_start1;
    it->nbuf2 = ip_end2 - ip_start2;

    npy_int64 npart1 = 0, npart2 = 0;
    for (npy_int64 ip = ip_start1; ip < ip_end1; ip++) if (!dead1[ip]) npart1++;
    for (npy_int64 ip = ip_start2; ip < ip_end2; ip++) if (!dead2[ip]) npart2++;
    it->npart1 = npart1;
    it->npart2 = npart2;

    npy_int64 nalloc;
    if (npart1 >= npart2) {
        it->npairs = npart1;
        it->npairs_not_repeated = npart2;
        nalloc = it->nbuf1 > 0 ? it->nbuf1 : 1;
    } else {
        it->npairs = npart2;
        it->npairs_not_repeated = npart1;
        nalloc = it->nbuf2 > 0 ? it->nbuf2 : 1;
    }
    if (idxbuf_ensure(buf, nalloc) != 0) return -1;
    it->shuffled_idx = buf->data;

    if (npart1 >= npart2) {
        for (npy_int64 i = 0; i < nalloc; i++) it->shuffled_idx[i] = ip_start1 + i;
    } else {
        for (npy_int64 i = 0; i < nalloc; i++) it->shuffled_idx[i] = ip_start2 + i;
    }
    pcg32_shuffle(rng, it->shuffled_idx, nalloc);

    if (npart1 == 0 || npart2 == 0) {
        it->npairs = 0;
        it->npairs_not_repeated = 0;
        it->dt_corr = 0;
    } else {
        it->dt_corr = it->npairs;
    }

    it->ip1 = -1;
    it->ip2 = -1;
    it->ipair = 0;
    return 0;
}

static inline int inter_pairing_next(InterPairing *it, const npy_bool *dead1, const npy_bool *dead2,
                                     npy_int64 *ip1_abs, npy_int64 *ip2_abs,
                                     double *w_corr, double *dt_corr) {
    if (it->ipair >= it->npairs) return 0;

    if (it->npart1 >= it->npart2) {
        /* advance ip1 on shuffled side 1 */
        for (npy_int64 j = it->ip1 + 1; j < it->nbuf1; j++) {
            if (!dead1[it->shuffled_idx[j]]) { it->ip1 = j; break; }
        }
        /* advance ip2 on side 2, cyclic blocks of npart2 */
        if ((it->ipair % it->npart2) == 0) {
            it->ip2 = -1;
        }
        for (npy_int64 j = (it->ip2 + 1) % it->nbuf2; j < it->nbuf2; j++) {
            if (!dead2[it->ip_start2 + j]) { it->ip2 = j; break; }
        }
        *ip1_abs = it->shuffled_idx[it->ip1];
        *ip2_abs = it->ip_start2 + it->ip2;

        if ((it->ipair % it->npairs_not_repeated) < (it->npairs % it->npairs_not_repeated)) {
            *w_corr = 1.0 / (double)(it->npart1 / it->npart2 + 1);
        } else {
            *w_corr = 1.0 / (double)(it->npart1 / it->npart2);
        }
    } else {
        /* advance ip2 on shuffled side 2 */
        for (npy_int64 j = it->ip2 + 1; j < it->nbuf2; j++) {
            if (!dead2[it->shuffled_idx[j]]) { it->ip2 = j; break; }
        }
        /* advance ip1 on side 1, cyclic blocks of npart1 */
        if ((it->ipair % it->npart1) == 0) {
            it->ip1 = -1;
        }
        for (npy_int64 j = (it->ip1 + 1) % it->nbuf1; j < it->nbuf1; j++) {
            if (!dead1[it->ip_start1 + j]) { it->ip1 = j; break; }
        }
        *ip1_abs = it->ip_start1 + it->ip1;
        *ip2_abs = it->shuffled_idx[it->ip2];

        if ((it->ipair % it->npairs_not_repeated) < (it->npairs % it->npairs_not_repeated)) {
            *w_corr = 1.0 / (double)(it->npart2 / it->npart1 + 1);
        } else {
            *w_corr = 1.0 / (double)(it->npart2 / it->npart1);
        }
    }

    *dt_corr = (double)it->dt_corr;
    it->ipair++;
    return 1;
}

/* ------------------------------------------------------------------ */
/* python argument helpers                                             */
/* ------------------------------------------------------------------ */

/* raw-pointer entry points: validate before PyArray_DATA. A strided view
 * or a scalar silently reads garbage (or segfaults), so reject anything
 * that is not a contiguous, aligned, native-endian 1-D array of the
 * expected dtype. */
static inline int check_particle_array(PyObject *obj, int type, const char *name) {
    if (!PyArray_Check(obj)
        || PyArray_TYPE((PyArrayObject*)obj) != type
        || PyArray_NDIM((PyArrayObject*)obj) != 1
        || !PyArray_ISCARRAY((PyArrayObject*)obj)
        || !PyArray_ISNOTSWAPPED((PyArrayObject*)obj)) {
        PyErr_Format(PyExc_TypeError,
                     "%s must be a contiguous 1-D array of dtype %s "
                     "(strided views/scalars read garbage through the raw pointer)",
                     name, type == NPY_DOUBLE ? "float64" : "bool");
        return -1;
    }
    return 0;
}

/* full (npatches, 2) state table [state, inc] per patch, for the
 * patches-level entries. A single (2,) row passed instead is contiguous
 * uint64 and silently reads out of bounds for npatches > 1. */
static inline int check_rng_states(PyObject *obj, npy_intp npatches, const char *name) {
    if (!PyArray_Check(obj)
        || PyArray_TYPE((PyArrayObject*)obj) != NPY_UINT64
        || PyArray_NDIM((PyArrayObject*)obj) != 2
        || PyArray_DIM((PyArrayObject*)obj, 0) != npatches
        || PyArray_DIM((PyArrayObject*)obj, 1) != 2
        || !PyArray_ISCARRAY((PyArrayObject*)obj)
        || !PyArray_ISNOTSWAPPED((PyArrayObject*)obj)) {
        PyErr_Format(PyExc_TypeError,
                     "%s must be a contiguous (%d, 2) uint64 array "
                     "(one PCG32 [state, inc] row per patch)",
                     name, (int)npatches);
        return -1;
    }
    return 0;
}

/* single (2,) state row, for the cell/kernel-level test entries. The whole
 * (n, 2) table, or a (1, 2) array (seed_rng_states(...) without [0]), is
 * contiguous uint64 and silently reads the first row - no error at all. */
static inline int check_rng_row(PyObject *obj) {
    if (!PyArray_Check(obj)
        || PyArray_TYPE((PyArrayObject*)obj) != NPY_UINT64
        || PyArray_NDIM((PyArrayObject*)obj) != 1
        || PyArray_SIZE((PyArrayObject*)obj) != 2
        || !PyArray_ISCARRAY((PyArrayObject*)obj)
        || !PyArray_ISNOTSWAPPED((PyArrayObject*)obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "rng must be a contiguous (2,) uint64 array "
                        "(one PCG32 [state, inc] row: "
                        "seed_rng_states(...) followed by [0])");
        return -1;
    }
    return 0;
}

/* six-array view (momentum-space data; x/y/z stay NULL) */
static inline int parse_particle(PyObject *ux_obj, PyObject *uy_obj, PyObject *uz_obj,
                                  PyObject *ig_obj, PyObject *w_obj, PyObject *dead_obj,
                                  double m, double q, ParticleDataC *part) {
    if (check_particle_array(ux_obj, NPY_DOUBLE, "ux") != 0
        || check_particle_array(uy_obj, NPY_DOUBLE, "uy") != 0
        || check_particle_array(uz_obj, NPY_DOUBLE, "uz") != 0
        || check_particle_array(ig_obj, NPY_DOUBLE, "inv_gamma") != 0
        || check_particle_array(w_obj, NPY_DOUBLE, "w") != 0
        || check_particle_array(dead_obj, NPY_BOOL, "is_dead") != 0) {
        return -1;
    }
    part->x = NULL;
    part->y = NULL;
    part->z = NULL;
    part->ux = (double*)PyArray_DATA((PyArrayObject*)ux_obj);
    part->uy = (double*)PyArray_DATA((PyArrayObject*)uy_obj);
    part->uz = (double*)PyArray_DATA((PyArrayObject*)uz_obj);
    part->inv_gamma = (double*)PyArray_DATA((PyArrayObject*)ig_obj);
    part->w = (double*)PyArray_DATA((PyArrayObject*)w_obj);
    part->is_dead = (npy_bool*)PyArray_DATA((PyArrayObject*)dead_obj);
    part->npart = (npy_int64)PyArray_SIZE((PyArrayObject*)dead_obj);
    part->m = m;
    part->q = q;
    return 0;
}

/* fetch one validated 1-D array attribute from a particle object; the data
 * pointer stays valid because the owning ParticlesBase outlives the call
 * (the module entry point holds the object list). Sets a Python exception
 * (AttributeError/TypeError/MemoryError) and returns -1 on failure. */
static inline int get_attr_particle_array(PyObject *obj, int type, const char *name,
                                          void **out, npy_int64 *npart) {
    PyObject *a = PyObject_GetAttrString(obj, name);
    if (a == NULL) return -1;
    if (check_particle_array(a, type, name) != 0) {
        Py_DECREF(a);
        return -1;
    }
    *out = PyArray_DATA((PyArrayObject*)a);
    if (npart) *npart = (npy_int64)PyArray_SIZE((PyArrayObject*)a);
    Py_DECREF(a);
    return 0;
}

/* fetch per-patch views from a list of particle objects (ParticlesBase or
 * anything exposing the attribute names). Attribute lookup happens per
 * call, so buffers reallocated by extend() are picked up automatically and
 * no view refresh is ever needed on the Python side. */
static inline int prefetch_particle(PyObject *parts_l, double m, double q,
                                    npy_intp npatches, ParticleDataC **parts_out) {
    if (!PyList_Check(parts_l)) {
        PyErr_SetString(PyExc_TypeError, "expected a list of particle objects");
        return -1;
    }
    ParticleDataC *parts = (ParticleDataC*)calloc(npatches, sizeof(ParticleDataC));
    if (parts == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    for (npy_intp i = 0; i < npatches; i++) {
        PyObject *obj = PyList_GET_ITEM(parts_l, i);
        if (get_attr_particle_array(obj, NPY_DOUBLE, "ux", (void**)&parts[i].ux, NULL) != 0
            || get_attr_particle_array(obj, NPY_DOUBLE, "uy", (void**)&parts[i].uy, NULL) != 0
            || get_attr_particle_array(obj, NPY_DOUBLE, "uz", (void**)&parts[i].uz, NULL) != 0
            || get_attr_particle_array(obj, NPY_DOUBLE, "inv_gamma", (void**)&parts[i].inv_gamma, NULL) != 0
            || get_attr_particle_array(obj, NPY_DOUBLE, "w", (void**)&parts[i].w, NULL) != 0
            || get_attr_particle_array(obj, NPY_BOOL, "is_dead", (void**)&parts[i].is_dead, &parts[i].npart) != 0) {
            free(parts);
            return -1;
        }
        parts[i].m = m;
        parts[i].q = q;
    }
    *parts_out = parts;
    return 0;
}

#endif /* LAMBDAPIC_COLLISION_UTILS_H */
