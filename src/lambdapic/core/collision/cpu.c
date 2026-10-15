/*
 * Binary Coulomb collision kernels (C port of the former numba cpu.py).
 *
 * Faithful 1:1 port of the Numba implementation:
 *   - debye length per cell (accumulate per species, reset on first species)
 *   - Debye-length constraint (interparticle distance floor)
 *   - intra/inter-species pairing iterators (shuffle + weight/dt corrections)
 *   - Coulomb scattering in the center-of-mass frame (Nanbu-Perez model,
 *     https://doi.org/10.1063/1.4742167)
 *
 * RNG: PCG32 (http://pcg-random.org), one stream per patch. State is a
 * (npatches, 2) uint64 array [state, inc] per patch row, owned by the
 * Python facade (Collision.rng_state_list). Patch-parallel loops use
 * OpenMP (replacing numba prange); each thread only touches its own
 * patch's stream, so results are independent of thread count.
 *
 * Shared helpers (PCG32, pairing iterators, particle views) live in
 * utils.h.
 */
#include "utils.h"

/* ------------------------------------------------------------------ */
/* physics kernels                                                     */
/* ------------------------------------------------------------------ */

/* Coulomb logarithm, https://doi.org/10.1063/1.4742167 (typo square removed) */
static inline double varying_lnLambda(const CollisionDataC *d, double debye_length_inv_square) {
    const double c = C_LIGHT;
    double q1q2 = fabs(d->q1 * d->q2);

    double b0 = q1q2 / (4.0 * PI * EPSILON_0 * c * c)
        * d->gamma_com / (d->m1 * d->gamma1_com + d->m2 * d->gamma2_com)
        * ((d->m1 * d->gamma1_com * d->m2 * d->gamma2_com) / (d->p1_com * d->p1_com) * c * c + 1.0);
    double bmin = dmax(PLANCK / 2.0 / d->p1_com, b0);

    if (debye_length_inv_square > 0) {
        double lambdaD2 = 1.0 / debye_length_inv_square;
        return dmax(2.0, 0.5 * log(1.0 + lambdaD2 / (bmin * bmin)));
    }
    return 2.0;
}

static inline void rotate_momentum(
    double px, double py, double pz, double p, double p_perp,
    double cosX, double sinX, double phi,
    double *out
) {
    double sin_phi, cos_phi;
    sincos(phi, &sin_phi, &cos_phi); /* one argument reduction instead of two */
    double sinXcosPhi = sinX * cos_phi;
    double sinXsinPhi = sinX * sin_phi;

    if (p_perp > 1.e-10 * p) {
        out[0] = (px * pz * sinXcosPhi - py * p * sinXsinPhi) / p_perp + px * cosX;
        out[1] = (py * pz * sinXcosPhi + px * p * sinXsinPhi) / p_perp + py * cosX;
        out[2] = -p_perp * sinXcosPhi + pz * cosX;
    } else {
        out[0] = p * sinXcosPhi;
        out[1] = p * sinXsinPhi;
        out[2] = p * cosX;
    }
}

/* Coulomb scattering in the COM frame; returns new p1 momentum in COM */
static inline void coulomb_scattering(
    const CollisionDataC *d,
    double cell_vol, double dt, double lnLambda,
    uint64_t *rng,
    double *out
) {
    double px1_com = d->px1_com, py1_com = d->py1_com, pz1_com = d->pz1_com;
    double p1_com = d->p1_com;

    if (p1_com == 0.0) {
        out[0] = px1_com; out[1] = py1_com; out[2] = pz1_com;
        return;
    }

    double w_max = dmax(d->w1, d->w2);
    double q1q2 = d->q1 * d->q2;
    double m1g1 = d->m1 * d->gamma1;
    double m2g2 = d->m2 * d->gamma2;

    double com_term = d->m1 * d->gamma1_com * d->m2 * d->gamma2_com
                    / (p1_com * p1_com) * C_LIGHT * C_LIGHT + 1.0;

    double s = w_max / cell_vol * dt * (lnLambda * q1q2 * q1q2)
        / (4.0 * PI * EPSILON_0 * EPSILON_0 * C_LIGHT * C_LIGHT * C_LIGHT * C_LIGHT * m1g1 * m2g2)
        * (d->gamma_com * p1_com) / (m1g1 + m2g2)
        * com_term * com_term;

    double U = pcg32_double(rng);

    double cosX, sinX;
    if (s < 4.0) {
        double alpha = 0.37 * s - 0.005 * s * s - 0.0064 * s * s * s;
        double sin2X2 = alpha * U / sqrt((1.0 - U) + alpha * alpha * U);
        cosX = 1.0 - 2.0 * sin2X2;
        sinX = 2.0 * sqrt(sin2X2 * (1.0 - sin2X2));
    } else {
        cosX = 2.0 * U - 1.0;
        sinX = sqrt(1.0 - cosX * cosX);
    }

    double phi = pcg32_uniform(rng, 0.0, 2.0 * PI);

    rotate_momentum(px1_com, py1_com, pz1_com, p1_com, d->p_perp, cosX, sinX, phi, out);
}

/* ------------------------------------------------------------------ */
/* collision kernel                                                    */
/* ------------------------------------------------------------------ */

/* by pointer + always_inline: two 72-byte structs by value cost ~144 B of
 * stack traffic per pair call, and gcc's cost model refuses to inline this
 * large kernel on its own (verified: symbol survived -O3). Force-flatten
 * like numba's inline='always' chain. */
static inline __attribute__((always_inline)) void collision_kernel(
    const ParticleDataC *part1, npy_int64 ip1,
    const ParticleDataC *part2, npy_int64 ip2,
    double lnLambda, double debye_length_inv_square,
    double dt_corr, double w_corr,
    double cell_vol, double dt,
    uint64_t *rng
) {
    double w1_corr = part1->w[ip1] * w_corr;
    double w2_corr = part2->w[ip2] * w_corr;
    double w_max = dmax(w1_corr, w2_corr);

    CollisionDataC d;
    /* corrected weights: s must scale with w_corr (matches the pre-port
     * numba collision_kernel, which passed w_corr-ed weights into
     * collision_data); reading the raw buffers would over-deflect by
     * npart_max/npart_min whenever the pairing repeats one side */
    collision_data_ww(*part1, ip1, w1_corr, *part2, ip2, w2_corr, &d);

    /* at rest */
    if (d.p1_com == 0.0) {
        return;
    }

    double lnLambda_;
    if (lnLambda > 0.0) {
        lnLambda_ = lnLambda;
    } else {
        lnLambda_ = varying_lnLambda(&d, debye_length_inv_square);
    }

    double p1_com_new[3];
    coulomb_scattering(&d, cell_vol, dt * dt_corr, lnLambda_, rng, p1_com_new);

    double U = pcg32_double(rng);
    if (w2_corr / w_max > U) {
        double p[3];
        boost_to_lab(p1_com_new[0], p1_com_new[1], p1_com_new[2],
                     d.gamma1_com, d.m1,
                     d.vx_com, d.vy_com, d.vz_com, d.v_com_square, d.gamma_com, p);
        part1->ux[ip1] = p[0] / d.m1 / C_LIGHT;
        part1->uy[ip1] = p[1] / d.m1 / C_LIGHT;
        part1->uz[ip1] = p[2] / d.m1 / C_LIGHT;
        part1->inv_gamma[ip1] = 1.0 / sqrt(part1->ux[ip1] * part1->ux[ip1]
                                        + part1->uy[ip1] * part1->uy[ip1]
                                        + part1->uz[ip1] * part1->uz[ip1] + 1.0);
    }
    if (w1_corr / w_max > U) {
        double p[3];
        boost_to_lab(-p1_com_new[0], -p1_com_new[1], -p1_com_new[2],
                     d.gamma2_com, d.m2,
                     d.vx_com, d.vy_com, d.vz_com, d.v_com_square, d.gamma_com, p);
        part2->ux[ip2] = p[0] / d.m2 / C_LIGHT;
        part2->uy[ip2] = p[1] / d.m2 / C_LIGHT;
        part2->uz[ip2] = p[2] / d.m2 / C_LIGHT;
        part2->inv_gamma[ip2] = 1.0 / sqrt(part2->ux[ip2] * part2->ux[ip2]
                                        + part2->uy[ip2] * part2->uy[ip2]
                                        + part2->uz[ip2] * part2->uz[ip2] + 1.0);
    }
}

/* ------------------------------------------------------------------ */
/* debye length                                                        */
/* ------------------------------------------------------------------ */

static void debye_length_cell(
    ParticleDataC part, npy_int64 ip_start, npy_int64 ip_end,
    double cell_vol,
    double *inv_d2_out, double *density_out
) {
    double w_total = 0.0;
    double ux_mean = 0.0, uy_mean = 0.0, uz_mean = 0.0;
    for (npy_int64 ip = ip_start; ip < ip_end; ip++) {
        if (part.is_dead[ip]) continue;
        w_total += part.w[ip];
        ux_mean += part.w[ip] * part.ux[ip];
        uy_mean += part.w[ip] * part.uy[ip];
        uz_mean += part.w[ip] * part.uz[ip];
    }

    if (w_total <= 0.0) {
        *inv_d2_out = -1.0;
        *density_out = 0.0;
        return;
    }

    double density = w_total / cell_vol;
    ux_mean /= w_total;
    uy_mean /= w_total;
    uz_mean /= w_total;

    double kT_mc2 = 0.0;
    for (npy_int64 ip = ip_start; ip < ip_end; ip++) {
        if (part.is_dead[ip]) continue;
        double dxu = part.ux[ip] - ux_mean;
        double dyu = part.uy[ip] - uy_mean;
        double dzu = part.uz[ip] - uz_mean;
        double u2 = dxu * dxu + dyu * dyu + dzu * dzu;
        /* T = <v*p> / 3 */
        kT_mc2 += part.w[ip] * u2 / sqrt(1.0 + u2) / 3.0;
    }
    kT_mc2 /= w_total;

    double kT = kT_mc2 * part.m * C_LIGHT * C_LIGHT;

    if (kT > 0.0) {
        *inv_d2_out = density * part.q * part.q / (EPSILON_0 * kT);
    } else {
        *inv_d2_out = INFINITY;
    }
    *density_out = density;
}

/* ------------------------------------------------------------------ */
/* cell-level cores (pure C, RNG state = pointer to 2 uint64)           */
/* ------------------------------------------------------------------ */

static int intra_collision_cell_core(
    ParticleDataC part, npy_int64 ip_start, npy_int64 ip_end,
    double lnLambda, double debye_length_inv_square,
    double cell_vol, double dt,
    uint64_t *rng, IdxBuf *buf
) {
    IntraPairing it;
    if (intra_pairing_init(&it, part.is_dead, ip_start, ip_end, rng, buf) != 0) return -1;
    npy_int64 ip1, ip2;
    double w_corr, dt_corr;
    /* local RNG copy: after inlining, gcc keeps the state in registers
     * across the whole cell loop instead of reloading rng[0]/rng[1]
     * from memory on every draw */
    uint64_t rng_local[2] = {rng[0], rng[1]};
    while (intra_pairing_next(&it, part.is_dead, &ip1, &ip2, &w_corr, &dt_corr)) {
        collision_kernel(&part, ip1, &part, ip2,
                         lnLambda, debye_length_inv_square,
                         dt_corr, w_corr, cell_vol, dt, rng_local);
    }
    rng[0] = rng_local[0];
    rng[1] = rng_local[1];
    return 0;
}

static int inter_collision_cell_core(
    ParticleDataC part1, npy_int64 ip_start1, npy_int64 ip_end1,
    ParticleDataC part2, npy_int64 ip_start2, npy_int64 ip_end2,
    double lnLambda, double debye_length_inv_square,
    double cell_vol, double dt,
    uint64_t *rng, IdxBuf *buf
) {
    InterPairing it;
    if (inter_pairing_init(&it, part1.is_dead, ip_start1, ip_end1,
                           part2.is_dead, ip_start2, ip_end2, rng, buf) != 0) return -1;
    npy_int64 ip1, ip2;
    double w_corr, dt_corr;
    /* local RNG copy, same rationale as the intra core */
    uint64_t rng_local[2] = {rng[0], rng[1]};
    while (inter_pairing_next(&it, part1.is_dead, part2.is_dead, &ip1, &ip2, &w_corr, &dt_corr)) {
        collision_kernel(&part1, ip1, &part2, ip2,
                         lnLambda, debye_length_inv_square,
                         dt_corr, w_corr, cell_vol, dt, rng_local);
    }
    rng[0] = rng_local[0];
    rng[1] = rng_local[1];
    return 0;
}

/* ------------------------------------------------------------------ */
/* module functions                                                    */
/* ------------------------------------------------------------------ */

static PyObject* seed_rng_states(PyObject* self, PyObject* args) {
    PyObject *seeds_obj;
    if (!PyArg_ParseTuple(args, "O", &seeds_obj)) return NULL;

    PyArrayObject *seeds = (PyArrayObject*)PyArray_FROMANY(seeds_obj, NPY_INT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (seeds == NULL) return NULL;

    npy_intp n = PyArray_DIM(seeds, 0);
    npy_intp dims[2] = {n, 2};
    PyArrayObject *out = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_UINT64);
    if (out == NULL) { Py_DECREF(seeds); return NULL; }

    uint64_t *states = (uint64_t*)PyArray_DATA(out);
    npy_int64 *s = (npy_int64*)PyArray_DATA(seeds);
    for (npy_intp i = 0; i < n; i++) {
        uint64_t seed = (uint64_t)s[i];
        pcg32_seed(&states[2 * i], seed, seed);
    }

    Py_DECREF(seeds);
    return (PyObject*)out;
}

static PyObject* debye_length_cell_py(PyObject* self, PyObject* args) {
    PyObject *ux_obj, *uy_obj, *uz_obj, *ig_obj, *w_obj, *dead_obj;
    double m, q, cell_vol;
    npy_int64 ip_start, ip_end;

    if (!PyArg_ParseTuple(args, "OOOOOOddnnd",
                          &ux_obj, &uy_obj, &uz_obj, &ig_obj, &w_obj, &dead_obj,
                          &m, &q, &ip_start, &ip_end, &cell_vol)) {
        return NULL;
    }

    ParticleDataC part;
    if (parse_particle(ux_obj, uy_obj, uz_obj, ig_obj, w_obj, dead_obj, m, q, &part) != 0) {
        return NULL;
    }

    double inv_d2, density;
    debye_length_cell(part, ip_start, ip_end, cell_vol, &inv_d2, &density);
    return Py_BuildValue("(dd)", inv_d2, density);
}

static PyObject* debye_length_patch_py(PyObject* self, PyObject* args) {
    PyObject *ux_obj, *uy_obj, *uz_obj, *ig_obj, *w_obj, *dead_obj;
    PyObject *bmin_obj, *bmax_obj, *debye_obj;
    double m, q, cell_vol;

    if (!PyArg_ParseTuple(args, "OOOOOOddOOdO",
                          &ux_obj, &uy_obj, &uz_obj, &ig_obj, &w_obj, &dead_obj,
                          &m, &q,
                          &bmin_obj, &bmax_obj,
                          &cell_vol,
                          &debye_obj)) {
        return NULL;
    }

    ParticleDataC part;
    if (parse_particle(ux_obj, uy_obj, uz_obj, ig_obj, w_obj, dead_obj, m, q, &part) != 0) {
        return NULL;
    }
    npy_int64 *bmin = (npy_int64*)PyArray_DATA((PyArrayObject*)bmin_obj);
    npy_int64 *bmax = (npy_int64*)PyArray_DATA((PyArrayObject*)bmax_obj);
    double *debye = (double*)PyArray_DATA((PyArrayObject*)debye_obj);
    npy_intp ncells = PyArray_SIZE((PyArrayObject*)bmin_obj);

    for (npy_intp icell = 0; icell < ncells; icell++) {
        double inv_d2, n;
        debye_length_cell(part, bmin[icell], bmax[icell], cell_vol, &inv_d2, &n);
        if (n > 0) {
            debye[icell] += inv_d2;
        }
    }
    Py_RETURN_NONE;
}

static PyObject* debye_length_patches(PyObject* self, PyObject* args) {
    PyObject *parts_l;
    PyObject *bmin_l, *bmax_l, *debye_l, *total_l;
    double m, q, cell_vol;
    int reset;

    if (!PyArg_ParseTuple(args, "OddOOdOOi",
                          &parts_l, &m, &q,
                          &bmin_l, &bmax_l,
                          &cell_vol,
                          &debye_l, &total_l,
                          &reset)) {
        return NULL;
    }

    npy_intp npatches = PyList_Size(parts_l);
    if (npatches < 0) return NULL;
    ParticleDataC *parts;
    if (prefetch_particle(parts_l, m, q, npatches, &parts) != 0) {
        return NULL;
    }
    double **debye = (double**)malloc(npatches * sizeof(double*));
    double **total = (double**)malloc(npatches * sizeof(double*));
    npy_int64 **bmin = (npy_int64**)malloc(npatches * sizeof(npy_int64*));
    npy_int64 **bmax = (npy_int64**)malloc(npatches * sizeof(npy_int64*));
    npy_intp *ncells = (npy_intp*)malloc(npatches * sizeof(npy_intp));
    if (!debye || !total || !bmin || !bmax || !ncells) {
        free(debye); free(total); free(bmin); free(bmax); free(ncells); free(parts);
        return PyErr_NoMemory();
    }
    for (npy_intp i = 0; i < npatches; i++) {
        debye[i] = (double*)GetPatchArrayData(debye_l, i);
        total[i] = (double*)GetPatchArrayData(total_l, i);
        bmin[i] = (npy_int64*)GetPatchArrayData(bmin_l, i);
        bmax[i] = (npy_int64*)GetPatchArrayData(bmax_l, i);
        ncells[i] = PyArray_SIZE((PyArrayObject*)PyList_GetItem(bmin_l, i));
    }

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        if (reset) {
            memset(debye[ipatch], 0, ncells[ipatch] * sizeof(double));
        }
        for (npy_intp icell = 0; icell < ncells[ipatch]; icell++) {
            npy_int64 ip_start = bmin[ipatch][icell];
            npy_int64 ip_end = bmax[ipatch][icell];
            double inv_d2, n;
            debye_length_cell(parts[ipatch], ip_start, ip_end, cell_vol, &inv_d2, &n);
            if (n > 0) {
                debye[ipatch][icell] += inv_d2;
                total[ipatch][icell] += n;
            }
        }
    }
    Py_END_ALLOW_THREADS

    free(debye); free(total); free(bmin); free(bmax); free(ncells); free(parts);
    Py_RETURN_NONE;
}

static PyObject* constrain_debye_length_patches(PyObject* self, PyObject* args) {
    PyObject *debye_l, *total_l;
    if (!PyArg_ParseTuple(args, "OO", &debye_l, &total_l)) return NULL;

    npy_intp npatches = PyList_Size(debye_l);
    double **debye = (double**)malloc(npatches * sizeof(double*));
    double **total = (double**)malloc(npatches * sizeof(double*));
    npy_intp *ncells = (npy_intp*)malloc(npatches * sizeof(npy_intp));
    if (!debye || !total || !ncells) {
        free(debye); free(total); free(ncells);
        return PyErr_NoMemory();
    }
    for (npy_intp i = 0; i < npatches; i++) {
        debye[i] = (double*)GetPatchArrayData(debye_l, i);
        total[i] = (double*)GetPatchArrayData(total_l, i);
        ncells[i] = PyArray_SIZE((PyArrayObject*)PyList_GetItem(debye_l, i));
    }

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        for (npy_intp icell = 0; icell < ncells[ipatch]; icell++) {
            double inv_d2 = debye[ipatch][icell];
            if (inv_d2 <= 0) {
                /* will be handled in varying_lnLambda */
                continue;
            }
            double d2 = 1.0 / inv_d2;
            double nmax = total[ipatch][icell];
            double rmin2 = pow(4.0 * PI * nmax / 3.0, -2.0 / 3.0);
            if (d2 < rmin2) {
                debye[ipatch][icell] = 1.0 / rmin2;
            }
        }
    }
    Py_END_ALLOW_THREADS

    free(debye); free(total); free(ncells);
    Py_RETURN_NONE;
}

static PyObject* intra_collision_patches(PyObject* self, PyObject* args) {
    PyObject *parts_l;
    PyObject *bmin_l, *bmax_l, *debye_l, *rng_l;
    double m, q, lnLambda, cell_vol, dt;

    if (!PyArg_ParseTuple(args, "OddOOdOddO",
                          &parts_l, &m, &q,
                          &bmin_l, &bmax_l,
                          &lnLambda, &debye_l, &cell_vol, &dt,
                          &rng_l)) {
        return NULL;
    }

    npy_intp npatches = PyList_Size(parts_l);
    if (npatches < 0) return NULL;
    if (check_rng_states(rng_l, npatches, "rng_state_list") != 0) return NULL;
    ParticleDataC *parts;
    if (prefetch_particle(parts_l, m, q, npatches, &parts) != 0) {
        return NULL;
    }
    double **debye = (double**)malloc(npatches * sizeof(double*));
    npy_int64 **bmin = (npy_int64**)malloc(npatches * sizeof(npy_int64*));
    npy_int64 **bmax = (npy_int64**)malloc(npatches * sizeof(npy_int64*));
    npy_intp *ncells = (npy_intp*)malloc(npatches * sizeof(npy_intp));
    uint64_t *rng = (uint64_t*)PyArray_DATA((PyArrayObject*)rng_l);
    if (!debye || !bmin || !bmax || !ncells) {
        free(debye); free(bmin); free(bmax); free(ncells); free(parts);
        return PyErr_NoMemory();
    }
    for (npy_intp i = 0; i < npatches; i++) {
        debye[i] = (double*)GetPatchArrayData(debye_l, i);
        bmin[i] = (npy_int64*)GetPatchArrayData(bmin_l, i);
        bmax[i] = (npy_int64*)GetPatchArrayData(bmax_l, i);
        ncells[i] = PyArray_SIZE((PyArrayObject*)PyList_GetItem(bmin_l, i));
    }

    int failed = 0;
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel
    {
        IdxBuf buf = {NULL, 0};
        #pragma omp for reduction(|:failed)
        for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
            for (npy_intp icell = 0; icell < ncells[ipatch]; icell++) {
                if (intra_collision_cell_core(parts[ipatch],
                                              bmin[ipatch][icell], bmax[ipatch][icell],
                                              lnLambda, debye[ipatch][icell],
                                              cell_vol, dt,
                                              &rng[2 * ipatch], &buf) != 0) {
                    failed = 1;
                }
            }
        }
        free(buf.data);
    }
    Py_END_ALLOW_THREADS

    free(debye); free(bmin); free(bmax); free(ncells); free(parts);
    if (failed) return PyErr_NoMemory();
    Py_RETURN_NONE;
}

static PyObject* inter_collision_patches(PyObject* self, PyObject* args) {
    PyObject *parts1_l, *bmin1_l, *bmax1_l;
    PyObject *parts2_l, *bmin2_l, *bmax2_l;
    PyObject *debye_l, *rng_l;
    double m1, q1, m2, q2, lnLambda, cell_vol, dt;

    if (!PyArg_ParseTuple(args, "OddOOOddOOdOddO",
                          &parts1_l, &m1, &q1,
                          &bmin1_l, &bmax1_l,
                          &parts2_l, &m2, &q2,
                          &bmin2_l, &bmax2_l,
                          &lnLambda, &debye_l, &cell_vol, &dt,
                          &rng_l)) {
        return NULL;
    }

    npy_intp npatches = PyList_Size(parts1_l);
    if (npatches < 0) return NULL;
    if (check_rng_states(rng_l, npatches, "rng_state_list") != 0) return NULL;
    ParticleDataC *parts1, *parts2;
    if (prefetch_particle(parts1_l, m1, q1, npatches, &parts1) != 0) {
        return NULL;
    }
    if (prefetch_particle(parts2_l, m2, q2, npatches, &parts2) != 0) {
        free(parts1);
        return NULL;
    }
    double **debye = (double**)malloc(npatches * sizeof(double*));
    npy_int64 **bmin1 = (npy_int64**)malloc(npatches * sizeof(npy_int64*));
    npy_int64 **bmax1 = (npy_int64**)malloc(npatches * sizeof(npy_int64*));
    npy_int64 **bmin2 = (npy_int64**)malloc(npatches * sizeof(npy_int64*));
    npy_int64 **bmax2 = (npy_int64**)malloc(npatches * sizeof(npy_int64*));
    npy_intp *ncells = (npy_intp*)malloc(npatches * sizeof(npy_intp));
    uint64_t *rng = (uint64_t*)PyArray_DATA((PyArrayObject*)rng_l);
    if (!debye || !bmin1 || !bmax1 || !bmin2 || !bmax2 || !ncells) {
        free(debye); free(bmin1); free(bmax1); free(bmin2); free(bmax2); free(ncells);
        free(parts1); free(parts2);
        return PyErr_NoMemory();
    }
    for (npy_intp i = 0; i < npatches; i++) {
        debye[i] = (double*)GetPatchArrayData(debye_l, i);
        bmin1[i] = (npy_int64*)GetPatchArrayData(bmin1_l, i);
        bmax1[i] = (npy_int64*)GetPatchArrayData(bmax1_l, i);
        bmin2[i] = (npy_int64*)GetPatchArrayData(bmin2_l, i);
        bmax2[i] = (npy_int64*)GetPatchArrayData(bmax2_l, i);
        ncells[i] = PyArray_SIZE((PyArrayObject*)PyList_GetItem(bmin1_l, i));
    }

    int failed = 0;
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel
    {
        IdxBuf buf = {NULL, 0};
        #pragma omp for reduction(|:failed)
        for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
            for (npy_intp icell = 0; icell < ncells[ipatch]; icell++) {
                if (inter_collision_cell_core(parts1[ipatch], bmin1[ipatch][icell], bmax1[ipatch][icell],
                                              parts2[ipatch], bmin2[ipatch][icell], bmax2[ipatch][icell],
                                              lnLambda, debye[ipatch][icell],
                                              cell_vol, dt,
                                              &rng[2 * ipatch], &buf) != 0) {
                    failed = 1;
                }
            }
        }
        free(buf.data);
    }
    Py_END_ALLOW_THREADS

    free(debye); free(bmin1); free(bmax1); free(bmin2); free(bmax2); free(ncells);
    free(parts1); free(parts2);
    if (failed) return PyErr_NoMemory();
    Py_RETURN_NONE;
}

static PyObject* intra_collision_cell_py(PyObject* self, PyObject* args) {
    PyObject *ux_obj, *uy_obj, *uz_obj, *ig_obj, *w_obj, *dead_obj, *rng_obj;
    double m, q, lnLambda, debye_inv2, cell_vol, dt;
    npy_int64 ip_start, ip_end;

    if (!PyArg_ParseTuple(args, "OOOOOOddnnddddO",
                          &ux_obj, &uy_obj, &uz_obj, &ig_obj, &w_obj, &dead_obj,
                          &m, &q, &ip_start, &ip_end,
                          &lnLambda, &debye_inv2, &cell_vol, &dt,
                          &rng_obj)) {
        return NULL;
    }
    if (check_rng_row(rng_obj) != 0) return NULL;

    ParticleDataC part;
    if (parse_particle(ux_obj, uy_obj, uz_obj, ig_obj, w_obj, dead_obj, m, q, &part) != 0) {
        return NULL;
    }
    uint64_t *rng = (uint64_t*)PyArray_DATA((PyArrayObject*)rng_obj);

    IdxBuf buf = {NULL, 0};
    int rc = intra_collision_cell_core(part, ip_start, ip_end,
                                       lnLambda, debye_inv2, cell_vol, dt, rng, &buf);
    free(buf.data);
    if (rc != 0) return PyErr_NoMemory();
    Py_RETURN_NONE;
}

static PyObject* inter_collision_cell_py(PyObject* self, PyObject* args) {
    PyObject *ux1_obj, *uy1_obj, *uz1_obj, *ig1_obj, *w1_obj, *dead1_obj, *rng_obj;
    PyObject *ux2_obj, *uy2_obj, *uz2_obj, *ig2_obj, *w2_obj, *dead2_obj;
    double m1, q1, m2, q2, lnLambda, debye_inv2, cell_vol, dt;
    npy_int64 ip_start1, ip_end1, ip_start2, ip_end2;

    if (!PyArg_ParseTuple(args, "OOOOOOddnnOOOOOOddnnddddO",
                          &ux1_obj, &uy1_obj, &uz1_obj, &ig1_obj, &w1_obj, &dead1_obj,
                          &m1, &q1, &ip_start1, &ip_end1,
                          &ux2_obj, &uy2_obj, &uz2_obj, &ig2_obj, &w2_obj, &dead2_obj,
                          &m2, &q2, &ip_start2, &ip_end2,
                          &lnLambda, &debye_inv2, &cell_vol, &dt,
                          &rng_obj)) {
        return NULL;
    }
    if (check_rng_row(rng_obj) != 0) return NULL;

    ParticleDataC part1, part2;
    if (parse_particle(ux1_obj, uy1_obj, uz1_obj, ig1_obj, w1_obj, dead1_obj, m1, q1, &part1) != 0
        || parse_particle(ux2_obj, uy2_obj, uz2_obj, ig2_obj, w2_obj, dead2_obj, m2, q2, &part2) != 0) {
        return NULL;
    }
    uint64_t *rng = (uint64_t*)PyArray_DATA((PyArrayObject*)rng_obj);

    IdxBuf buf = {NULL, 0};
    int rc = inter_collision_cell_core(part1, ip_start1, ip_end1,
                                       part2, ip_start2, ip_end2,
                                       lnLambda, debye_inv2, cell_vol, dt, rng, &buf);
    free(buf.data);
    if (rc != 0) return PyErr_NoMemory();
    Py_RETURN_NONE;
}

static PyMethodDef CollisionMethods[] = {
    {"seed_rng_states", seed_rng_states, METH_VARARGS,
     "seed_rng_states(seeds: int64[n]) -> uint64[n, 2]\n\nPCG32 states [state, inc] per seed."},
    {"debye_length_cell", debye_length_cell_py, METH_VARARGS,
     "debye_length_cell(ux, uy, uz, inv_gamma, w, is_dead, m, q, ip_start, ip_end, cell_vol) -> (inv_debye_square, density)"},
    {"debye_length_patch", debye_length_patch_py, METH_VARARGS,
     "debye_length_patch(ux, uy, uz, inv_gamma, w, is_dead, m, q, bucket_bound_min, bucket_bound_max, cell_vol, debye_inv2)"},
    {"debye_length_patches", debye_length_patches, METH_VARARGS,
     "debye_length_patches(particles_list, m, q, bucket_bound_min_list, bucket_bound_max_list, cell_vol, debye_inv2_list, total_density_list, reset)\n\nparticles_list: list of ParticlesBase (attributes fetched per call)."},
    {"constrain_debye_length_patches", constrain_debye_length_patches, METH_VARARGS,
     "constrain_debye_length_patches(debye_inv2_list, total_density_list)"},
    {"intra_collision_patches", intra_collision_patches, METH_VARARGS,
     "intra_collision_patches(particles_list, m, q, bucket_bound_min_list, bucket_bound_max_list, lnLambda, debye_inv2_list, cell_vol, dt, rng_state_list)"},
    {"inter_collision_patches", inter_collision_patches, METH_VARARGS,
     "inter_collision_patches(particles1_list, m1, q1, bounds1, particles2_list, m2, q2, bounds2, lnLambda, debye_inv2_list, cell_vol, dt, rng_state_list)"},
    {"intra_collision_cell", intra_collision_cell_py, METH_VARARGS,
     "intra_collision_cell(ux, uy, uz, inv_gamma, w, is_dead, m, q, ip_start, ip_end, lnLambda, debye_inv2, cell_vol, dt, rng)"},
    {"inter_collision_cell", inter_collision_cell_py, METH_VARARGS,
     "inter_collision_cell(lists1..., m1, q1, ip_start1, ip_end1, lists2..., m2, q2, ip_start2, ip_end2, lnLambda, debye_inv2, cell_vol, dt, rng)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef collisionmodule = {
    PyModuleDef_HEAD_INIT,
    "cpu",
    "Binary collision kernels (C port)",
    -1,
    CollisionMethods
};

PyMODINIT_FUNC PyInit_cpu(void) {
    import_array();
    return PyModule_Create(&collisionmodule);
}
