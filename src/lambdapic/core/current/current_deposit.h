#ifndef CURRENT_DEPOSIT_H
#define CURRENT_DEPOSIT_H

#include <math.h>
#include "../utils/cutils.h"

static inline void calculate_S(double delta, int shift, double* S) {
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

// Specialized shape factors for shift == 0 (S0 always, and S1 when the
// particle does not cross a cell boundary, which is the common case).
// Produces bitwise-identical results to calculate_S(delta, 0, S).
static inline void calculate_S0(double delta, double* S) {
    double delta2 = delta * delta;
    S[0] = 0.0;
    S[1] = 0.5 * (delta2 + delta + 0.25);
    S[2] = 0.75 - delta2;
    S[3] = 0.5 * (delta2 - delta + 0.25);
    S[4] = 0.0;
}

// Precompute the 5 wrapped stencil indices around i0, keeping the wrap
// branch out of the inner deposition loops. The while loops cost no more
// than a single if for in-range indices and wrap robustly for arbitrarily
// far out-of-range particles.
#define PRECOMPUTE_WRAP_INDICES(idxs, i0, n)            \
    do {                                                \
        for (int _i = 0; _i < 5; _i++) {                \
            int _idx = (i0) + (_i - 2);                 \
            while (_idx < 0) _idx += (n);               \
            while (_idx >= (n)) _idx -= (n);            \
            (idxs)[_i] = _idx;                          \
        }                                               \
    } while (0)

__attribute__((always_inline)) inline static void current_deposit_2d(
    double* restrict rho, double* restrict jx, double* restrict jy, double* restrict jz,
    double x, double y,
    double ux, double uy, double uz, double inv_gamma,
    npy_intp nx, npy_intp ny,
    double dx, double dy, double x0, double y0, double dt, double w, double q
) {
    double vx = ux * LIGHT_SPEED * inv_gamma;
    double vy = uy * LIGHT_SPEED * inv_gamma;
    double vz = uz * LIGHT_SPEED * inv_gamma;
    double x_old = x - vx * 0.5 * dt - x0;
    double y_old = y - vy * 0.5 * dt - y0;
    double x_adv = x + vx * 0.5 * dt - x0;
    double y_adv = y + vy * 0.5 * dt - y0;

    double x_over_dx0 = x_old / dx;
    double y_over_dy0 = y_old / dy;
    double x_over_dx1 = x_adv / dx;
    double y_over_dy1 = y_adv / dy;

    int ix0, iy0, ix1, iy1;
    // Fast path: for non-negative coordinates, (int) is equivalent to floor
    if (x_old >= 0 && x_adv >= 0 && y_old >= 0 && y_adv >= 0) {
        ix0 = (int)(x_over_dx0 + 0.5);
        iy0 = (int)(y_over_dy0 + 0.5);
        ix1 = (int)(x_over_dx1 + 0.5);
        iy1 = (int)(y_over_dy1 + 0.5);
    } else {
        ix0 = (int)floor(x_over_dx0 + 0.5);
        iy0 = (int)floor(y_over_dy0 + 0.5);
        ix1 = (int)floor(x_over_dx1 + 0.5);
        iy1 = (int)floor(y_over_dy1 + 0.5);
    }

    double S0x[5], S0y[5];
    calculate_S0(ix0 - x_over_dx0, S0x);
    calculate_S0(iy0 - y_over_dy0, S0y);

    int dcell_x = ix1 - ix0;
    int dcell_y = iy1 - iy0;

    double S1x[5], S1y[5], DSx[5], DSy[5];
    if (dcell_x == 0) calculate_S0(ix1 - x_over_dx1, S1x);
    else              calculate_S(ix1 - x_over_dx1, dcell_x, S1x);
    if (dcell_y == 0) calculate_S0(iy1 - y_over_dy1, S1y);
    else              calculate_S(iy1 - y_over_dy1, dcell_y, S1y);

    npy_intp i, j;
    for (i = 0; i < 5; i++) {
        DSx[i] = S1x[i] - S0x[i];
        DSy[i] = S1y[i] - S0y[i];
    }

    double charge_density = q * w / (dx * dy);
    double factor = charge_density / dt;
    double factor_dx = factor * dx;
    double factor_dy = factor * dy;
    double factor_dt_vz = factor * dt * vz;
    const double one_twelfth = 1.0 / 12.0;

    // These bounds are correct only when a particle crosses at most one cell
    // per step (|dcell| <= 1), which is the usual PIC CFL condition.
    int i_start = dcell_x < 0 ? 0 : 1;
    int i_end = dcell_x > 0 ? 5 : 4;
    int j_start = dcell_y < 0 ? 0 : 1;
    int j_end = dcell_y > 0 ? 5 : 4;

    int ixs[5], iys[5];
    PRECOMPUTE_WRAP_INDICES(ixs, ix0, nx);
    PRECOMPUTE_WRAP_INDICES(iys, iy0, ny);

    double jx_buff[5] = {0, 0, 0, 0, 0};

    for (i = i_start; i < i_end; i++) {
        double jy_buff = 0.0;
        npy_intp ix_ny = (npy_intp)ixs[i] * ny;
        double a = S0x[i] + 0.5 * DSx[i];
        double factor_dx_DSx_i = factor_dx * DSx[i];
        double one_twelfth_DSx_i = one_twelfth * DSx[i];
        for (j = j_start; j < j_end; j++) {
            double b = S0y[j] + 0.5 * DSy[j];
            double wy = DSy[j] * a;
            double wz = a * b + one_twelfth_DSx_i * DSy[j];

            jx_buff[j] -= factor_dx_DSx_i * b;
            jy_buff -= factor_dy * wy;

            npy_intp idx = iys[j] + ix_ny;
            jx[idx] += jx_buff[j];
            jy[idx] += jy_buff;
            jz[idx] += factor_dt_vz * wz;
            rho[idx] += charge_density * S1x[i] * S1y[j];
        }
    }
}

// Cell-deposition loop nest for current_deposit_2d_fast, factored out so the
// common no-cell-crossing case can be instantiated with compile-time constant
// bounds (fully unrolled by the compiler) without duplicating the body.
__attribute__((always_inline)) inline static void current_deposit_2d_fast_cells(
    double* restrict rho, double* restrict jx, double* restrict jy, double* restrict jz,
    const double* restrict S0x, const double* restrict S0y,
    const double* restrict S1x, const double* restrict S1y,
    const double* restrict DSx, const double* restrict DSy,
    const int* restrict ixs, const int* restrict iys, npy_intp ny,
    double charge_density, double factor_dx, double factor_dy, double factor_dt_vz,
    int i_start, int i_end, int j_start, int j_end
) {
    const double one_twelfth = 1.0 / 12.0;
    double jx_buff[5] = {0, 0, 0, 0, 0};

    for (int i = i_start; i < i_end; i++) {
        double jy_buff = 0.0;
        npy_intp ix_ny = (npy_intp)ixs[i] * ny;
        double a = S0x[i] + 0.5 * DSx[i];
        double factor_dx_DSx_i = factor_dx * DSx[i];
        double one_twelfth_DSx_i = one_twelfth * DSx[i];
        for (int j = j_start; j < j_end; j++) {
            double b = S0y[j] + 0.5 * DSy[j];
            double wy = DSy[j] * a;
            double wz = a * b + one_twelfth_DSx_i * DSy[j];

            jx_buff[j] -= factor_dx_DSx_i * b;
            jy_buff -= factor_dy * wy;

            npy_intp idx = iys[j] + ix_ny;
            jx[idx] += jx_buff[j];
            jy[idx] += jy_buff;
            jz[idx] += factor_dt_vz * wz;
            rho[idx] += charge_density * S1x[i] * S1y[j];
        }
    }
}

__attribute__((always_inline)) inline static void current_deposit_2d_fast(
    double* restrict rho, double* restrict jx, double* restrict jy, double* restrict jz,
    double x, double y,
    double ux, double uy, double uz, double inv_gamma,
    npy_intp nx, npy_intp ny,
    double dx, double dy, double x0, double y0, double dt, double w,
    double q_over_dx_dy, double q_over_dy_dt, double q_over_dx_dt
) {
    double vx = ux * LIGHT_SPEED * inv_gamma;
    double vy = uy * LIGHT_SPEED * inv_gamma;
    double vz = uz * LIGHT_SPEED * inv_gamma;
    double x_old = x - vx * 0.5 * dt - x0;
    double y_old = y - vy * 0.5 * dt - y0;
    double x_adv = x + vx * 0.5 * dt - x0;
    double y_adv = y + vy * 0.5 * dt - y0;

    double x_over_dx0 = x_old / dx;
    double y_over_dy0 = y_old / dy;
    double x_over_dx1 = x_adv / dx;
    double y_over_dy1 = y_adv / dy;

    int ix0, iy0, ix1, iy1;
    if (x_old >= 0 && x_adv >= 0 && y_old >= 0 && y_adv >= 0) {
        ix0 = (int)(x_over_dx0 + 0.5);
        iy0 = (int)(y_over_dy0 + 0.5);
        ix1 = (int)(x_over_dx1 + 0.5);
        iy1 = (int)(y_over_dy1 + 0.5);
    } else {
        ix0 = (int)floor(x_over_dx0 + 0.5);
        iy0 = (int)floor(y_over_dy0 + 0.5);
        ix1 = (int)floor(x_over_dx1 + 0.5);
        iy1 = (int)floor(y_over_dy1 + 0.5);
    }

    double S0x[5], S0y[5];
    calculate_S0(ix0 - x_over_dx0, S0x);
    calculate_S0(iy0 - y_over_dy0, S0y);

    int dcell_x = ix1 - ix0;
    int dcell_y = iy1 - iy0;

    double S1x[5], S1y[5], DSx[5], DSy[5];
    if (dcell_x == 0) calculate_S0(ix1 - x_over_dx1, S1x);
    else              calculate_S(ix1 - x_over_dx1, dcell_x, S1x);
    if (dcell_y == 0) calculate_S0(iy1 - y_over_dy1, S1y);
    else              calculate_S(iy1 - y_over_dy1, dcell_y, S1y);

    npy_intp i, j;
    for (i = 0; i < 5; i++) {
        DSx[i] = S1x[i] - S0x[i];
        DSy[i] = S1y[i] - S0y[i];
    }

    double charge_density = q_over_dx_dy * w;
    double factor_dx = q_over_dy_dt * w;
    double factor_dy = q_over_dx_dt * w;
    double factor_dt_vz = charge_density * vz;
    const double one_twelfth = 1.0 / 12.0;

    // These bounds are correct only when a particle crosses at most one cell
    // per step (|dcell| <= 1), which is the usual PIC CFL condition.
    int i_start = dcell_x < 0 ? 0 : 1;
    int i_end = dcell_x > 0 ? 5 : 4;
    int j_start = dcell_y < 0 ? 0 : 1;
    int j_end = dcell_y > 0 ? 5 : 4;

    int ixs[5], iys[5];
    PRECOMPUTE_WRAP_INDICES(ixs, ix0, nx);
    PRECOMPUTE_WRAP_INDICES(iys, iy0, ny);

    if (dcell_x == 0 && dcell_y == 0) {
        // Common case (no cell crossing): compile-time constant trip counts.
        current_deposit_2d_fast_cells(
            rho, jx, jy, jz, S0x, S0y, S1x, S1y, DSx, DSy, ixs, iys, ny,
            charge_density, factor_dx, factor_dy, factor_dt_vz, 1, 4, 1, 4
        );
    } else {
        current_deposit_2d_fast_cells(
            rho, jx, jy, jz, S0x, S0y, S1x, S1y, DSx, DSy, ixs, iys, ny,
            charge_density, factor_dx, factor_dy, factor_dt_vz,
            i_start, i_end, j_start, j_end
        );
    }
}

// Cell-deposition loop nest for current_deposit_3d_fast, factored out so the
// common no-cell-crossing case can be instantiated with compile-time constant
// bounds (fully unrolled by the compiler) without duplicating the body.
// ix0w/iy0w/iz0w must already be wrapped into [0, n); the per-cell single
// wrap below then suffices for any particle position.
__attribute__((always_inline)) inline static void current_deposit_3d_fast_cells(
    double* restrict rho, double* restrict jx, double* restrict jy, double* restrict jz,
    const double* restrict S0x, const double* restrict S0y, const double* restrict S0z,
    const double* restrict S1x, const double* restrict S1y, const double* restrict S1z,
    const double* restrict DSx, const double* restrict DSy, const double* restrict DSz,
    int ix0w, int iy0w, int iz0w,
    npy_intp nx, npy_intp ny, npy_intp nz,
    double charge_density, double factor_dx, double factor_dy, double factor_dz,
    int i_start, int i_end, int j_start, int j_end, int k_start, int k_end
) {
    double jx_buff[5][5] = {{0}};
    npy_intp ny_nz = ny * nz;

    for (int i = i_start; i < i_end; i++) {
        int ix = ix0w + (i - 2);
        if (ix < 0) ix += nx;
        else if (ix >= nx) ix -= nx;

        double a_x = S0x[i] + 0.5 * DSx[i];
        double c_x = 0.5 * S0x[i] + one_third * DSx[i];
        double factor_dx_DSx_i = factor_dx * DSx[i];

        double jy_buff[5] = {0};

        for (int j = j_start; j < j_end; j++) {
            int iy = iy0w + (j - 2);
            if (iy < 0) iy += ny;
            else if (iy >= ny) iy -= ny;

            double a_y = S0y[j] + 0.5 * DSy[j];
            double c_y = 0.5 * S0y[j] + one_third * DSy[j];
            double factor_dy_DSy_j = factor_dy * DSy[j];
            double term_jz_ij = a_x * S0y[j] + c_x * DSy[j];

            double jz_buff = 0;

            for (int k = k_start; k < k_end; k++) {
                int iz = iz0w + (k - 2);
                if (iz < 0) iz += nz;
                else if (iz >= nz) iz -= nz;

                double term_jx = a_y * S0z[k] + c_y * DSz[k];
                double term_jy = a_x * S0z[k] + c_x * DSz[k];

                jx_buff[k][j] -= factor_dx_DSx_i * term_jx;
                jy_buff[k] -= factor_dy_DSy_j * term_jy;
                jz_buff -= factor_dz * DSz[k] * term_jz_ij;

                npy_intp idx = iz + iy * nz + ix * ny_nz;
                jx[idx] += jx_buff[k][j];
                jy[idx] += jy_buff[k];
                jz[idx] += jz_buff;
                rho[idx] += charge_density * S1x[i] * S1y[j] * S1z[k];
            }
        }
    }
}

__attribute__((always_inline)) inline static void current_deposit_3d_fast(
    double* restrict rho, double* restrict jx, double* restrict jy, double* restrict jz,
    double x, double y, double z,
    double ux, double uy, double uz, double inv_gamma,
    npy_intp nx, npy_intp ny, npy_intp nz,
    double dx, double dy, double dz, double x0, double y0, double z0, double dt, double w,
    double q_over_dx_dy_dz, double q_over_dy_dz_dt, double q_over_dx_dz_dt, double q_over_dx_dy_dt
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
    double y_over_dy0 = y_old / dy;
    double z_over_dz0 = z_old / dz;
    double x_over_dx1 = x_adv / dx;
    double y_over_dy1 = y_adv / dy;
    double z_over_dz1 = z_adv / dz;

    int ix0, iy0, iz0, ix1, iy1, iz1;
    if (x_old >= 0 && x_adv >= 0 && y_old >= 0 && y_adv >= 0 && z_old >= 0 && z_adv >= 0) {
        ix0 = (int)(x_over_dx0 + 0.5);
        iy0 = (int)(y_over_dy0 + 0.5);
        iz0 = (int)(z_over_dz0 + 0.5);
        ix1 = (int)(x_over_dx1 + 0.5);
        iy1 = (int)(y_over_dy1 + 0.5);
        iz1 = (int)(z_over_dz1 + 0.5);
    } else {
        ix0 = (int)floor(x_over_dx0 + 0.5);
        iy0 = (int)floor(y_over_dy0 + 0.5);
        iz0 = (int)floor(z_over_dz0 + 0.5);
        ix1 = (int)floor(x_over_dx1 + 0.5);
        iy1 = (int)floor(y_over_dy1 + 0.5);
        iz1 = (int)floor(z_over_dz1 + 0.5);
    }

    double S0x[5], S0y[5], S0z[5];
    calculate_S0(ix0 - x_over_dx0, S0x);
    calculate_S0(iy0 - y_over_dy0, S0y);
    calculate_S0(iz0 - z_over_dz0, S0z);

    int dcell_x = ix1 - ix0;
    int dcell_y = iy1 - iy0;
    int dcell_z = iz1 - iz0;

    double S1x[5], S1y[5], S1z[5], DSx[5], DSy[5], DSz[5];
    if (dcell_x == 0) calculate_S0(ix1 - x_over_dx1, S1x);
    else              calculate_S(ix1 - x_over_dx1, dcell_x, S1x);
    if (dcell_y == 0) calculate_S0(iy1 - y_over_dy1, S1y);
    else              calculate_S(iy1 - y_over_dy1, dcell_y, S1y);
    if (dcell_z == 0) calculate_S0(iz1 - z_over_dz1, S1z);
    else              calculate_S(iz1 - z_over_dz1, dcell_z, S1z);

    for (int i = 0; i < 5; i++) {
        DSx[i] = S1x[i] - S0x[i];
        DSy[i] = S1y[i] - S0y[i];
        DSz[i] = S1z[i] - S0z[i];
    }

    double charge_density = q_over_dx_dy_dz * w;
    double factor_dx = q_over_dy_dz_dt * w;
    double factor_dy = q_over_dx_dz_dt * w;
    double factor_dz = q_over_dx_dy_dt * w;

    // These bounds are correct only when a particle crosses at most one cell
    // per step (|dcell| <= 1), which is the usual PIC CFL condition.
    int i_start = dcell_x < 0 ? 0 : 1;
    int i_end = dcell_x > 0 ? 5 : 4;
    int j_start = dcell_y < 0 ? 0 : 1;
    int j_end = dcell_y > 0 ? 5 : 4;
    int k_start = dcell_z < 0 ? 0 : 1;
    int k_end = dcell_z > 0 ? 5 : 4;

    // Wrap the stencil base indices into [0, n) once per particle (almost
    // always zero iterations). The per-cell single wrap below is then
    // sufficient for arbitrarily far out-of-range particles, since the
    // 5-point stencil spans at most one period. The shape factors above
    // keep using the original ix0/iy0/iz0 (delta is unaffected).
    int ix0w = ix0, iy0w = iy0, iz0w = iz0;
    while (ix0w < 0) ix0w += nx;
    while (ix0w >= nx) ix0w -= nx;
    while (iy0w < 0) iy0w += ny;
    while (iy0w >= ny) iy0w -= ny;
    while (iz0w < 0) iz0w += nz;
    while (iz0w >= nz) iz0w -= nz;

    if (dcell_x == 0 && dcell_y == 0 && dcell_z == 0) {
        // Common case (no cell crossing): compile-time constant trip counts.
        current_deposit_3d_fast_cells(
            rho, jx, jy, jz, S0x, S0y, S0z, S1x, S1y, S1z, DSx, DSy, DSz,
            ix0w, iy0w, iz0w, nx, ny, nz,
            charge_density, factor_dx, factor_dy, factor_dz, 1, 4, 1, 4, 1, 4
        );
    } else {
        current_deposit_3d_fast_cells(
            rho, jx, jy, jz, S0x, S0y, S0z, S1x, S1y, S1z, DSx, DSy, DSz,
            ix0w, iy0w, iz0w, nx, ny, nz,
            charge_density, factor_dx, factor_dy, factor_dz,
            i_start, i_end, j_start, j_end, k_start, k_end
        );
    }
}

#endif /* CURRENT_DEPOSIT_H */
