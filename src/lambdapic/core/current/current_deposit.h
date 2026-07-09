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
    calculate_S(ix0 - x_over_dx0, 0, S0x);
    calculate_S(iy0 - y_over_dy0, 0, S0y);

    int dcell_x = ix1 - ix0;
    int dcell_y = iy1 - iy0;

    double S1x[5], S1y[5], DSx[5], DSy[5];
    calculate_S(ix1 - x_over_dx1, dcell_x, S1x);
    calculate_S(iy1 - y_over_dy1, dcell_y, S1y);

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

    double jx_buff[5] = {0, 0, 0, 0, 0};

    for (i = i_start; i < i_end; i++) {
        int ix = ix0 + (i - 2);
        if (ix < 0) ix = nx + ix;
        double jy_buff = 0.0;
        npy_intp ix_ny = ix * ny;
        double a = S0x[i] + 0.5 * DSx[i];
        double factor_dx_DSx_i = factor_dx * DSx[i];
        double one_twelfth_DSx_i = one_twelfth * DSx[i];
        for (j = j_start; j < j_end; j++) {
            int iy = iy0 + (j - 2);
            if (iy < 0) iy = ny + iy;
            double b = S0y[j] + 0.5 * DSy[j];
            double wy = DSy[j] * a;
            double wz = a * b + one_twelfth_DSx_i * DSy[j];

            jx_buff[j] -= factor_dx_DSx_i * b;
            jy_buff -= factor_dy * wy;

            npy_intp idx = iy + ix_ny;
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
    calculate_S(ix0 - x_over_dx0, 0, S0x);
    calculate_S(iy0 - y_over_dy0, 0, S0y);

    int dcell_x = ix1 - ix0;
    int dcell_y = iy1 - iy0;

    double S1x[5], S1y[5], DSx[5], DSy[5];
    calculate_S(ix1 - x_over_dx1, dcell_x, S1x);
    calculate_S(iy1 - y_over_dy1, dcell_y, S1y);

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

    double jx_buff[5] = {0, 0, 0, 0, 0};

    for (i = i_start; i < i_end; i++) {
        int ix = ix0 + (i - 2);
        if (ix < 0) ix = nx + ix;
        double jy_buff = 0.0;
        npy_intp ix_ny = ix * ny;
        double a = S0x[i] + 0.5 * DSx[i];
        double factor_dx_DSx_i = factor_dx * DSx[i];
        double one_twelfth_DSx_i = one_twelfth * DSx[i];
        for (j = j_start; j < j_end; j++) {
            int iy = iy0 + (j - 2);
            if (iy < 0) iy = ny + iy;
            double b = S0y[j] + 0.5 * DSy[j];
            double wy = DSy[j] * a;
            double wz = a * b + one_twelfth_DSx_i * DSy[j];

            jx_buff[j] -= factor_dx_DSx_i * b;
            jy_buff -= factor_dy * wy;

            npy_intp idx = iy + ix_ny;
            jx[idx] += jx_buff[j];
            jy[idx] += jy_buff;
            jz[idx] += factor_dt_vz * wz;
            rho[idx] += charge_density * S1x[i] * S1y[j];
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
    calculate_S(ix0 - x_over_dx0, 0, S0x);
    calculate_S(iy0 - y_over_dy0, 0, S0y);
    calculate_S(iz0 - z_over_dz0, 0, S0z);

    int dcell_x = ix1 - ix0;
    int dcell_y = iy1 - iy0;
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

    double jx_buff[5][5] = {{0}};
    npy_intp ny_nz = ny * nz;

    for (int i = i_start; i < i_end; i++) {
        int ix = ix0 + (i - 2);
        if (ix < 0) ix = nx + ix;

        double a_x = S0x[i] + 0.5 * DSx[i];
        double c_x = 0.5 * S0x[i] + one_third * DSx[i];
        double factor_dx_DSx_i = factor_dx * DSx[i];

        double jy_buff[5] = {0};

        for (int j = j_start; j < j_end; j++) {
            int iy = iy0 + (j - 2);
            if (iy < 0) iy = ny + iy;

            double a_y = S0y[j] + 0.5 * DSy[j];
            double c_y = 0.5 * S0y[j] + one_third * DSy[j];
            double factor_dy_DSy_j = factor_dy * DSy[j];
            double term_jz_ij = a_x * S0y[j] + c_x * DSy[j];

            double jz_buff = 0;

            for (int k = k_start; k < k_end; k++) {
                int iz = iz0 + (k - 2);
                if (iz < 0) iz = nz + iz;

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

#endif /* CURRENT_DEPOSIT_H */
