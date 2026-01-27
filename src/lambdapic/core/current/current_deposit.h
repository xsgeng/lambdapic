#ifndef CURRENT_DEPOSIT_H
#define CURRENT_DEPOSIT_H

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

inline static void current_deposit_2d(
    double* rho, double* jx, double* jy, double* jz,
    double x, double y,
    double ux, double uy, double uz, double inv_gamma,
    npy_intp nx, npy_intp ny,
    double dx, double dy, double x0, double y0, double dt, double w, double q
) {
    double vx, vy, vz;
    double x_old, y_old, x_adv, y_adv;
    double S0x[5], S1x[5], S0y[5], S1y[5], DSx[5], DSy[5], jx_buff[5];

    npy_intp i, j;

    npy_intp dcell_x, dcell_y, ix0, iy0, ix1, iy1, ix, iy;

    double x_over_dx0, x_over_dx1, y_over_dy0, y_over_dy1;

    double charge_density, factor, jy_buff;

    vx = ux * LIGHT_SPEED * inv_gamma;
    vy = uy * LIGHT_SPEED * inv_gamma;
    vz = uz * LIGHT_SPEED * inv_gamma;
    x_old = x - vx * 0.5 * dt - x0;
    y_old = y - vy * 0.5 * dt - y0;
    x_adv = x + vx * 0.5 * dt - x0;
    y_adv = y + vy * 0.5 * dt - y0;

    x_over_dx0 = x_old / dx;
    ix0 = (int)floor(x_over_dx0 + 0.5);
    y_over_dy0 = y_old / dy;
    iy0 = (int)floor(y_over_dy0 + 0.5);

    calculate_S(ix0 - x_over_dx0, 0, S0x);
    calculate_S(iy0 - y_over_dy0, 0, S0y);

    x_over_dx1 = x_adv / dx;
    ix1 = (int)floor(x_over_dx1 + 0.5);
    dcell_x = ix1 - ix0;

    y_over_dy1 = y_adv / dy;
    iy1 = (int)floor(y_over_dy1 + 0.5);
    dcell_y = iy1 - iy0;

    calculate_S(ix1 - x_over_dx1, dcell_x, S1x);
    calculate_S(iy1 - y_over_dy1, dcell_y, S1y);

    for (i = 0; i < 5; i++) {
        DSx[i] = S1x[i] - S0x[i];
        DSy[i] = S1y[i] - S0y[i];
        jx_buff[i] = 0;
    }

    charge_density = q * w / (dx * dy);
    factor = charge_density / dt;

    for (i = fmin(1, 1 + dcell_x); i < fmax(4, 4 + dcell_x); i++) {
        ix = ix0 + (i - 2);
        if (ix < 0) {
            ix = nx + ix;
        }
        jy_buff = 0.0;
        for (j = fmin(1, 1 + dcell_y); j < fmax(4, 4 + dcell_y); j++) {
            iy = iy0 + (j - 2);
            if (iy < 0) {
                iy = ny + iy;
            }
            double wx = DSx[i] * (S0y[j] + 0.5 * DSy[j]);
            double wy = DSy[j] * (S0x[i] + 0.5 * DSx[i]);
            double wz = S0x[i] * S0y[j] + 0.5 * DSx[i] * S0y[j] + 0.5 * S0x[i] * DSy[j] + one_third * DSx[i] * DSy[j];

            jx_buff[j] -= factor * dx * wx;
            jy_buff -= factor * dy * wy;

            jx[INDEX2(ix, iy)] += jx_buff[j];
            jy[INDEX2(ix, iy)] += jy_buff;
            jz[INDEX2(ix, iy)] += factor * dt * wz * vz;
            rho[INDEX2(ix, iy)] += charge_density * S1x[i] * S1y[j];
        }
    }
}

#endif /* CURRENT_DEPOSIT_H */