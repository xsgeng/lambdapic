#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <math.h>

#include "../../utils/cutils.h"
#include "../../current/current_deposit.h"

// Particle block size for strip mining. The per-strip working set
// (14 arrays x 8 B x 256 = 28.7 KB for plain Boris, +3 spin arrays =
// 34.8 KB for the TBMT variant) is sized to stay L1-resident across
// the passes below.
#define STRIP_SIZE 256


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

__attribute__((always_inline)) inline static void push_position_3d(
    double* restrict x, double* restrict y, double* restrict z,
    double ux, double uy, double uz,
    double inv_gamma,
    double cdt
) {
    *x += cdt * inv_gamma * ux;
    *y += cdt * inv_gamma * uy;
    *z += cdt * inv_gamma * uz;
}


// Standalone TBMT (Thomas-Bargmann-Michel-Telegdi) spin rotation for one
// particle, to be applied AFTER the full Boris push. The momentum is not
// modified (no Stern-Gerlach feedback); only sx/sy/sz are updated.
//
// The effective field must be evaluated at the mid-step momentum u^-
// (after the first E half kick), like in the fused Boris+TBMT scheme.
// It is reconstructed exactly from the final momentum u^+ at no extra
// storage cost: the Boris rotation preserves the norm, so
//   u_rot = u^+ - efactor*E,   gamma^- = sqrt(1 + |u_rot|^2),
//   u^-   = M(-T) u_rot,       T = bfactor*B/gamma^-,
// where M(-T) is the inverse Boris rotation (rotation by the opposite
// angle around the same axis).
//
// The spin is then Boris-rotated around the TBMT effective field (SI units):
//   B_eff = (ae + 1/gamma) B
//         - ae*gamma/(gamma+1) * (beta . B) beta
//         - (ae + 1/(gamma+1)) * (beta x E)/c
// so that ds/dt = (q/m) s x B_eff.
__attribute__((always_inline)) inline static void tbmt_spin(
    double* restrict sx, double* restrict sy, double* restrict sz,
    double ux, double uy, double uz,
    double Ex, double Ey, double Ez,
    double Bx, double By, double Bz,
    double efactor, double bfactor, double ae
) {

    // --- reconstruct the mid-step (half E kick) momentum u^- ---
    double urx = ux - efactor * Ex;
    double ury = uy - efactor * Ey;
    double urz = uz - efactor * Ez;

    double inv_gamma_minus = 1.0 / sqrt(1.0 + urx * urx + ury * ury + urz * urz);
    double gamma = 1.0 / inv_gamma_minus;

    // inverse Boris rotation: u^- = M(-T) u_rot with T = bfactor*B/gamma^-
    double Tx = -bfactor * Bx * inv_gamma_minus;
    double Ty = -bfactor * By * inv_gamma_minus;
    double Tz = -bfactor * Bz * inv_gamma_minus;

    double upx = urx + ury * Tz - urz * Ty;
    double upy = ury + urz * Tx - urx * Tz;
    double upz = urz + urx * Ty - ury * Tx;

    double Tfactor = 2.0 / (1.0 + Tx * Tx + Ty * Ty + Tz * Tz);
    double Sx = Tfactor * Tx;
    double Sy = Tfactor * Ty;
    double Sz = Tfactor * Tz;

    double umx = urx + upy * Sz - upz * Sy;
    double umy = ury + upz * Sx - upx * Sz;
    double umz = urz + upx * Sy - upy * Sx;

    // beta = u / gamma
    double betax = umx * inv_gamma_minus;
    double betay = umy * inv_gamma_minus;
    double betaz = umz * inv_gamma_minus;

    // --- TBMT effective field (SI units) ---
    double beta_dot_B = betax * Bx + betay * By + betaz * Bz;
    double pref1 = ae * gamma / (gamma + 1.0) * beta_dot_B;
    double pref2 = ae + 1.0 / (gamma + 1.0);
    double inv_c = 1.0 / LIGHT_SPEED;

    double Bx_eff = (ae + inv_gamma_minus) * Bx - pref1 * betax - pref2 * inv_c * (betay * Ez - betaz * Ey);
    double By_eff = (ae + inv_gamma_minus) * By - pref1 * betay - pref2 * inv_c * (betaz * Ex - betax * Ez);
    double Bz_eff = (ae + inv_gamma_minus) * Bz - pref1 * betaz - pref2 * inv_c * (betax * Ey - betay * Ex);

    // --- spin rotation (Boris scheme) ---
    double Tsx = bfactor * Bx_eff;
    double Tsy = bfactor * By_eff;
    double Tsz = bfactor * Bz_eff;

    double T2 = Tsx * Tsx + Tsy * Tsy + Tsz * Tsz;
    double Ssx = 2.0 * Tsx / (1.0 + T2);
    double Ssy = 2.0 * Tsy / (1.0 + T2);
    double Ssz = 2.0 * Tsz / (1.0 + T2);

    double spx = *sx + *sy * Tsz - *sz * Tsy;
    double spy = *sy + *sz * Tsx - *sx * Tsz;
    double spz = *sz + *sx * Tsy - *sy * Tsx;

    *sx = *sx + spy * Ssz - spz * Ssy;
    *sy = *sy + spz * Ssx - spx * Ssz;
    *sz = *sz + spx * Ssy - spy * Ssx;
}

__attribute__((always_inline)) inline static void get_gx(double delta, double* gx) {
    double delta2 = delta * delta;
    gx[0] = 0.5 * (0.25 + delta2 + delta);
    gx[1] = 0.75 - delta2;
    gx[2] = 0.5 * (0.25 + delta2 - delta);
}

// Slow path: uses INDEX3 with periodic boundary checks
__attribute__((always_inline)) inline static double interp_field_safe_3d(
    double* restrict field,
    double* restrict facx, double* restrict facy, double* restrict facz,
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

__attribute__((always_inline)) inline static double interp_field_fast_3d(
    double* restrict field,
    double* restrict facx, double* restrict facy, double* restrict facz,
    int ix, int iy, int iz,
    int ny, int nz
) {
    int ny_nz = ny * nz;
    int im1 = (ix - 1) * ny_nz, i0 = ix * ny_nz, ip1 = (ix + 1) * ny_nz;
    int jm1 = (iy - 1) * nz, j0 = iy * nz, jp1 = (iy + 1) * nz;
    int km1 = iz - 1, k0 = iz, kp1 = iz + 1;
    return facz[0] * (facy[0] * (facx[0] * field[im1 + jm1 + km1]
                              + facx[1] * field[i0  + jm1 + km1]
                              + facx[2] * field[ip1 + jm1 + km1])
                    + facy[1] * (facx[0] * field[im1 + j0  + km1]
                              + facx[1] * field[i0  + j0  + km1]
                              + facx[2] * field[ip1 + j0  + km1])
                    + facy[2] * (facx[0] * field[im1 + jp1 + km1]
                              + facx[1] * field[i0  + jp1 + km1]
                              + facx[2] * field[ip1 + jp1 + km1]))
         + facz[1] * (facy[0] * (facx[0] * field[im1 + jm1 + k0]
                              + facx[1] * field[i0  + jm1 + k0]
                              + facx[2] * field[ip1 + jm1 + k0])
                    + facy[1] * (facx[0] * field[im1 + j0  + k0]
                              + facx[1] * field[i0  + j0  + k0]
                              + facx[2] * field[ip1 + j0  + k0])
                    + facy[2] * (facx[0] * field[im1 + jp1 + k0]
                              + facx[1] * field[i0  + jp1 + k0]
                              + facx[2] * field[ip1 + jp1 + k0]))
         + facz[2] * (facy[0] * (facx[0] * field[im1 + jm1 + kp1]
                              + facx[1] * field[i0  + jm1 + kp1]
                              + facx[2] * field[ip1 + jm1 + kp1])
                    + facy[1] * (facx[0] * field[im1 + j0  + kp1]
                              + facx[1] * field[i0  + j0  + kp1]
                              + facx[2] * field[ip1 + j0  + kp1])
                    + facy[2] * (facx[0] * field[im1 + jp1 + kp1]
                              + facx[1] * field[i0  + jp1 + kp1]
                              + facx[2] * field[ip1 + jp1 + kp1]));
}

__attribute__((always_inline)) inline static void interpolation_3d(
    double x, double y, double z,
    double* restrict ex_part, double* restrict ey_part, double* restrict ez_part,
    double* restrict bx_part, double* restrict by_part, double* restrict bz_part,
    double* restrict ex, double* restrict ey, double* restrict ez,
    double* restrict bx, double* restrict by, double* restrict bz,
    double inv_dx, double inv_dy, double inv_dz, double x0, double y0, double z0,
    npy_intp nx, npy_intp ny, npy_intp nz
) {

    double gx[3];
    double gy[3];
    double gz[3];
    double hx[3];
    double hy[3];
    double hz[3];

    double x_over_dx = (x - x0) * inv_dx;
    double y_over_dy = (y - y0) * inv_dy;
    double z_over_dz = (z - z0) * inv_dz;

    // Fast pre-computation with (int) - safe if result passes bounds check
    int ix1_fast = (int)(x_over_dx + 0.5);
    int ix2_fast = (int)x_over_dx;
    int iy1_fast = (int)(y_over_dy + 0.5);
    int iy2_fast = (int)y_over_dy;
    int iz1_fast = (int)(z_over_dz + 0.5);
    int iz2_fast = (int)z_over_dz;

    int use_fast_path = (ix2_fast > 0 && ix2_fast < nx - 1 && ix1_fast > 0 && ix1_fast < nx - 1 &&
                         iy2_fast > 0 && iy2_fast < ny - 1 && iy1_fast > 0 && iy1_fast < ny - 1 &&
                         iz2_fast > 0 && iz2_fast < nz - 1 && iz1_fast > 0 && iz1_fast < nz - 1);

    if (use_fast_path) {
        get_gx(ix1_fast - x_over_dx, gx);
        get_gx(ix2_fast - x_over_dx + 0.5, hx);
        get_gx(iy1_fast - y_over_dy, gy);
        get_gx(iy2_fast - y_over_dy + 0.5, hy);
        get_gx(iz1_fast - z_over_dz, gz);
        get_gx(iz2_fast - z_over_dz + 0.5, hz);

        *ex_part = interp_field_fast_3d(ex, hx, gy, gz, ix2_fast, iy1_fast, iz1_fast, ny, nz);
        *ey_part = interp_field_fast_3d(ey, gx, hy, gz, ix1_fast, iy2_fast, iz1_fast, ny, nz);
        *ez_part = interp_field_fast_3d(ez, gx, gy, hz, ix1_fast, iy1_fast, iz2_fast, ny, nz);
        *bx_part = interp_field_fast_3d(bx, gx, hy, hz, ix1_fast, iy2_fast, iz2_fast, ny, nz);
        *by_part = interp_field_fast_3d(by, hx, gy, hz, ix2_fast, iy1_fast, iz2_fast, ny, nz);
        *bz_part = interp_field_fast_3d(bz, hx, hy, gz, ix2_fast, iy2_fast, iz1_fast, ny, nz);
    } else {
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

        *ex_part = interp_field_safe_3d(ex, hx, gy, gz, ix2, iy1, iz1, nx, ny, nz);
        *ey_part = interp_field_safe_3d(ey, gx, hy, gz, ix1, iy2, iz1, nx, ny, nz);
        *ez_part = interp_field_safe_3d(ez, gx, gy, hz, ix1, iy1, iz2, nx, ny, nz);
        *bx_part = interp_field_safe_3d(bx, gx, hy, hz, ix1, iy2, iz2, nx, ny, nz);
        *by_part = interp_field_safe_3d(by, hx, gy, hz, ix2, iy1, iz2, nx, ny, nz);
        *bz_part = interp_field_safe_3d(bz, hx, hy, gz, ix2, iy2, iz1, nx, ny, nz);
    }
}


// Shared unified push: one-pass interpolation + Boris push + current
// deposition over all patches, strip-mined. When sx/sy/sz are non-NULL,
// an extra TBMT spin pass runs per strip, right after the Boris push of
// that strip (u and fields stay L1-resident).
static void unified_push_3d(
    npy_intp npatches, npy_intp nx, npy_intp ny, npy_intp nz,
    double dx, double dy, double dz,
    double dt, double q, double m, double ae,
    double** ex, double** ey, double** ez,
    double** bx, double** by, double** bz,
    double** rho, double** jx, double** jy, double** jz,
    double* x0, double* y0, double* z0,
    double** x, double** y, double** z,
    double** ux, double** uy, double** uz, double** inv_gamma,
    double** sx, double** sy, double** sz,
    double** ex_part, double** ey_part, double** ez_part,
    double** bx_part, double** by_part, double** bz_part,
    npy_bool** is_dead, double** w, npy_intp* npart
) {
    // release GIL
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (npy_intp ipatch = 0; ipatch < npatches; ipatch++) {
        // Cache patch-local pointers to avoid double indirection
        double* x_patch = x[ipatch];
        double* y_patch = y[ipatch];
        double* z_patch = z[ipatch];
        double* ux_patch = ux[ipatch];
        double* uy_patch = uy[ipatch];
        double* uz_patch = uz[ipatch];
        double* inv_gamma_patch = inv_gamma[ipatch];
        double* sx_patch = sx ? sx[ipatch] : NULL;
        double* sy_patch = sy ? sy[ipatch] : NULL;
        double* sz_patch = sz ? sz[ipatch] : NULL;
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
        double z0_patch = z0[ipatch];
        npy_intp npart_patch = npart[ipatch];

        const double efactor = q * dt / (2 * m * LIGHT_SPEED);
        const double bfactor = q * dt / (2 * m);
        const double cdt_half = LIGHT_SPEED * 0.5 * dt;
        const double q_over_dx_dy_dz = q / (dx * dy * dz);
        const double q_over_dy_dz_dt = q / (dy * dz * dt);
        const double q_over_dx_dz_dt = q / (dx * dz * dt);
        const double q_over_dx_dy_dt = q / (dx * dy * dt);
        const double inv_dx = 1.0 / dx;
        const double inv_dy = 1.0 / dy;
        const double inv_dz = 1.0 / dz;


        // Strip-mine over particle blocks: the strip working set stays
        // L1-resident across the passes below. Each strip is checked for
        // dead/NaN particles with a branch-free reduction; clean strips take
        // the split path whose loops are free of branches and vectorize
        // well, dirty strips fall back to the scalar fused loops with
        // per-particle checks.
        for (npy_intp s = 0; s < npart_patch; s += STRIP_SIZE) {
            npy_intp e = s + STRIP_SIZE;
            if (e > npart_patch) e = npart_patch;
            npy_intp ns = e - s;

            int clean_strip = 1;
            for (npy_intp ip = s; ip < e; ip++) {
                clean_strip &= !(is_dead_patch[ip] | isnan(x_patch[ip]) | isnan(y_patch[ip]) | isnan(z_patch[ip]));
            }

            if (clean_strip) {
                // Phase 1 (clean): position push + interpolation
                #pragma omp simd
                for (npy_intp i = 0; i < ns; i++) {
                    const npy_intp ip = s + i;
                    push_position_3d(
                        &x_patch[ip], &y_patch[ip], &z_patch[ip],
                        ux_patch[ip], uy_patch[ip], uz_patch[ip], inv_gamma_patch[ip],
                        cdt_half
                    );
                    interpolation_3d(
                        x_patch[ip], y_patch[ip], z_patch[ip],
                        &ex_part_patch[ip], &ey_part_patch[ip], &ez_part_patch[ip],
                        &bx_part_patch[ip], &by_part_patch[ip], &bz_part_patch[ip],
                        ex_field, ey_field, ez_field,
                        bx_field, by_field, bz_field,
                        inv_dx, inv_dy, inv_dz, x0_patch, y0_patch, z0_patch,
                        nx, ny, nz
                    );
                }
                // Boris + position push (branch-free, vectorizes)
                #pragma omp simd
                for (npy_intp i = 0; i < ns; i++) {
                    const npy_intp ip = s + i;
                    boris(
                        &ux_patch[ip], &uy_patch[ip], &uz_patch[ip], &inv_gamma_patch[ip],
                        ex_part_patch[ip], ey_part_patch[ip], ez_part_patch[ip],
                        bx_part_patch[ip], by_part_patch[ip], bz_part_patch[ip],
                        efactor, bfactor
                    );
                    push_position_3d(
                        &x_patch[ip], &y_patch[ip], &z_patch[ip],
                        ux_patch[ip], uy_patch[ip], uz_patch[ip], inv_gamma_patch[ip],
                        cdt_half
                    );
                }
                // TBMT spin pass (branch-free, vectorizes); no-op without
                // spin. Runs right after the Boris push of this strip while
                // u and fields stay L1-resident.
                if (sx) {
                    #pragma omp simd
                    for (npy_intp i = 0; i < ns; i++) {
                        const npy_intp ip = s + i;
                        tbmt_spin(
                            &sx_patch[ip], &sy_patch[ip], &sz_patch[ip],
                            ux_patch[ip], uy_patch[ip], uz_patch[ip],
                            ex_part_patch[ip], ey_part_patch[ip], ez_part_patch[ip],
                            bx_part_patch[ip], by_part_patch[ip], bz_part_patch[ip],
                            efactor, bfactor, ae
                        );
                    }
                }
                // Phase 2 (clean): current deposit, no per-particle checks.
                // No simd pragma: same-cell particles accumulate into the
                // same field cells (read-modify-write scatter), so iterations
                // are not independent.
                for (npy_intp i = 0; i < ns; i++) {
                    const npy_intp ip = s + i;
                    current_deposit_3d_fast(
                        rho_field, jx_field, jy_field, jz_field,
                        x_patch[ip], y_patch[ip], z_patch[ip],
                        ux_patch[ip], uy_patch[ip], uz_patch[ip], inv_gamma_patch[ip],
                        nx, ny, nz,
                        dx, dy, dz, x0_patch, y0_patch, z0_patch, dt, w_patch[ip],
                        q_over_dx_dy_dz, q_over_dy_dz_dt, q_over_dx_dz_dt, q_over_dx_dy_dt
                    );
                }
            } else {
                // Phase 1: position push, interpolation, Boris push, position push
                for (npy_intp ip = s; ip < e; ip++) {
                    if (is_dead_patch[ip]) continue;
                    if (isnan(x_patch[ip]) || isnan(y_patch[ip]) || isnan(z_patch[ip])) continue;
                    push_position_3d(
                        &x_patch[ip], &y_patch[ip], &z_patch[ip],
                        ux_patch[ip], uy_patch[ip], uz_patch[ip], inv_gamma_patch[ip],
                        cdt_half
                    );
                    interpolation_3d(
                        x_patch[ip], y_patch[ip], z_patch[ip],
                        &ex_part_patch[ip], &ey_part_patch[ip], &ez_part_patch[ip],
                        &bx_part_patch[ip], &by_part_patch[ip], &bz_part_patch[ip],
                        ex_field, ey_field, ez_field,
                        bx_field, by_field, bz_field,
                        inv_dx, inv_dy, inv_dz, x0_patch, y0_patch, z0_patch,
                        nx, ny, nz
                    );
                    boris(
                        &ux_patch[ip], &uy_patch[ip], &uz_patch[ip], &inv_gamma_patch[ip],
                        ex_part_patch[ip], ey_part_patch[ip], ez_part_patch[ip],
                        bx_part_patch[ip], by_part_patch[ip], bz_part_patch[ip],
                        efactor, bfactor
                    );
                    push_position_3d(
                        &x_patch[ip], &y_patch[ip], &z_patch[ip],
                        ux_patch[ip], uy_patch[ip], uz_patch[ip], inv_gamma_patch[ip],
                        cdt_half
                    );
                    // TBMT spin push; no-op without spin
                    if (sx) {
                        tbmt_spin(
                            &sx_patch[ip], &sy_patch[ip], &sz_patch[ip],
                            ux_patch[ip], uy_patch[ip], uz_patch[ip],
                            ex_part_patch[ip], ey_part_patch[ip], ez_part_patch[ip],
                            bx_part_patch[ip], by_part_patch[ip], bz_part_patch[ip],
                            efactor, bfactor, ae
                        );
                    }
                }

                // Phase 2: current deposit
                for (npy_intp ip = s; ip < e; ip++) {
                    if (is_dead_patch[ip]) continue;
                    if (isnan(x_patch[ip]) || isnan(y_patch[ip]) || isnan(z_patch[ip])) continue;
                    current_deposit_3d_fast(
                        rho_field, jx_field, jy_field, jz_field,
                        x_patch[ip], y_patch[ip], z_patch[ip],
                        ux_patch[ip], uy_patch[ip], uz_patch[ip], inv_gamma_patch[ip],
                        nx, ny, nz,
                        dx, dy, dz, x0_patch, y0_patch, z0_patch, dt, w_patch[ip],
                        q_over_dx_dy_dz, q_over_dy_dz_dt, q_over_dx_dz_dt, q_over_dx_dy_dt
                    );
                }
            }
        }
    }
    // acquire GIL
    Py_END_ALLOW_THREADS
}

static PyObject* unified_boris_pusher_cpu_3d(PyObject* self, PyObject* args) {
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
    AUTOFREE double **ex_part    = get_attr_array_double(particles_list, npatches, "ex_part");
    AUTOFREE double **ey_part    = get_attr_array_double(particles_list, npatches, "ey_part");
    AUTOFREE double **ez_part    = get_attr_array_double(particles_list, npatches, "ez_part");
    AUTOFREE double **bx_part    = get_attr_array_double(particles_list, npatches, "bx_part");
    AUTOFREE double **by_part    = get_attr_array_double(particles_list, npatches, "by_part");
    AUTOFREE double **bz_part    = get_attr_array_double(particles_list, npatches, "bz_part");
    AUTOFREE npy_bool **is_dead  = get_attr_array_bool(particles_list, npatches, "is_dead");
    AUTOFREE double **w          = get_attr_array_double(particles_list, npatches, "w");
    AUTOFREE npy_intp *npart     = get_attr_int(particles_list, npatches, "npart");

    unified_push_3d(
        npatches, nx, ny, nz, dx, dy, dz, dt, q, m, 0.0,
        ex, ey, ez, bx, by, bz, rho, jx, jy, jz,
        x0, y0, z0,
        x, y, z, ux, uy, uz, inv_gamma,
        NULL, NULL, NULL,
        ex_part, ey_part, ez_part, bx_part, by_part, bz_part,
        is_dead, w, npart
    );

    Py_RETURN_NONE;
}

static PyObject* unified_boris_tbmt_pusher_cpu_3d(PyObject* self, PyObject* args) {
    PyObject *fields_list, *particles_list;
    npy_intp npatches;
    double dt, q, m, ae;

    if (!PyArg_ParseTuple(args, "OOndddd",
        &particles_list, &fields_list,
        &npatches, &dt, &q, &m, &ae)) {
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
    AUTOFREE double **sx         = get_attr_array_double(particles_list, npatches, "sx");
    AUTOFREE double **sy         = get_attr_array_double(particles_list, npatches, "sy");
    AUTOFREE double **sz         = get_attr_array_double(particles_list, npatches, "sz");
    AUTOFREE double **ex_part    = get_attr_array_double(particles_list, npatches, "ex_part");
    AUTOFREE double **ey_part    = get_attr_array_double(particles_list, npatches, "ey_part");
    AUTOFREE double **ez_part    = get_attr_array_double(particles_list, npatches, "ez_part");
    AUTOFREE double **bx_part    = get_attr_array_double(particles_list, npatches, "bx_part");
    AUTOFREE double **by_part    = get_attr_array_double(particles_list, npatches, "by_part");
    AUTOFREE double **bz_part    = get_attr_array_double(particles_list, npatches, "bz_part");
    AUTOFREE npy_bool **is_dead  = get_attr_array_bool(particles_list, npatches, "is_dead");
    AUTOFREE double **w          = get_attr_array_double(particles_list, npatches, "w");
    AUTOFREE npy_intp *npart     = get_attr_int(particles_list, npatches, "npart");

    unified_push_3d(
        npatches, nx, ny, nz, dx, dy, dz, dt, q, m, ae,
        ex, ey, ez, bx, by, bz, rho, jx, jy, jz,
        x0, y0, z0,
        x, y, z, ux, uy, uz, inv_gamma,
        sx, sy, sz,
        ex_part, ey_part, ez_part, bx_part, by_part, bz_part,
        is_dead, w, npart
    );

    Py_RETURN_NONE;
}

static PyMethodDef CpuMethods[] = {
    {"unified_boris_pusher_cpu_3d", unified_boris_pusher_cpu_3d, METH_VARARGS, "Unified Boris Pusher"},
    {"unified_boris_tbmt_pusher_cpu_3d", unified_boris_tbmt_pusher_cpu_3d, METH_VARARGS, "Unified Boris Pusher with TBMT spin precession"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cpumodule = {
    PyModuleDef_HEAD_INIT,
    "unified_pusher_3d",
    NULL,
    -1,
    CpuMethods
};

PyMODINIT_FUNC PyInit_unified_pusher_3d(void) {
    import_array();
    return PyModule_Create(&cpumodule);
}
