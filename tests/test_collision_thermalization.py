"""Physics-accuracy tests for the binary-collision rate.

Three independent layers, all against analytic Coulomb theory:

1. Beam deflection rate: two cold counter-streaming beams, one collision step.
   The mean deflection <1-cosX> equals the Rutherford momentum-transfer rate
   nu_tr*dt with nu_tr = 4*pi*n*(q^2/4*pi*eps0)^2*lnL/(mu^2*v_rel^3).
   Validates the Nanbu-Perez s parameter directly, with no thermal averaging.
   The model is designed to give <1-cosX> = 0.987*s at small s (alpha-fit).
   Run for all three lnLambda dispatch paths: constant, varying with no Debye
   length (floor lnL=2), and varying with a Debye length (Perez 2012 bmin).

2. One-step energy transfer: two Maxwellian populations at different
   temperatures, a single collision step, compared against the exact
   pair-level prediction computed (double sum over all N^2 pairs) from the
   same particle sample:
       dtheta_1 = -<(u1^2 - u2^2) f(s_ij)> / 6,
       s_ij = n*q^4*lnL*dt/(pi*eps0^2*m^2*c^3) / |u1-u2|^3   (Rutherford),
       f(s) = <1-cos chi> response of the alpha-fit (0.987*s at small s,
              saturating toward 1 as s -> isotropic).
   With f = s this reduces analytically to the Landau-Spitzer equilibration
   rate for Maxwellians (NRL Plasma Formulary, printed p. 35). Validates the
   pairing count, dt_corr, w_corr bookkeeping and the two-body kinematics.

3. Multi-step equilibration rate vs Landau-Spitzer. The measured rate is
   ~0.5x the Landau-Spitzer value, robustly across dt, mass ratio,
   temperature ratio and particle count. This is NOT a porting error:
   the pre-port Numba implementation (0aa0335) reproduces it (0.567), and
   layer 2 shows the per-step transfer against the *initial* sample is
   exact. The slowdown is intrinsic to the one-partner-per-step binary
   model: the transfer integral J of the *evolving* discrete sample decays
   as single-pair collisions correlate the two populations. Cross-validated
   against Smilei (independent implementation of the same algorithm): its
   published thermalisation benchmark gives 0.76x LS, and the same
   parameters run through lambdapic give 0.38x (inter-only) / 0.62x (with
   intra) -- Smilei lands closer to LS because its benchmark drives strong
   intra collisions (coulomb_log 1000) that keep both species Maxwellian.
   The test
   therefore asserts a wide band [0.40, 0.65] that still catches gross rate
   bugs (missing dt_corr => rate/N, broken lnLambda => rate/lnL, wrong s
   power => orders of magnitude) while documenting the intrinsic deviation.

References: NRL Plasma Formulary (Huba 2004), "Thermal Equilibration",
printed p. 35; Nanbu & Yonemura, J. Comput. Phys. 145, 639 (1998).
"""

import numpy as np
import pytest
from scipy.constants import c, e, epsilon_0, m_e, pi

from lambdapic.core.collision.cpu import inter_collision_cell, seed_rng_states
from lambdapic.callback.utils import SetTemperature

LN_LAMBDA = 2.0
DENSITY = 1.0e28  # m^-3
CELL_VOL = 1.0e-18  # m^3
THETA1, THETA2 = 0.02, 0.005  # kT/mc^2

# deterministic <1-cos chi> response of the alpha-fit, precomputed on a
# log grid: f(s) -> 0.987*s for s << 1, saturating toward 1 (isotropic)
_UQ = (np.arange(128) + 0.5) / 128.0
_SGRID = np.logspace(-8, 6, 8192)


def _response_grid():
    sg = _SGRID
    alpha = 0.37 * sg - 0.005 * sg**2 - 0.0064 * sg**3
    integ = _UQ / np.sqrt((1.0 - _UQ) + alpha[..., None]**2 * _UQ)
    f = 2.0 * alpha * np.mean(integ, axis=-1)
    return np.where(sg < 4.0, f, 1.0)


_RESPONSE_GRID = _response_grid()


def _angle_response(s):
    """<1-cos chi>(s) of the Nanbu alpha-fit used by the C kernel.

    sin^2(chi/2) = alpha*U/sqrt(1-U+alpha^2 U), U ~ U(0,1),
    alpha = 0.37 s - 0.005 s^2 - 0.0064 s^3  (s < 4), cosX = 2U-1  (s >= 4).
    """
    return np.interp(s, _SGRID, _RESPONSE_GRID)


def make_rng(seed: int) -> np.ndarray:
    """One PCG32 state (uint64[2]) for the C kernels."""
    return seed_rng_states(np.array([seed], dtype=np.int64))[0]


def kinetic_theta(inv_gamma: np.ndarray) -> float:
    """theta_T = (2/3)(<gamma> - 1): exact energy measure, agrees with
    <u^2>/3 to O(theta^2)."""
    return 2.0 / 3.0 * (np.mean(1.0 / inv_gamma) - 1.0)


def landau_spitzer_nu(theta_sum: float, n: float, lnL: float, m: float, q: float) -> float:
    """Temperature equilibration rate nu_{a|b} (SI, Maxwellians, no drift),
    theta_sum = kT_a/mc^2 + kT_b/mc^2 (equal masses).

    Landau-Spitzer / NRL: nu = (8*sqrt(2*pi)/3) n q^4 lnL
        / ((4*pi*eps0)^2 m^2 (kT_a/m_a + kT_b/m_b)^{3/2}).
    """
    return (8.0 * np.sqrt(2.0 * pi) / 3.0) * n * q**4 * lnL / (
        (4.0 * pi * epsilon_0)**2 * m**2 * (c**2 * theta_sum)**1.5
    )


def sample_population(n: int, theta: float, seed: int):
    """Maxwell-Juttner population as contiguous 1D momentum arrays."""
    ux, uy, uz = SetTemperature.sample_maxwell_juttner(n, theta, np.random.default_rng(seed))
    inv_gamma = 1.0 / np.sqrt(1.0 + ux**2 + uy**2 + uz**2)
    return ux, uy, uz, inv_gamma


def _run_beam_deflection(ln_lambda: float, debye_inv_sq: float, ln_eff: float | None = None,
                         u0: float = 0.02, n: int = 4096, s_target: float = 1.0e-3,
                         nseeds: int = 24) -> float:
    """One collision step of two cold counter-streaming beams; return <1-cosX>
    averaged over nseeds RNG streams (the deflection distribution has a
    log-divergent tail, so single-seed means carry ~4% noise).

    dt is set so the Rutherford parameter s = nu_tr(ln_eff)*dt = s_target
    (well inside the linear-response regime of the alpha-fit). ln_eff is
    the Coulomb logarithm the rate should follow (the passed constant, or
    the varying-path value); u0 = 0.02 keeps the relativistic correction
    (see test_beam_deflection_relativistic_factor) at 0.13%."""
    mu = m_e / 2.0
    v_rel = 2.0 * u0 * c
    nu_tr = (DENSITY * 4.0 * pi * (e**2 / (4.0 * pi * epsilon_0))**2
             * (ln_lambda if ln_eff is None else ln_eff)
             / (mu**2 * v_rel**3))
    dt = s_target / nu_tr

    ig_val = 1.0 / np.sqrt(1.0 + u0**2)
    w = np.full(n, DENSITY * CELL_VOL / n)
    out = []
    for seed in range(nseeds):
        ux1 = np.full(n, u0); uy1 = np.zeros(n); uz1 = np.zeros(n)
        ux2 = np.full(n, -u0); uy2 = np.zeros(n); uz2 = np.zeros(n)
        ig1 = np.full(n, ig_val); ig2 = np.full(n, ig_val)
        dead = np.zeros(n, dtype=bool)
        inter_collision_cell(
            ux1, uy1, uz1, ig1, w, dead, m_e, -e, 0, n,
            ux2, uy2, uz2, ig2, w, dead, m_e, -e, 0, n,
            ln_lambda, debye_inv_sq, CELL_VOL, dt, make_rng(100 + seed),
        )
        # beam 1 keeps |u| (elastic), so <cosX> = <ux>/u0
        out.append(1.0 - np.mean(ux1) / u0)
    return float(np.mean(out))


def _beam_lnlambda_eff(u0: float = 0.02, debye_inv_sq: float = 2.5e17) -> float:
    """Perez-2012 varying lnLambda for identical cold beam pairs (v_com = 0),
    mirroring varying_lnLambda in cpu.c: bmin = max(h/(2 p1_com), b_90);
    lnL = max(2, 0.5 ln(1 + lambdaD^2/bmin^2))."""
    h = 6.62607015e-34  # PLANCK in cpu.c utils.h
    p1_com = m_e * u0 * c
    gamma0 = np.sqrt(1.0 + u0**2)
    b0 = (e**2 / (4.0 * pi * epsilon_0 * c**2)) / (2.0 * m_e * gamma0) * (
        (m_e**2 * gamma0**2 * c**2 / p1_com**2) + 1.0)
    bmin = max(h / (2.0 * p1_com), b0)
    return max(2.0, 0.5 * np.log(1.0 + (1.0 / debye_inv_sq) / bmin**2))


def test_beam_deflection_rate():
    """Layer 1: <1-cosX> per step equals the Rutherford transport rate
    (constant-lnLambda path). The model is designed to give <1-cosX> =
    0.987*s at small s (alpha-fit)."""
    measured = _run_beam_deflection(LN_LAMBDA, 0.0)
    expected = 1.0e-3  # s_target: nu_tr*dt by construction
    assert measured == pytest.approx(0.987 * expected, rel=0.03), (
        f"beam deflection <1-cosX> = {measured:.5f}, expected {0.987 * expected:.5f}"
    )


def test_beam_deflection_relativistic_factor():
    """At relativistic beam speed the kernel's s carries Nanbu's relativistic
    com_term: for the equal-mass beam pair (COM at rest),
        s = s_Rutherford * (1 + 2 u0^2)^2 / gamma^3
    (closed form of the com_term/gamma factors in coulomb_scattering)."""
    for u0 in (0.15, 0.25):
        measured = _run_beam_deflection(LN_LAMBDA, 0.0, u0=u0)
        rel = (1.0 + 2.0 * u0**2)**2 / (1.0 + u0**2)**1.5
        assert measured == pytest.approx(0.987 * 1.0e-3 * rel, rel=0.03), (
            f"relativistic beam rate at u0={u0}: {measured:.5f}, "
            f"expected {0.987 * 1.0e-3 * rel:.5f} (factor {rel:.4f})"
        )


def test_beam_deflection_varying_lnlambda_floor():
    """lnLambda=0 with no Debye length: varying path falls back to lnL=2,
    giving exactly the same rate as the constant-lnL=2 beam test."""
    measured = _run_beam_deflection(0.0, 0.0, ln_eff=2.0)
    assert measured == pytest.approx(0.987 * 1.0e-3, rel=0.03), (
        f"varying-lnLambda floor rate off: {measured:.5f} vs {0.987 * 1.0e-3:.5f}"
    )


def test_beam_deflection_varying_lnlambda_debye():
    """lnLambda=0 with a Debye length: the rate follows the Perez-2012
    lnL(bmin) (quantum bmin = h/2p for these beam parameters)."""
    debye_inv_sq = 2.5e17  # lambda_D = 2 nm
    lnl = _beam_lnlambda_eff(debye_inv_sq=debye_inv_sq)
    assert lnl > 3.0  # genuinely on the Debye branch, not the floor
    measured = _run_beam_deflection(0.0, debye_inv_sq, ln_eff=lnl)
    assert measured == pytest.approx(0.987 * 1.0e-3, rel=0.03), (
        f"varying-lnLambda Debye rate off: {measured:.5f} vs "
        f"{0.987 * 1.0e-3:.5f} (lnL_eff = {lnl:.3f})"
    )


def _one_step_transfer(seed: int, n: int, dt: float):
    """One collision step between two Maxwellians; return (measured dtheta1,
    exact pair-level prediction incl. the angle-response of the algorithm)."""
    ux1, uy1, uz1, ig1 = sample_population(n, THETA1, seed)
    ux2, uy2, uz2, ig2 = sample_population(n, THETA2, seed + 1)
    w = np.full(n, DENSITY * CELL_VOL / n)
    dead = np.zeros(n, dtype=bool)

    # exact pair-level prediction: <dtheta_1> = -<(u1^2 - u2^2) f(s)>/6 with
    # s_ij = C_S/|u1-u2|^3 the Rutherford parameter (mu = m/2 folded in) and
    # f(s) the algorithm's <1-cos chi> response (0.987 s small-s, saturating).
    # f = s reduces analytically to the Landau-Spitzer rate for Maxwellians.
    C_S = DENSITY * e**4 * LN_LAMBDA * dt / (pi * epsilon_0**2 * m_e**2 * c**3)
    u1sq = ux1**2 + uy1**2 + uz1**2
    u2sq = ux2**2 + uy2**2 + uz2**2
    total = 0.0
    for i0 in range(0, n, 256):
        dx = ux1[i0:i0+256, None] - ux2[None, :]
        dy = uy1[i0:i0+256, None] - uy2[None, :]
        dz = uz1[i0:i0+256, None] - uz2[None, :]
        du3 = (dx**2 + dy**2 + dz**2)**1.5
        s_ij = C_S / np.maximum(du3, 1e-300)
        total += np.sum((u1sq[i0:i0+256, None] - u2sq[None, :]) * _angle_response(s_ij))
    predicted = -total / n**2 / 6.0

    th1_before = kinetic_theta(ig1)
    inter_collision_cell(
        ux1, uy1, uz1, ig1, w, dead, m_e, -e, 0, n,
        ux2, uy2, uz2, ig2, w, dead, m_e, -e, 0, n,
        LN_LAMBDA, 0.0, CELL_VOL, dt, make_rng(seed + 2),
    )
    measured = kinetic_theta(ig1) - th1_before
    return measured, predicted


def test_one_step_energy_transfer_matches_landau_prediction():
    """Layer 2: single-step energy transfer equals the exact pair-level
    prediction (Landau x the algorithm's own angle response) from the same
    particle sample."""
    n = 2048
    nu = landau_spitzer_nu(THETA1 + THETA2, DENSITY, LN_LAMBDA, m_e, e)
    dt = 2.0 / (nu * 320)  # same per-step strength as layer 3

    ratios = []
    for seed in range(200, 200 + 32):
        measured, predicted = _one_step_transfer(seed, n, dt)
        assert measured < 0 and predicted < 0  # hot population cools
        ratios.append(measured / predicted)

    mean_ratio = float(np.mean(ratios))
    print(f"one-step transfer / exact pair prediction: {mean_ratio:.4f} "
          f"(+- {np.std(ratios) / np.sqrt(len(ratios)):.4f} s.e.)")
    # residual ~1% bias: relativistic kinematics vs the nonrelativistic
    # prediction formula at theta ~ 0.02; the pair bookkeeping itself is exact.
    assert 0.94 < mean_ratio < 1.10, (
        f"one-step energy transfer off: measured/predicted = {mean_ratio:.3f}")


def _equilibration_trajectory(dt: float, nsteps: int, n: int, seed: int) -> np.ndarray:
    """Run inter-collisions for nsteps; return (T1-T2)(t)/(T1-T2)(0)."""
    ux1, uy1, uz1, ig1 = sample_population(n, THETA1, seed)
    ux2, uy2, uz2, ig2 = sample_population(n, THETA2, seed + 1)
    w = np.full(n, DENSITY * CELL_VOL / n)
    dead = np.zeros(n, dtype=bool)

    d = np.empty(nsteps + 1)
    d[0] = kinetic_theta(ig1) - kinetic_theta(ig2)
    rng = make_rng(seed + 2)
    for istep in range(nsteps):
        inter_collision_cell(
            ux1, uy1, uz1, ig1, w, dead, m_e, -e, 0, n,
            ux2, uy2, uz2, ig2, w, dead, m_e, -e, 0, n,
            LN_LAMBDA, 0.0, CELL_VOL, dt, rng,
        )
        d[istep + 1] = kinetic_theta(ig1) - kinetic_theta(ig2)
    return d / d[0]


def test_equilibration_rate_vs_landau_spitzer():
    """Layer 3: two-temperature equilibration rate vs Landau-Spitzer.

    See module docstring: the binary model equilibrates at ~0.5x the
    Landau-Spitzer rate (verified identical in the pre-port Numba
    implementation; per-step transfer is exact - layer 2). Assert a wide
    band that still catches gross rate errors.
    """
    n, nsteps, nruns = 4096, 320, 24
    nu = landau_spitzer_nu(THETA1 + THETA2, DENSITY, LN_LAMBDA, m_e, e)
    dt = 2.0 / (nu * nsteps)  # ~4 e-foldings of (T1-T2) at the LS rate

    d_avg = np.zeros(nsteps + 1)
    for irun in range(nruns):
        d_avg += _equilibration_trajectory(dt, nsteps, n, seed=1000 + 10 * irun)
    d_avg /= nruns

    t = np.arange(nsteps + 1) * dt

    # sanity: decays on average, no NaNs, energy flows only hot -> cold
    assert np.all(np.isfinite(d_avg))
    assert np.all(d_avg[1:] <= d_avg[:-1] + 0.01)

    # rate fit above the noise floor
    mask = d_avg > 0.25
    slope = -np.polyfit(t[mask], np.log(d_avg[mask]), 1)[0]
    fit_ratio = slope / (2.0 * nu)

    print(f"equilibration rate / Landau-Spitzer = {fit_ratio:.3f}")
    assert 0.40 < fit_ratio < 0.65, (
        f"equilibration rate ratio {fit_ratio:.3f} outside the expected "
        f"band [0.40, 0.65]; see module docstring for the intrinsic ~0.5x "
        f"deviation of the binary model"
    )

    # endpoint: ~4 LS e-foldings at the model's ~0.5x rate leaves
    # d ~ exp(-2) ~ 0.14 of d0; catches "no collisions" (=1) and
    # "over-collide" (drops to the fluctuation floor ~0.02)
    assert 0.08 < d_avg[-1] < 0.30, f"endpoint d/d0 = {d_avg[-1]:.3f}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
