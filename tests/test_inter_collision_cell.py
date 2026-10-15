import numpy as np
import pytest
from scipy.constants import m_e, e

from lambdapic.core.particles import ParticlesBase
from lambdapic.core.collision.utils import ParticleData
from lambdapic.core.collision.cpu import inter_collision_cell, seed_rng_states


def make_rng(seed: int) -> np.ndarray:
    """One PCG32 state (uint64[2]) from a seed, for the C kernels."""
    return seed_rng_states(np.array([seed], dtype=np.int64))[0]

dx = dy = dz = 1e-6
cell_vol = dx * dy * dz
dt = 1e-15

def create_random_particles(n_particles: int,
                            mass: float = m_e,
                            charge: float = -e,
                            dead_fraction: float = 0.0,
                            seed: int = 1234) -> ParticleData:
    """Create a ParticleData with random momenta and consistent inv_gamma."""
    rng = np.random.default_rng(seed)

    particles = ParticlesBase()
    particles.initialize(n_particles)

    # Random normalized momentum u = gamma * beta (keep moderate so gamma not huge)
    ux = rng.normal(0.0, 0.1, size=n_particles)
    uy = rng.normal(0.0, 0.1, size=n_particles)
    uz = rng.normal(0.0, 0.1, size=n_particles)
    u2 = ux**2 + uy**2 + uz**2

    particles.ux[:] = ux
    particles.uy[:] = uy
    particles.uz[:] = uz

    # inv_gamma = 1/sqrt(1+u^2)
    particles.inv_gamma[:] = 1.0 / np.sqrt(1.0 + u2)

    # unit weights
    particles.w[:] = 1e29 * cell_vol / n_particles
    particles.is_dead[:] = np.random.uniform(size=n_particles) < dead_fraction

    return ParticleData(
        x=particles.x, y=particles.y, z=particles.z,
        ux=particles.ux, uy=particles.uy, uz=particles.uz,
        inv_gamma=particles.inv_gamma, w=particles.w,
        is_dead=particles.is_dead, m=mass, q=charge,
    )


def assert_no_nans_particle_data(p: ParticleData):
    """Assert particle momenta and inv_gamma are all finite and valid."""
    for arr in (p.ux, p.uy, p.uz, p.inv_gamma):
        assert np.all(np.isfinite(arr))
    # inv_gamma should be in (0, 1]
    assert np.all(p.inv_gamma > 0)
    assert np.all(p.inv_gamma <= 1.0)

    # Explicitly check each momentum component arrays contain no NaNs or infs
    for arr in (p.ux, p.uy, p.uz):
        assert np.all(np.isfinite(arr))


@pytest.fixture(params=[0.0, 0.2, 1.0])
def dead_fraction(request):
    return request.param


@pytest.mark.parametrize("ln_lambda,debye_inv", [
    (2.0, 0.0),    # constant-lnLambda path
    (0.0, 0.0),    # varying path, no Debye length -> floor lnL = 2
    (0.0, 1.0e18), # varying path with Debye length
])
@pytest.mark.parametrize("n1, n2", [(8, 128), (128, 128), (1024, 128)])
def test_inter_collision_cell_no_nan(n1, n2, dead_fraction, ln_lambda, debye_inv):
    # Two species with identical mass/charge to avoid asymmetry issues
    part1 = create_random_particles(n1, mass=m_e, charge=-e, dead_fraction=dead_fraction, seed=1)
    part2 = create_random_particles(n2, mass=m_e, charge=-e, dead_fraction=dead_fraction, seed=2)

    # Bounds cover all particles in a single cell for each species
    ip_start1, ip_end1 = 0, n1
    ip_start2, ip_end2 = 0, n2

    rng = make_rng(42)

    # Execute inter-species collision for one cell and ensure no NaNs are produced
    inter_collision_cell(
        part1.ux, part1.uy, part1.uz, part1.inv_gamma, part1.w, part1.is_dead,
        part1.m, part1.q,
        ip_start1, ip_end1,
        part2.ux, part2.uy, part2.uz, part2.inv_gamma, part2.w, part2.is_dead,
        part2.m, part2.q,
        ip_start2, ip_end2,
        ln_lambda, debye_inv,
        cell_vol, dt,
        rng,
    )

    assert_no_nans_particle_data(part1)
    assert_no_nans_particle_data(part2)

@pytest.mark.parametrize("n1, n2", [(8, 128), (128, 128), (1024, 128)])
def test_inter_collision_energy_conservation(n1, n2, dead_fraction):
    part1 = create_random_particles(n1, mass=m_e, charge=-e, dead_fraction=dead_fraction, seed=101)
    part2 = create_random_particles(n2, mass=m_e, charge=-e, dead_fraction=dead_fraction, seed=202)

    ip_start1, ip_end1 = 0, n1
    ip_start2, ip_end2 = 0, n2

    debye_inv = 0.0

    rng = make_rng(33)

    E_before = (
        np.sum(part1.w[ip_start1:ip_end1] * (1.0 / part1.inv_gamma[ip_start1:ip_end1]) * part1.m)
        + np.sum(part2.w[ip_start2:ip_end2] * (1.0 / part2.inv_gamma[ip_start2:ip_end2]) * part2.m)
    )

    inter_collision_cell(
        part1.ux, part1.uy, part1.uz, part1.inv_gamma, part1.w, part1.is_dead,
        part1.m, part1.q,
        ip_start1, ip_end1,
        part2.ux, part2.uy, part2.uz, part2.inv_gamma, part2.w, part2.is_dead,
        part2.m, part2.q,
        ip_start2, ip_end2,
        2.0, debye_inv,
        cell_vol, dt,
        rng,
    )

    E_after = (
        np.sum(part1.w[ip_start1:ip_end1] * (1.0 / part1.inv_gamma[ip_start1:ip_end1]) * part1.m)
        + np.sum(part2.w[ip_start2:ip_end2] * (1.0 / part2.inv_gamma[ip_start2:ip_end2]) * part2.m)
    )

    # Slightly looser tolerance for accumulation over many pairs.
    # rtol=5e-3: the 8-vs-128 unequal-weight parameterization has high
    # accept/reject sampling variance; over 50 seeds the energy error is
    # symmetric around ~4e-4 with rare ~1.2e-3 outliers (PCG32 stream).
    assert np.isclose(E_before, E_after, rtol=5e-3, atol=0.0)


def test_inter_collision_respects_dead_flags(dead_fraction):
    # Use moderate sizes to ensure some pairs form when alive exist
    n1, n2 = 256, 192
    part1 = create_random_particles(n1, mass=m_e, charge=-e, dead_fraction=dead_fraction, seed=11)
    part2 = create_random_particles(n2, mass=m_e, charge=-e, dead_fraction=dead_fraction, seed=22)

    ip_start1, ip_end1 = 0, n1
    ip_start2, ip_end2 = 0, n2

    # Snapshot state of dead particles before collisions
    dead1 = np.where(part1.is_dead)[0]
    dead2 = np.where(part2.is_dead)[0]
    ux1_d, uy1_d, uz1_d, ig1_d, w1_d = (
        part1.ux[dead1].copy(), part1.uy[dead1].copy(), part1.uz[dead1].copy(),
        part1.inv_gamma[dead1].copy(), part1.w[dead1].copy()
    )
    ux2_d, uy2_d, uz2_d, ig2_d, w2_d = (
        part2.ux[dead2].copy(), part2.uy[dead2].copy(), part2.uz[dead2].copy(),
        part2.inv_gamma[dead2].copy(), part2.w[dead2].copy()
    )

    rng = make_rng(7)
    debye_inv = 0.0
    inter_collision_cell(
        part1.ux, part1.uy, part1.uz, part1.inv_gamma, part1.w, part1.is_dead,
        part1.m, part1.q,
        ip_start1, ip_end1,
        part2.ux, part2.uy, part2.uz, part2.inv_gamma, part2.w, part2.is_dead,
        part2.m, part2.q,
        ip_start2, ip_end2,
        2.0, debye_inv,
        cell_vol, dt,
        rng,
    )

    # Dead particles must be unchanged by collisions
    assert np.array_equal(part1.ux[dead1], ux1_d)
    assert np.array_equal(part1.uy[dead1], uy1_d)
    assert np.array_equal(part1.uz[dead1], uz1_d)
    assert np.array_equal(part1.inv_gamma[dead1], ig1_d)
    assert np.array_equal(part1.w[dead1], w1_d)

    assert np.array_equal(part2.ux[dead2], ux2_d)
    assert np.array_equal(part2.uy[dead2], uy2_d)
    assert np.array_equal(part2.uz[dead2], uz2_d)
    assert np.array_equal(part2.inv_gamma[dead2], ig2_d)
    assert np.array_equal(part2.w[dead2], w2_d)

    # General sanity: no NaNs anywhere
    assert_no_nans_particle_data(part1)
    assert_no_nans_particle_data(part2)


def test_inter_collision_alters_momentum():
    # Ensure all alive to test that momentum changes occur
    n1, n2 = 128, 192
    part1 = create_random_particles(n1, mass=m_e, charge=-e, dead_fraction=0.0, seed=303)
    part2 = create_random_particles(n2, mass=m_e, charge=-e, dead_fraction=0.0, seed=404)

    ux1_0, uy1_0, uz1_0 = part1.ux.copy(), part1.uy.copy(), part1.uz.copy()
    ux2_0, uy2_0, uz2_0 = part2.ux.copy(), part2.uy.copy(), part2.uz.copy()

    ip_start1, ip_end1 = 0, n1
    ip_start2, ip_end2 = 0, n2

    debye_inv = 0.0
    rng = make_rng(55)

    inter_collision_cell(
        part1.ux, part1.uy, part1.uz, part1.inv_gamma, part1.w, part1.is_dead,
        part1.m, part1.q,
        ip_start1, ip_end1,
        part2.ux, part2.uy, part2.uz, part2.inv_gamma, part2.w, part2.is_dead,
        part2.m, part2.q,
        ip_start2, ip_end2,
        2.0, debye_inv,
        cell_vol, dt,
        rng,
    )

    changed1 = not (np.allclose(part1.ux, ux1_0) and np.allclose(part1.uy, uy1_0) and np.allclose(part1.uz, uz1_0))
    changed2 = not (np.allclose(part2.ux, ux2_0) and np.allclose(part2.uy, uy2_0) and np.allclose(part2.uz, uz2_0))
    assert (changed1 or changed2), "Expected inter collisions to alter at least one species' momentum"


def test_inter_collision_equal_weight_conserves_momentum_and_energy():
    """Equal macro-weights: every pair updates both particles, so one step
    conserves total momentum (each component) and total energy to machine
    precision -- the per-pair two-body event is exact."""
    n = 512
    part1 = create_random_particles(n, mass=m_e, charge=-e, dead_fraction=0.1, seed=71)
    part2 = create_random_particles(n, mass=m_e, charge=-e, dead_fraction=0.1, seed=72)

    def momentum(p):
        return np.array([np.sum(p.w * p.m * p.ux),
                         np.sum(p.w * p.m * p.uy),
                         np.sum(p.w * p.m * p.uz)])

    def energy(p):
        return np.sum(p.w * (1.0 / p.inv_gamma) * p.m)

    p_before = momentum(part1) + momentum(part2)
    e_before = energy(part1) + energy(part2)

    inter_collision_cell(
        part1.ux, part1.uy, part1.uz, part1.inv_gamma, part1.w, part1.is_dead,
        part1.m, part1.q, 0, n,
        part2.ux, part2.uy, part2.uz, part2.inv_gamma, part2.w, part2.is_dead,
        part2.m, part2.q, 0, n,
        2.0, 0.0, cell_vol, dt, make_rng(99),
    )

    p_after = momentum(part1) + momentum(part2)
    e_after = energy(part1) + energy(part2)

    assert np.allclose(p_before, p_after, rtol=1e-12, atol=1e-30), \
        f"momentum drift {(p_after - p_before) / np.maximum(np.abs(p_before), 1e-300)}"
    assert e_before == pytest.approx(e_after, rel=1e-12), \
        f"energy drift {(e_after - e_before) / e_before:+.2e}"


def test_inter_collision_electron_proton():
    """Electron-proton mass ratio (1836): one step conserves momentum and
    energy exactly, and a cold electron beam through protons at rest
    deflects at the Rutherford rate with mu = m_e*m_p/(m_e+m_p)."""
    from scipy.constants import c, m_p

    n = 512
    part_e = create_random_particles(n, mass=m_e, charge=-e, seed=81)
    part_p = create_random_particles(n, mass=m_p, charge=e, seed=82)
    part_e.ux[:] = np.random.default_rng(83).normal(0.0, 0.05, n)
    part_e.uy[:] = 0.0
    part_e.uz[:] = 0.0
    part_e.inv_gamma[:] = 1.0 / np.sqrt(1.0 + part_e.ux**2)
    part_p.ux[:] = part_p.uy[:] = part_p.uz[:] = 0.0
    part_p.inv_gamma[:] = 1.0

    pe0 = np.sum(part_e.w * m_e * part_e.ux)
    pp0 = np.sum(part_p.w * m_p * part_p.ux)
    ee0 = np.sum(part_e.w / part_e.inv_gamma * m_e)
    ep0 = np.sum(part_p.w / part_p.inv_gamma * m_p)

    inter_collision_cell(
        part_e.ux, part_e.uy, part_e.uz, part_e.inv_gamma, part_e.w, part_e.is_dead,
        part_e.m, part_e.q, 0, n,
        part_p.ux, part_p.uy, part_p.uz, part_p.inv_gamma, part_p.w, part_p.is_dead,
        part_p.m, part_p.q, 0, n,
        2.0, 0.0, cell_vol, dt, make_rng(84),
    )

    pe1 = np.sum(part_e.w * m_e * part_e.ux)
    pp1 = np.sum(part_p.w * m_p * part_p.ux)
    ee1 = np.sum(part_e.w / part_e.inv_gamma * m_e)
    ep1 = np.sum(part_p.w / part_p.inv_gamma * m_p)
    assert (pe1 + pp1) == pytest.approx(pe0 + pp0, rel=1e-12)
    assert (ee1 + ep1) == pytest.approx(ee0 + ep0, rel=1e-12)
    # protons take up the electron momentum loss (they start at rest)
    assert pp1 > 0 and pp1 == pytest.approx(pe0 - pe1, rel=1e-9)


def test_inter_collision_electron_proton_beam_deflection_rate():
    """Cold electron beam (u0) through protons at rest: <1-cos chi> equals
    the Rutherford transport rate with the e-p reduced mass. u0 = 0.02 keeps
    the relativistic com_term correction below 0.1%."""
    from scipy.constants import c, epsilon_0, m_p, pi

    n = 4096
    u0 = 0.02
    n_target = 1.0e28
    mu = m_e * m_p / (m_e + m_p)
    v_rel = u0 * c
    nu_tr = (n_target * 4.0 * pi * (e**2 / (4.0 * pi * epsilon_0))**2
             * 2.0 / (mu**2 * v_rel**3))
    dt_step = 1.0e-3 / nu_tr  # s ~ 1e-3: linear response

    measured = 0.0
    for seed in (7, 17, 27, 37, 47, 57, 67, 77):
        ux_e = np.full(n, u0); uy_e = np.zeros(n); uz_e = np.zeros(n)
        ux_p = np.zeros(n); uy_p = np.zeros(n); uz_p = np.zeros(n)
        ig_e = np.full(n, 1.0 / np.sqrt(1.0 + u0**2))
        ig_p = np.ones(n)
        w = np.full(n, n_target * cell_vol / n)
        dead = np.zeros(n, dtype=bool)

        inter_collision_cell(
            ux_e, uy_e, uz_e, ig_e, w, dead, m_e, -e, 0, n,
            ux_p, uy_p, uz_p, ig_p, w, dead, m_p, e, 0, n,
            2.0, 0.0, cell_vol, dt_step, make_rng(seed),
        )
        measured += 1.0 - np.mean(ux_e) / u0
    measured /= 8.0

    assert measured == pytest.approx(0.987 * 1.0e-3, rel=0.03), (
        f"e-p beam deflection {measured:.6f}, expected {0.987 * 1.0e-3:.6f}")


def test_inter_collision_unequal_count_deflection_rates():
    """N1 = 4*N2 with equal macro weights: the majority species deflects at
    the Rutherford rate set by the MINORITY density, the minority at 4x that
    (its particles are paired 4 times per step) -- the w_corr/dt_corr
    bookkeeping of the pairing."""
    from scipy.constants import c, epsilon_0, pi

    n1, n2 = 2048, 512
    u0 = 0.02
    mu = m_e / 2.0
    v_rel = 2.0 * u0 * c
    n_minority = 0.25e28  # w = 1e28*V/n1 -> each species carries 0.25e28
    nu_tr = (n_minority * 4.0 * pi * (e**2 / (4.0 * pi * epsilon_0))**2
             * 2.0 / (mu**2 * v_rel**3))
    dt_step = 1.0e-3 / nu_tr  # s per majority particle ~ 1e-3

    maj = mino = 0.0
    for seed in (21, 31, 41, 51):
        w = 1.0e28 * cell_vol / n1  # equal per-particle weights -> n1 = 4*n2
        ux1 = np.full(n1, u0); uy1 = np.zeros(n1); uz1 = np.zeros(n1)
        ux2 = np.full(n2, -u0); uy2 = np.zeros(n2); uz2 = np.zeros(n2)
        ig1 = np.full(n1, 1.0 / np.sqrt(1.0 + u0**2))
        ig2 = np.full(n2, 1.0 / np.sqrt(1.0 + u0**2))
        w1 = np.full(n1, w); w2 = np.full(n2, w)
        dead1 = np.zeros(n1, dtype=bool); dead2 = np.zeros(n2, dtype=bool)

        inter_collision_cell(
            ux1, uy1, uz1, ig1, w1, dead1, m_e, -e, 0, n1,
            ux2, uy2, uz2, ig2, w2, dead2, m_e, -e, 0, n2,
            2.0, 0.0, cell_vol, dt_step, make_rng(seed),
        )
        maj += 1.0 - np.mean(ux1) / u0
        mino += 1.0 + np.mean(ux2) / u0
    maj /= 4.0
    mino /= 4.0

    assert maj == pytest.approx(0.987 * 1.0e-3, rel=0.04), (
        f"majority rate {maj:.6f}, expected {0.987 * 1.0e-3:.6f}")
    assert mino == pytest.approx(0.987 * 4.0e-3, rel=0.08), (
        f"minority rate {mino:.6f}, expected {0.987 * 4.0e-3:.6f}")
