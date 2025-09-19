import numpy as np
import pytest
from scipy.constants import m_e, e

from lambdapic.core.particles import ParticlesBase
from lambdapic.core.collision.utils import ParticleData
from lambdapic.core.collision.cpu import inter_collision_cell, debye_length_cell

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


@pytest.mark.parametrize("n1, n2", [(8, 128), (128, 128), (1024, 128)])
def test_inter_collision_cell_lnLambda0_no_nan(n1, n2, dead_fraction):
    # Two species with identical mass/charge to avoid asymmetry issues
    part1 = create_random_particles(n1, mass=m_e, charge=-e, dead_fraction=dead_fraction, seed=1)
    part2 = create_random_particles(n2, mass=m_e, charge=-e, dead_fraction=dead_fraction, seed=2)

    # Bounds cover all particles in a single cell for each species
    ip_start1, ip_end1 = 0, n1
    ip_start2, ip_end2 = 0, n2

    debye_inv = 0.0

    rng = np.random.default_rng(42)

    # Execute inter-species collision for one cell and ensure no NaNs are produced
    inter_collision_cell(
        part1, ip_start1, ip_end1,
        part2, ip_start2, ip_end2,
        2.0, debye_inv,  # lnLambda=0 -> varying_lnLambda path
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

    rng = np.random.default_rng(33)

    E_before = (
        np.sum(part1.w[ip_start1:ip_end1] * (1.0 / part1.inv_gamma[ip_start1:ip_end1]) * part1.m)
        + np.sum(part2.w[ip_start2:ip_end2] * (1.0 / part2.inv_gamma[ip_start2:ip_end2]) * part2.m)
    )

    inter_collision_cell(
        part1, ip_start1, ip_end1,
        part2, ip_start2, ip_end2,
        2.0, debye_inv,
        cell_vol, dt,
        rng,
    )

    E_after = (
        np.sum(part1.w[ip_start1:ip_end1] * (1.0 / part1.inv_gamma[ip_start1:ip_end1]) * part1.m)
        + np.sum(part2.w[ip_start2:ip_end2] * (1.0 / part2.inv_gamma[ip_start2:ip_end2]) * part2.m)
    )

    # Slightly looser tolerance for accumulation over many pairs
    assert np.isclose(E_before, E_after, rtol=1e-3, atol=0.0)


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

    rng = np.random.default_rng(7)
    debye_inv = 0.0
    inter_collision_cell(
        part1, ip_start1, ip_end1,
        part2, ip_start2, ip_end2,
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
    rng = np.random.default_rng(55)

    inter_collision_cell(
        part1, ip_start1, ip_end1,
        part2, ip_start2, ip_end2,
        2.0, debye_inv,
        cell_vol, dt,
        rng,
    )

    changed1 = not (np.allclose(part1.ux, ux1_0) and np.allclose(part1.uy, uy1_0) and np.allclose(part1.uz, uz1_0))
    changed2 = not (np.allclose(part2.ux, ux2_0) and np.allclose(part2.uy, uy2_0) and np.allclose(part2.uz, uz2_0))
    assert (changed1 or changed2), "Expected inter collisions to alter at least one species' momentum"
