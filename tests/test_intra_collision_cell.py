import numpy as np
import pytest
from scipy.constants import m_e, e

from lambdapic.core.particles import ParticlesBase
from lambdapic.core.collision.utils import ParticleData
from lambdapic.core.collision.cpu import intra_collision_cell


def create_random_particles(
    n_particles: int,
    mass: float = m_e,
    charge: float = -e,
    dead_fraction: float = 0.0,
    seed: int = 1234,
) -> ParticleData:
    """Create a ParticleData with random momenta and consistent inv_gamma."""
    rng = np.random.default_rng(seed)

    particles = ParticlesBase()
    particles.initialize(n_particles)

    # Random normalized momentum u = gamma * beta (keep moderate so gamma not huge)
    ux = rng.normal(0.0, 0.001, size=n_particles)
    uy = rng.normal(0.0, 0.001, size=n_particles)
    uz = rng.normal(0.0, 0.001, size=n_particles)
    u2 = ux**2 + uy**2 + uz**2

    particles.ux[:] = ux
    particles.uy[:] = uy
    particles.uz[:] = uz

    # inv_gamma = 1/sqrt(1+u^2)
    particles.inv_gamma[:] = 1.0 / np.sqrt(1.0 + u2)

    particles.w[:] = 1.0e45
    particles.is_dead[:] = np.random.uniform(size=n_particles) < dead_fraction

    return ParticleData(
        x=particles.x,
        y=particles.y,
        z=particles.z,
        ux=particles.ux,
        uy=particles.uy,
        uz=particles.uz,
        inv_gamma=particles.inv_gamma,
        w=particles.w,
        is_dead=particles.is_dead,
        m=mass,
        q=charge,
    )


def assert_no_nans_particle_data(p: ParticleData):
    """Assert particle momenta and inv_gamma are all finite and valid."""
    for arr in (p.ux, p.uy, p.uz, p.inv_gamma):
        assert np.all(np.isfinite(arr))
    # inv_gamma should be in (0, 1]
    assert np.all(p.inv_gamma > 0)
    assert np.all(p.inv_gamma <= 1.0)

    # Explicitly check momentum components contain no NaNs or infs
    for arr in (p.ux, p.uy, p.uz):
        assert np.all(np.isfinite(arr))


@pytest.fixture(params=[0.0, 0.2, 1.0])
def dead_fraction(request):
    return request.param


@pytest.mark.parametrize("n", [8, 128, 1024])
def test_intra_collision_cell_no_nan(n, dead_fraction):
    # One species, identical mass/charge for all particles
    part = create_random_particles(n, mass=m_e, charge=-e, dead_fraction=dead_fraction, seed=1)

    # Bounds cover all particles in a single cell
    ip_start, ip_end = 0, n

    # Cell volume and timestep
    dx = dy = dz = 1e-6
    cell_vol = dx * dy * dz
    dt = 1e-15

    debye_inv = 0.0
    rng = np.random.default_rng(42)

    # Execute intra-species collision for one cell and ensure no NaNs are produced
    intra_collision_cell(
        part,
        ip_start,
        ip_end,
        2.0,  # constant lnLambda path
        debye_inv,
        cell_vol,
        dt,
        rng,
    )

    assert_no_nans_particle_data(part)

@pytest.mark.parametrize("n", [8, 128, 1024])
def test_intra_collision_energy_conservation(n, dead_fraction):
    part = create_random_particles(n, mass=m_e, charge=-e, dead_fraction=dead_fraction, seed=101)

    ip_start, ip_end = 0, n

    dx = dy = dz = 1e-6
    cell_vol = dx * dy * dz
    dt = 1e-15

    debye_inv = 0.0
    rng = np.random.default_rng(33)

    # Total energy proxy: sum(w * gamma * m). c^2 factor cancels in comparison.
    E_before = np.sum(part.w[ip_start:ip_end] * (1.0 / part.inv_gamma[ip_start:ip_end]) * part.m)

    intra_collision_cell(
        part,
        ip_start,
        ip_end,
        2.0,
        debye_inv,
        cell_vol,
        dt,
        rng,
    )

    E_after = np.sum(part.w[ip_start:ip_end] * (1.0 / part.inv_gamma[ip_start:ip_end]) * part.m)

    # Slightly looser tolerance for accumulation over many pairs
    assert np.isclose(E_before, E_after, rtol=1e-10, atol=0.0)

def test_intra_collision_alters_momentum():
    n = 128
    # Ensure all alive to test that momentum changes occur
    part = create_random_particles(n, mass=m_e, charge=-e, dead_fraction=0.0, seed=101)
    ux0 = part.ux.copy()
    uy0 = part.uy.copy()
    uz0 = part.uz.copy()

    ip_start, ip_end = 0, n

    dx = dy = dz = 1e-6
    cell_vol = dx * dy * dz
    dt = 1e-15

    debye_inv = 0.0
    rng = np.random.default_rng(33)

    intra_collision_cell(
        part,
        ip_start,
        ip_end,
        2.0,
        debye_inv,
        cell_vol,
        dt,
        rng,
    )

    changed = False
    if not np.allclose(part.ux, ux0) or \
        not np.allclose(part.uy, uy0) or \
        not np.allclose(part.uz, uz0):
        changed = True
    assert changed, "Expected intra collisions to alter species 0 momentum"


def test_intra_collision_respects_dead_flags(dead_fraction):
    n = 256
    part = create_random_particles(n, mass=m_e, charge=-e, dead_fraction=dead_fraction, seed=2024)

    ip_start, ip_end = 0, n

    dx = dy = dz = 1e-6
    cell_vol = dx * dy * dz
    dt = 1e-15

    # Snapshot state of dead particles before collisions
    dead_idx = np.where(part.is_dead)[0]
    ux_d = part.ux[dead_idx].copy()
    uy_d = part.uy[dead_idx].copy()
    uz_d = part.uz[dead_idx].copy()
    ig_d = part.inv_gamma[dead_idx].copy()
    w_d = part.w[dead_idx].copy()

    debye_inv = 0.0
    rng = np.random.default_rng(99)

    intra_collision_cell(
        part,
        ip_start,
        ip_end,
        2.0,
        debye_inv,
        cell_vol,
        dt,
        rng,
    )

    # Dead particles must be unchanged by collisions
    assert np.array_equal(part.ux[dead_idx], ux_d)
    assert np.array_equal(part.uy[dead_idx], uy_d)
    assert np.array_equal(part.uz[dead_idx], uz_d)
    assert np.array_equal(part.inv_gamma[dead_idx], ig_d)
    assert np.array_equal(part.w[dead_idx], w_d)

    # General sanity: no NaNs anywhere
    assert_no_nans_particle_data(part)
