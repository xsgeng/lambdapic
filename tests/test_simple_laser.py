import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import c, e, m_e, pi

from lambdapic import Simulation
from lambdapic.callback.laser import SimpleLaser2D

um = 1e-6
l0 = 0.8 * um
omega0 = 2 * np.pi * c / l0


@pytest.fixture
def small_sim():
    """Create a small initialized 2D simulation for laser unit tests."""
    nx = 64
    ny = 64
    dx = l0 / 20
    dy = l0 / 20

    sim = Simulation(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        npatch_x=2,
        npatch_y=2,
        dt_cfl=0.95,
    )
    sim.initialize()
    return sim


class TestSimpleLaserEllipticity:
    """Unit tests for SimpleLaser ellipticity support."""

    def test_linear_polarization_default(self, small_sim):
        """Default ellipticity=0 should give pure linear polarization."""
        sim = small_sim
        laser = SimpleLaser2D(a0=1.0, w0=2e-6, ctau=5e-6, l0=l0, pol_angle=0.0)

        sim.time = laser.ctau / c
        patch = sim.patches[0]
        ey, ez = laser._calculate_bound_fields(sim, patch)

        assert ey is not None
        assert ez is not None
        # For pol_angle=0 and ellipticity=0, Ez should be zero everywhere
        assert_allclose(ez, 0.0, atol=1e-15)
        # Ey should be non-zero at the center
        center_idx = ey.shape[0] // 2
        assert np.abs(ey[center_idx]) > 0.0

    def test_linear_polarization_rotated(self, small_sim):
        """Ellipticity=0 with non-zero pol_angle should give linear polarization."""
        sim = small_sim
        pol = pi / 4
        laser = SimpleLaser2D(a0=1.0, w0=2e-6, ctau=5e-6, l0=l0, pol_angle=pol)

        sim.time = laser.ctau / c
        patch = sim.patches[0]
        ey, ez = laser._calculate_bound_fields(sim, patch)

        # Ey and Ez should be proportional to cos(pol) and sin(pol)
        # At any given point, the ratio should be constant (line in y-z plane)
        mask = np.abs(ey) > 1e-15
        if mask.any():
            ratios = ez[mask] / ey[mask]
            assert_allclose(ratios, np.tan(pol), rtol=1e-10)

    def test_circular_polarization(self, small_sim):
        """Ellipticity=1 with pol_angle=0 should give circular polarization."""
        sim = small_sim
        ctau = 5e-6
        cep = np.pi / 4 - omega0 * ctau / c
        laser = SimpleLaser2D(a0=1.0, w0=2e-6, ctau=ctau, l0=l0,
                              pol_angle=0.0, ellipticity=1.0, cep=cep)

        sim.time = ctau / c
        patch = sim.patches[0]
        ey, ez = laser._calculate_bound_fields(sim, patch)

        assert np.abs(ey).max() > 0.0
        assert np.abs(ez).max() > 0.0
        assert_allclose(np.abs(ey), np.abs(ez), rtol=1e-10)

    def test_circular_polarization_intensity_conservation(self, small_sim):
        """Peak total field for circular pol should be a0/sqrt(2) per component."""
        sim = small_sim
        a0 = 2.0
        ctau = 5e-6
        cep = np.pi / 4 - omega0 * ctau / c
        laser_lin = SimpleLaser2D(a0=a0, w0=2e-6, ctau=ctau, l0=l0,
                                  pol_angle=0.0, ellipticity=0.0, cep=cep)
        laser_circ = SimpleLaser2D(a0=a0, w0=2e-6, ctau=ctau, l0=l0,
                                   pol_angle=0.0, ellipticity=1.0, cep=cep)

        sim.time = ctau / c
        patch = sim.patches[0]
        ey_lin, ez_lin = laser_lin._calculate_bound_fields(sim, patch)
        ey_circ, ez_circ = laser_circ._calculate_bound_fields(sim, patch)

        peak_lin = np.abs(ey_lin).max()
        peak_circ = np.abs(ey_circ).max()
        assert_allclose(peak_circ, peak_lin / np.sqrt(2), rtol=1e-10)

    def test_ellipticity_invalid_range(self):
        """Ellipticity outside [-1, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="Ellipticity"):
            SimpleLaser2D(a0=1.0, w0=2e-6, ctau=5e-6, l0=l0, ellipticity=1.5)

        with pytest.raises(ValueError, match="Ellipticity"):
            SimpleLaser2D(a0=1.0, w0=2e-6, ctau=5e-6, l0=l0, ellipticity=-1.5)

    def test_handedness_sign(self, small_sim):
        """Positive and negative ellipticity should produce opposite Ez signs."""
        sim = small_sim
        ctau = 5e-6
        laser_pos = SimpleLaser2D(a0=1.0, w0=2e-6, ctau=ctau, l0=l0,
                                  pol_angle=0.0, ellipticity=1.0)
        laser_neg = SimpleLaser2D(a0=1.0, w0=2e-6, ctau=ctau, l0=l0,
                                  pol_angle=0.0, ellipticity=-1.0)

        sim.time = ctau / c
        patch = sim.patches[0]
        ey_pos, ez_pos = laser_pos._calculate_bound_fields(sim, patch)
        ey_neg, ez_neg = laser_neg._calculate_bound_fields(sim, patch)

        assert_allclose(ey_pos, ey_neg, rtol=1e-10)
        assert_allclose(ez_pos, -ez_neg, rtol=1e-10)

    def test_elliptical_polarization_peak_ratio(self, small_sim):
        """For general ellipticity, the peak ratio of minor/major should equal |epsilon|."""
        sim = small_sim
        eps = 0.5
        ctau = 5e-6
        cep = np.pi / 4 - omega0 * ctau / c
        laser = SimpleLaser2D(a0=1.0, w0=2e-6, ctau=ctau, l0=l0,
                              pol_angle=0.0, ellipticity=eps, cep=cep)

        sim.time = ctau / c
        patch = sim.patches[0]
        ey, ez = laser._calculate_bound_fields(sim, patch)

        peak_major = np.abs(ey).max()
        peak_minor = np.abs(ez).max()
        assert_allclose(peak_minor / peak_major, abs(eps), rtol=1e-10)

    def test_pol_angle_rotates_ellipse_axis(self, small_sim):
        """Changing pol_angle should rotate the major/minor axes of the polarization ellipse."""
        sim = small_sim
        pol = np.pi / 6
        eps = 0.5
        ctau = 5e-6

        # phase = pi/2 -> (ey, ez) lies on the major axis endpoint
        cep_major = np.pi / 2 - omega0 * ctau / c
        laser_major = SimpleLaser2D(a0=1.0, w0=2e-6, ctau=ctau, l0=l0,
                                    pol_angle=pol, ellipticity=eps, cep=cep_major)
        sim.time = ctau / c
        patch = sim.patches[0]
        ey, ez = laser_major._calculate_bound_fields(sim, patch)

        center_idx = ey.shape[0] // 2
        ratio = ez[center_idx] / ey[center_idx]
        assert_allclose(ratio, np.tan(pol), rtol=1e-10)

        # phase = 0 -> (ey, ez) lies on the minor axis endpoint
        cep_minor = -omega0 * ctau / c
        laser_minor = SimpleLaser2D(a0=1.0, w0=2e-6, ctau=ctau, l0=l0,
                                    pol_angle=pol, ellipticity=eps, cep=cep_minor)
        ey, ez = laser_minor._calculate_bound_fields(sim, patch)

        center_idx = ey.shape[0] // 2
        ratio = ez[center_idx] / ey[center_idx]
        assert_allclose(ratio, np.tan(pol + np.pi / 2), rtol=1e-10)

