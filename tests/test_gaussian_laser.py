import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import c, e, m_e, pi

from lambdapic import Simulation
from lambdapic.callback.laser import GaussianLaser2D

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


class TestGaussianLaserEllipticity:
    """Unit tests for GaussianLaser ellipticity support."""

    def test_linear_polarization_default(self, small_sim):
        """Default ellipticity=0 should give pure linear polarization."""
        sim = small_sim
        laser = GaussianLaser2D(a0=1.0, l0=l0, w0=2e-6, ctau=5e-6, pol_angle=0.0)

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
        laser = GaussianLaser2D(a0=1.0, l0=l0, w0=2e-6, ctau=5e-6, pol_angle=pol)

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
        laser = GaussianLaser2D(a0=1.0, l0=l0, w0=2e-6, ctau=ctau,
                                pol_angle=0.0, ellipticity=1.0)

        sim.time = ctau / c
        patch = sim.patches[0]
        ey, ez = laser._calculate_bound_fields(sim, patch)

        x_rel = sim.cpml_thickness * sim.dx
        boundary_w, boundary_R, boundary_psi = laser._gaussian_beam_params(x_rel)
        r = laser._get_r(sim, patch)
        expected_amp = laser.E0 * (laser.w0 / boundary_w) * np.exp(-r**2 / boundary_w**2)
        tprof = np.exp(-(c * sim.time - laser.x0)**2 / laser.ctau**2)
        A = expected_amp * tprof

        major = minor = 1.0 / np.sqrt(2)
        mask = np.abs(A) > 1e-3 * np.abs(A).max()
        lhs = (ey[mask] / (A[mask] * major))**2 + (ez[mask] / (A[mask] * minor))**2
        assert_allclose(lhs, 1.0, rtol=1e-6)

    def test_circular_polarization_intensity_conservation(self, small_sim):
        """Peak total field for circular pol should be a0/sqrt(2) per component."""
        sim = small_sim
        a0 = 2.0
        ctau = 5e-6
        cep = np.pi / 4 - omega0 * ctau / c
        laser_lin = GaussianLaser2D(a0=a0, l0=l0, w0=2e-6, ctau=ctau,
                                    pol_angle=0.0, ellipticity=0.0, cep=cep)
        laser_circ = GaussianLaser2D(a0=a0, l0=l0, w0=2e-6, ctau=ctau,
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
            GaussianLaser2D(a0=1.0, l0=l0, w0=2e-6, ctau=5e-6, ellipticity=1.5)

        with pytest.raises(ValueError, match="Ellipticity"):
            GaussianLaser2D(a0=1.0, l0=l0, w0=2e-6, ctau=5e-6, ellipticity=-1.5)

    def test_handedness_sign(self, small_sim):
        """Positive and negative ellipticity should produce opposite Ez signs."""
        sim = small_sim
        ctau = 5e-6
        laser_pos = GaussianLaser2D(a0=1.0, l0=l0, w0=2e-6, ctau=ctau,
                                    pol_angle=0.0, ellipticity=1.0)
        laser_neg = GaussianLaser2D(a0=1.0, l0=l0, w0=2e-6, ctau=ctau,
                                    pol_angle=0.0, ellipticity=-1.0)

        sim.time = ctau / c
        patch = sim.patches[0]
        ey_pos, ez_pos = laser_pos._calculate_bound_fields(sim, patch)
        ey_neg, ez_neg = laser_neg._calculate_bound_fields(sim, patch)

        assert_allclose(ey_pos, ey_neg, rtol=1e-10)
        assert_allclose(ez_pos, -ez_neg, rtol=1e-10)

    def test_elliptical_polarization_peak_ratio(self, small_sim):
        """For general ellipticity, verify the polarization ellipse equation."""
        sim = small_sim
        eps = 0.5
        ctau = 5e-6
        laser = GaussianLaser2D(a0=1.0, l0=l0, w0=2e-6, ctau=ctau,
                                pol_angle=0.0, ellipticity=eps)

        sim.time = ctau / c
        patch = sim.patches[0]
        ey, ez = laser._calculate_bound_fields(sim, patch)

        x_rel = sim.cpml_thickness * sim.dx
        boundary_w, boundary_R, boundary_psi = laser._gaussian_beam_params(x_rel)
        r = laser._get_r(sim, patch)
        expected_amp = laser.E0 * (laser.w0 / boundary_w) * np.exp(-r**2 / boundary_w**2)
        tprof = np.exp(-(c * sim.time - laser.x0)**2 / laser.ctau**2)
        A = expected_amp * tprof

        norm = np.sqrt(1 + eps**2)
        major = 1.0 / norm
        minor = eps / norm

        mask = np.abs(A) > 1e-3 * np.abs(A).max()
        lhs = (ey[mask] / (A[mask] * major))**2 + (ez[mask] / (A[mask] * minor))**2
        assert_allclose(lhs, 1.0, rtol=1e-6)

    def test_pol_angle_rotates_ellipse_axis(self, small_sim):
        """Changing pol_angle should rotate the major/minor axes of the polarization ellipse."""
        sim = small_sim
        pol = np.pi / 6
        eps = 0.5
        ctau = 5e-6

        x_rel = sim.cpml_thickness * sim.dx
        laser_ref = GaussianLaser2D(a0=1.0, l0=l0, w0=2e-6, ctau=ctau)
        boundary_w, boundary_R, boundary_psi = laser_ref._gaussian_beam_params(x_rel)
        phase_base = omega0 * ctau / c - laser_ref.k0 * x_rel - boundary_psi

        # For major axis: phase = pi/2 -> sin=1, cos=0
        cep_major = np.pi / 2 - phase_base
        laser_major = GaussianLaser2D(a0=1.0, l0=l0, w0=2e-6, ctau=ctau,
                                      pol_angle=pol, ellipticity=eps, cep=cep_major)
        sim.time = ctau / c
        patch = sim.patches[0]
        ey, ez = laser_major._calculate_bound_fields(sim, patch)

        center_idx = ey.shape[0] // 2
        ratio = ez[center_idx] / ey[center_idx]
        assert_allclose(ratio, np.tan(pol), rtol=1e-2)

        # For minor axis: phase = 0 -> sin=0, cos=1
        cep_minor = -phase_base
        laser_minor = GaussianLaser2D(a0=1.0, l0=l0, w0=2e-6, ctau=ctau,
                                      pol_angle=pol, ellipticity=eps, cep=cep_minor)
        ey, ez = laser_minor._calculate_bound_fields(sim, patch)

        center_idx = ey.shape[0] // 2
        ratio = ez[center_idx] / ey[center_idx]
        assert_allclose(ratio, np.tan(pol + np.pi / 2), rtol=1e-2)
