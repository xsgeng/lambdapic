"""Tests for SetMomentum and SetMomentumAndTemperature callbacks."""
import numpy as np
import pytest
from scipy.constants import c, epsilon_0, e, m_e, pi

from lambdapic import Electron, Simulation
from lambdapic.callback.utils import SetMomentum, SetMomentumAndTemperature

l0 = 0.8e-6
omega0 = 2 * pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2


def _gather_species_momenta(sim, species):
    """Collect alive-particle momenta and inv_gamma across all patches."""
    ispec = species.ispec
    ux, uy, uz, inv_gamma = [], [], [], []
    for p in sim.patches:
        part = p.particles[ispec]
        alive = part.is_alive
        if not alive.any():
            continue
        ux.append(part.ux[alive])
        uy.append(part.uy[alive])
        uz.append(part.uz[alive])
        inv_gamma.append(part.inv_gamma[alive])
    return (
        np.concatenate(ux),
        np.concatenate(uy),
        np.concatenate(uz),
        np.concatenate(inv_gamma),
    )


@pytest.mark.unit
class TestSetMomentumInvGamma:
    """Verify inv_gamma is consistent with the stored momenta."""

    def setup_method(self) -> None:
        self.nx = 64
        self.ny = 64
        self.dx = l0 / 20
        self.dy = l0 / 20

        self.sim = Simulation(
            nx=self.nx,
            ny=self.ny,
            dx=self.dx,
            dy=self.dy,
            npatch_x=4,
            npatch_y=4,
            boundary_conditions={
                "xmin": "periodic",
                "xmax": "periodic",
                "ymin": "periodic",
                "ymax": "periodic",
            },
        )

        ne = 0.01 * nc
        self.ele = Electron(density=lambda x, y: ne, ppc=8)
        self.sim.add_species([self.ele])
        self.sim.initialize()

    def test_add_inv_gamma_matches_total_momentum(self) -> None:
        """With add=True, inv_gamma must reflect old + delta, not the delta alone."""
        SetMomentum(self.ele, [3.0, 0.0, 0.0], add=False)(self.sim)
        SetMomentum(self.ele, [1.0, 2.0, 0.0], add=True)(self.sim)

        ux, uy, uz, inv_gamma = _gather_species_momenta(self.sim, self.ele)

        # Resulting momentum is old (3,0,0) + delta (1,2,0) = (4,2,0).
        expected_inv_gamma = 1.0 / np.sqrt(1.0 + ux**2 + uy**2 + uz**2)
        np.testing.assert_allclose(inv_gamma, expected_inv_gamma, rtol=1e-12)
        np.testing.assert_allclose(ux, 4.0, rtol=1e-12)
        np.testing.assert_allclose(uy, 2.0, rtol=1e-12)

    def test_set_inv_gamma_matches_target(self) -> None:
        """With add=False (default), inv_gamma matches the set target."""
        SetMomentum(self.ele, [3.0, 4.0, 0.0])(self.sim)

        ux, uy, uz, inv_gamma = _gather_species_momenta(self.sim, self.ele)

        expected_inv_gamma = 1.0 / np.sqrt(1.0 + ux**2 + uy**2 + uz**2)
        np.testing.assert_allclose(inv_gamma, expected_inv_gamma, rtol=1e-12)
        np.testing.assert_allclose(inv_gamma, 1.0 / np.sqrt(26.0), rtol=1e-12)


@pytest.mark.unit
class TestSetMomentumAndTemperature:
    """Verify thermal spread survives the bulk-momentum set."""

    def setup_method(self) -> None:
        self.nx = 64
        self.ny = 64
        self.dx = l0 / 20
        self.dy = l0 / 20

        self.sim = Simulation(
            nx=self.nx,
            ny=self.ny,
            dx=self.dx,
            dy=self.dy,
            npatch_x=4,
            npatch_y=4,
            boundary_conditions={
                "xmin": "periodic",
                "xmax": "periodic",
                "ymin": "periodic",
                "ymax": "periodic",
            },
        )

        ne = 0.01 * nc
        self.ele = Electron(density=lambda x, y: ne, ppc=8)
        self.sim.add_species([self.ele])
        self.sim.initialize()

    def test_thermal_spread_survives_default_add_false(self) -> None:
        """Default add=False yields a thermal beam with bulk drift, not a cold beam."""
        ux_bulk = 10.0
        kT_eV = 1.0e4  # 10 keV thermal spread

        SetMomentumAndTemperature(self.ele, [ux_bulk, 0.0, 0.0], kT_eV)(self.sim)

        ux, uy, uz, inv_gamma = _gather_species_momenta(self.sim, self.ele)

        assert ux.std() > 1e-6, "ux has no thermal spread"
        assert uy.std() > 1e-6, "uy has no thermal spread"

        np.testing.assert_allclose(ux.mean(), ux_bulk, rtol=5e-2)
        np.testing.assert_allclose(uy.mean(), 0.0, atol=5e-2)

        expected_inv_gamma = 1.0 / np.sqrt(1.0 + ux**2 + uy**2 + uz**2)
        np.testing.assert_allclose(inv_gamma, expected_inv_gamma, rtol=1e-12)

    def test_add_true_preserves_thermal_plus_bulk(self) -> None:
        """With add=True, the result is existing + bulk + thermal with correct inv_gamma."""
        SetMomentum(self.ele, [5.0, 0.0, 0.0], add=False)(self.sim)

        ux_bulk_delta = 5.0
        kT_eV = 5.0e3

        SetMomentumAndTemperature(
            self.ele, [ux_bulk_delta, 0.0, 0.0], kT_eV, add=True
        )(self.sim)

        ux, uy, uz, inv_gamma = _gather_species_momenta(self.sim, self.ele)

        assert ux.std() > 1e-6, "thermal spread lost"
        np.testing.assert_allclose(ux.mean(), 10.0, rtol=5e-2)

        expected_inv_gamma = 1.0 / np.sqrt(1.0 + ux**2 + uy**2 + uz**2)
        np.testing.assert_allclose(inv_gamma, expected_inv_gamma, rtol=1e-12)
