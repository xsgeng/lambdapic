"""Tests for numerical heating in Simulation2D.

This test verifies that the total energy (particle kinetic + field energy)
remains conserved in a plasma without external fields, which is a fundamental
property that any PIC code should satisfy.
"""

import pytest
from scipy.constants import c, e, epsilon_0, m_e, m_p, mu_0, pi

from lambdapic import Electron, SetTemperature, Simulation, Species

# Physical constants and parameters
l0 = 0.8e-6  # wavelength
omega0 = 2 * pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2  # Critical density


def calculate_total_energy(sim: Simulation) -> float:
    """Calculate total energy (kinetic + field) in the simulation.

    Args:
        sim: The simulation instance.

    Returns:
        Total energy in joules.
    """
    e_tot = 0.0

    for p in sim.patches:
        for s in sim.species:
            part = p.particles[s.ispec]
            if part.npart > 0:
                gamma = 1.0 / part.inv_gamma
                e_tot += (part.w * (gamma - 1) * s.m * c**2).sum(where=part.is_alive)

        e_tot += (
            0.5 * epsilon_0 * (
                p.fields.ex[:p.nx, :p.ny]**2 +
                p.fields.ey[:p.nx, :p.ny]**2 +
                p.fields.ez[:p.nx, :p.ny]**2
            ) +
            0.5 / mu_0 * (
                p.fields.bx[:p.nx, :p.ny]**2 +
                p.fields.by[:p.nx, :p.ny]**2 +
                p.fields.bz[:p.nx, :p.ny]**2
            )
        ).sum() * sim.dx * sim.dy

    return e_tot


@pytest.mark.unit
class TestNumericalHeatingSimulation2D:
    """Test numerical heating conservation for standard Simulation2D."""

    def setup_method(self) -> None:
        """Set up a 2D simulation with electrons and deuterium ions."""
        # Use smaller grid for unit testing
        self.nx = 64
        self.ny = 64
        self.dx = l0 / 20
        self.dy = l0 / 20

        self.npatch_x = 4
        self.npatch_y = 4

        # Create simulation
        self.sim = Simulation(
            nx=self.nx,
            ny=self.ny,
            dx=self.dx,
            dy=self.dy,
            npatch_x=self.npatch_x,
            npatch_y=self.npatch_y,
            boundary_conditions={
                "xmin": "periodic",
                "xmax": "periodic",
                "ymin": "periodic",
                "ymax": "periodic",
            },
            sim_time=1e-15,
        )

        # Plasma density
        ne = 1 * nc

        # Create species
        self.ele = Electron(
            density=lambda x, y: ne,
            ppc=10
        )
        self.deut = Species(
            name="D",
            charge=1,
            mass=2 * m_p / m_e,
            density=lambda x, y: ne,
            ppc=10
        )

        self.sim.add_species([self.ele, self.deut])

    def test_energy_conservation_simulation2d(self):
        """Test that total energy is conserved in Simulation2D.

        The test initializes a plasma with thermal energy and runs
        for several timesteps, verifying that total energy change
        is within acceptable numerical error bounds.
        """
        # Initialize simulation
        self.sim.initialize()

        # Set temperature (1 keV)
        kT = 1e3 * e
        SetTemperature(self.ele, kT / e)(self.sim)
        SetTemperature(self.deut, kT / e)(self.sim)

        # Calculate initial total energy
        e_initial = calculate_total_energy(self.sim)

        self.sim.run()

        # Calculate final total energy
        e_final = calculate_total_energy(self.sim)

        # Calculate relative energy change
        rel_change = abs(e_final - e_initial) / e_initial

        # Energy should be conserved within 1% (allowing for numerical heating)
        assert rel_change < 0.01, (
            f"Energy not conserved: initial={e_initial:.6e}, "
            f"final={e_final:.6e}, relative change={rel_change:.6e}"
        )
