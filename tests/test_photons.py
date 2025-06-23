import pytest
import numpy as np
from scipy.constants import c, e, epsilon_0, m_e, pi
from lambdapic import Simulation, Electron, Proton, Photon

# Constants
um = 1e-6
l0 = 0.8 * um
omega0 = 2 * np.pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2


@pytest.fixture
def photon_simulation(tmp_path):
    """Fixture to create and configure a photon generation simulation."""
    nx = 128  # Reduced resolution for faster tests
    ny = 128
    dx = l0 / 20
    dy = l0 / 20
    Lx = nx * dx
    Ly = ny * dy

    def density(n0):
        def _density(x, y):
            ne = 0.0
            if x > 10*dx:
                ne = n0
            return ne
        return _density

    # Create simulation
    sim = Simulation(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        npatch_x=8,
        npatch_y=8,
        dt_cfl=0.95
    )

    # Add species
    ele = Electron(density=density(5*nc), ppc=4, radiation="photons")  # Reduced ppc for testing
    proton = Proton(density=density(5*nc), ppc=4)
    pho = Photon()
    ele.set_photon(pho)

    sim.add_species([ele, proton, pho])

    sim.initialize()

    a0 = 100
    w0 = 2e-6
    for p in sim.patches:
        xaxis = p.fields.xaxis
        yaxis = p.fields.yaxis
        x, y = np.meshgrid(xaxis, yaxis, indexing='ij')
        p.fields.ey[:, :] = a0*m_e*c*omega0/e * np.sin(x/(Lx/2) * pi) * np.sin(x/l0 * 2*pi) * np.exp(-(y-Ly/2)**2/w0**2) * (x <= Lx/2)
        p.fields.bz[:, :] = p.fields.ey[:, :]/c

    # Return configured simulation
    return sim

def test_photon_generation(photon_simulation):
    """Test photon generation via QED radiation."""
    sim = photon_simulation
    
    # Run simulation for a few steps
    sim.run(10)
    