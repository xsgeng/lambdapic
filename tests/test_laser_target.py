import pytest
import numpy as np
from scipy.constants import c, e, epsilon_0, m_e
from lambdapic import Simulation, Electron, Proton, Species
from lambdapic.callback.laser import GaussianLaser2D

# Constants
um = 1e-6
l0 = 0.8 * um
t0 = l0 / c
omega0 = 2 * np.pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2

@pytest.fixture
def laser_simulation(tmp_path):
    """Fixture to create and configure a laser-target simulation."""
    nx = 256  # Reduced resolution for faster tests
    ny = 256
    dx = l0 / 20
    dy = l0 / 20
    Lx = nx * dx
    Ly = ny * dy

    def density(n0):
        def _density(x, y):
            ne = 0.0
            if x > Lx/2 and x < Lx/2+1*um:
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

    # Setup laser
    laser = GaussianLaser2D(
        a0=5,  # Reduced intensity for testing
        w0=2e-6,
        l0=0.8e-6,
        ctau=5e-6,
        focus_position=Lx/2,
    )

    # Add species
    ele = Electron(density=density(10*nc), ppc=4)  # Reduced ppc for testing
    proton = Proton(density=density(10*nc/8*2), ppc=4)
    carbon = Species(name="C", charge=6, mass=12*1800, density=density(10*nc/8), ppc=4)

    sim.add_species([ele, carbon, proton])

    # Setup output directory
    output_dir = tmp_path / "laser_target_output"
    output_dir.mkdir()

    # Return configured simulation and paths
    return {
        'sim': sim,
        'laser': laser,
        'output_dir': output_dir,
        'species': [ele, carbon, proton]
    }

def test_laser_target_simulation(laser_simulation):
    """Test the laser-target interaction simulation."""
    sim = laser_simulation['sim']
    
    sim.run(10)