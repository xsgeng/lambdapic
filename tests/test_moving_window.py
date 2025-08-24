import pytest
import numpy as np
from scipy.constants import c, e, epsilon_0, m_e, pi
from lambdapic import Simulation, Simulation3D, Electron, Proton
from lambdapic.callback.utils import MovingWindow

# Constants
um = 1e-6
l0 = 0.8 * um
omega0 = 2 * np.pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2


@pytest.fixture
def lwfa_simulation2d(tmp_path):
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
    ele = Electron(density=density(0.01*nc), ppc=4)  # Reduced ppc for testing
    proton = Proton(density=density(0.01*nc), ppc=4)

    sim.add_species([ele, proton])

    sim.initialize()

    a0 = 100
    w0 = 2e-6
    for p in sim.patches:
        xaxis = p.fields.xaxis
        yaxis = p.fields.yaxis
        x, y = np.meshgrid(xaxis, yaxis, indexing='ij')
        p.fields.ey[:, :] = a0*m_e*c*omega0/e * np.sin(x/(Lx/2) * pi) * np.sin(x/l0 * 2*pi) * np.exp(-(y-Ly/2)**2/w0**2) * (x >= Lx/2)
        p.fields.bz[:, :] = p.fields.ey[:, :]/c

    win = MovingWindow(c, start_time=0.0)
    # Return configured simulation
    return sim, win

@pytest.fixture
def lwfa_simulation3d(tmp_path):
    nx = 32  # Reduced resolution for faster tests
    ny = 32
    nz = 32
    dx = l0 / 20
    dy = l0 / 20
    dz = l0 / 20
    Lx = nx * dx
    Ly = ny * dy
    Lz = nz * dz

    def density(n0):
        def _density(x, y, z):
            ne = 0.0
            if x > 10*dx:
                ne = n0
            return ne
        return _density

    # Create simulation
    sim = Simulation3D(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx,
        dy=dy,
        dz=dz,
        npatch_x=4,
        npatch_y=4,
        npatch_z=4,
        dt_cfl=0.95
    )

    # Add species
    ele = Electron(density=density(0.01*nc), ppc=4)  # Reduced ppc for testing
    proton = Proton(density=density(0.01*nc), ppc=4)

    sim.add_species([ele, proton])

    sim.initialize()

    a0 = 100
    w0 = 2e-6
    for p in sim.patches:
        xaxis = p.fields.xaxis
        yaxis = p.fields.yaxis
        zaxis = p.fields.zaxis
        x, y, z = np.meshgrid(xaxis, yaxis, zaxis, indexing='ij')
        r2 = (y-Ly/2)**2 + (z-Lz/2)**2
        p.fields.ey[:, :, :] = a0*m_e*c*omega0/e * np.sin(x/(Lx/2) * pi) * np.sin(x/l0 * 2*pi) * np.exp(-r2/w0**2) * (x >= Lx/2)
        p.fields.bz[:, :, :] = p.fields.ey[:, :, :]/c

    win = MovingWindow(c, start_time=0.0)
    # Return configured simulation
    return sim, win

def test_lwfa(lwfa_simulation2d):
    sim, win = lwfa_simulation2d
    
    # Run simulation for a few steps
    sim.run(sim.nx_per_patch*2, callbacks=[win])
    
def test_lwfa_3d(lwfa_simulation3d):
    sim, win = lwfa_simulation3d
    
    # Run simulation for a few steps
    sim.run(sim.nx_per_patch*2, callbacks=[win])
    