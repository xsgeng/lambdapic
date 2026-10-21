import pytest
import numpy as np
from scipy.constants import c, e, epsilon_0, m_e, pi
from lambdapic import Simulation, Simulation3D, Electron, Proton
from lambdapic.callback.utils import MovingWindow
from lambdapic.core.boundary.cpml import PMLX, PMLXmin, PMLXmax

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


def _assert_x_pml_placement(sim):
    """X-direction PMLs must sit on the current left/right edge patches only."""
    for p in sim.patches:
        pmls_x = [pml for pml in p.pml_boundary if isinstance(pml, PMLX)]
        if p.ipatch_x == 0:
            assert len(pmls_x) == 1, "leftmost patch must carry exactly one X PML"
            assert isinstance(pmls_x[0], PMLXmin)
        elif p.ipatch_x == sim.npatch_x - 1:
            assert len(pmls_x) == 1, "rightmost patch must carry exactly one X PML"
            assert isinstance(pmls_x[0], PMLXmax)
        else:
            assert len(pmls_x) == 0, "interior patch must not carry an X PML"


def test_moving_window_keeps_x_pml_on_edge_patches(lwfa_simulation2d):
    sim, win = lwfa_simulation2d
    # With start_time=0, the first MovingWindow call shifts the window by one patch
    sim.run(1, callbacks=[win])
    _assert_x_pml_placement(sim)


def test_moving_window_keeps_x_pml_on_edge_patches_3d(lwfa_simulation3d):
    sim, win = lwfa_simulation3d
    sim.run(1, callbacks=[win])
    _assert_x_pml_placement(sim)


def test_moving_window_fills_former_pml_strip(lwfa_simulation2d):
    sim, win = lwfa_simulation2d

    # Warmup without the window: the particle sync removes every particle
    # inside the initial PML shells (beyond the half-cell tolerance the
    # particle sync grants at patch edges), so the shells are particle-free
    sim.run(2)
    for p in sim.patches:
        if p.ipatch_x == sim.npatch_x - 1:
            part = p.particles[0]
            in_pml = part.x > p.xmax + p.dx / 2  # p.xmax excludes the PML shell
            assert not np.any(part.is_alive & in_pml)

    # First MovingWindow call shifts the window by one patch to the right;
    # the former PML strip of the old rightmost patches becomes physical
    sim.run(1, callbacks=[win])

    # The old rightmost patches are interior now (ipatch_x == npatch_x - 2):
    # their formerly-PML tail must be repopulated with alive electrons,
    # otherwise a vacuum strip drifts through the plasma
    ppc = 4  # electron ppc of the lwfa_simulation2d fixture
    for p in sim.patches:
        if p.ipatch_x == sim.npatch_x - 2:
            part = p.particles[0]
            strip = part.x >= p.xaxis[-sim.cpml_thickness] - p.dx / 2
            n_expected = sim.cpml_thickness * sim.ny_per_patch * ppc
            assert np.sum(part.is_alive & strip) >= n_expected / 2


def test_moving_window_pml_shells_stay_particle_free(lwfa_simulation2d):
    sim, win = lwfa_simulation2d
    # Long enough for two window shifts
    sim.run(sim.nx_per_patch * 2, callbacks=[win])

    # Particles that end up inside the (retained) PML shells are removed by
    # the particle sync, including the ones freshly loaded near the new edge
    for p in sim.patches:
        for part in p.particles:
            in_pml = (part.x < p.xmin - p.dx / 2) | (part.x > p.xmax + p.dx / 2)
            assert not np.any(part.is_alive & in_pml)


def test_moving_window_marks_pml_shell_particles_dead_immediately(lwfa_simulation2d):
    sim, win = lwfa_simulation2d
    # Invoke the callback exactly as the run loop does at the start stage
    win(sim)

    # Particles inside the regenerated PML shells (the freshly loaded ones in
    # the wrapped patch and the old ones now covered by the new PMLXmin) must
    # already be dead: they must not be pushed or deposit current inside the
    # PML during the remainder of the step
    for p in sim.patches:
        for part in p.particles:
            in_pml = (part.x < p.xmin - p.dx / 2) | (part.x > p.xmax + p.dx / 2)
            assert not np.any(part.is_alive & in_pml)


def test_moving_window_left_fills_former_pml_strip():
    # Uniform density: the shared fixture has no plasma at the left edge,
    # but the former left PML strip must contain plasma to be observable
    n0 = 0.01 * nc
    sim = Simulation(
        nx=128, ny=128,
        dx=l0 / 20, dy=l0 / 20,
        npatch_x=8, npatch_y=8,
        dt_cfl=0.95,
    )
    ele = Electron(density=lambda x, y: n0, ppc=4)
    sim.add_species([ele])
    sim.initialize()

    win = MovingWindow(-c, start_time=0.0)
    # The first left shift needs two patch widths of accumulated movement
    sim.run(50, callbacks=[win])
    assert sim.patches.xmin_global < 0, "window must have shifted left by one patch"
    _assert_x_pml_placement(sim)

    # The old leftmost patches are interior now (ipatch_x == 1): their
    # formerly-PML head must be repopulated with alive electrons
    ppc = 4
    for p in sim.patches:
        if p.ipatch_x == 1:
            part = p.particles[0]
            strip = part.x <= p.xaxis[sim.cpml_thickness - 1] + p.dx / 2
            n_expected = sim.cpml_thickness * sim.ny_per_patch * ppc
            assert np.sum(part.is_alive & strip) >= n_expected / 2
    