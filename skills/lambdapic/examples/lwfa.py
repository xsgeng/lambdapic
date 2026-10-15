import numpy as np
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi

from lambdapic import (
    Electron,
    ExtractSpeciesDensity,
    MovingWindow,
    PlotFields,
    Proton,
    SaveFieldsToHDF5,
    SaveSpeciesDensityToHDF5,
    SimpleLaser2D,
    Simulation,
    Species,
    c,
    callback,
    e,
    epsilon_0,
    get_fields,
    m_e,
    pi,
)

um = 1e-6
l0 = 0.8 * um
t0 = l0 / c
omega0 = 2 * pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2

nx = 500
ny = 800
dx = l0 / 20
dy = l0 / 20

Lx = nx * dx
Ly = ny * dy


def density(n0):
    def _density(x, y):
        ne = 0.0
        if x > 1*um:
            ne = n0
        if abs(y - Ly/2) > Ly/2 - 1*um:
            ne = 0
        return ne
    return _density

# move velocity supports constant velocity and time-dependent velocity
# here lambda t: c + (t-Lx/c)*0 is just a constant for demonstration
movingwindow = MovingWindow(velocity=lambda t: c + (t-Lx/c)*0)

laser = SimpleLaser2D(
    a0=2,
    w0=5e-6,
    l0=0.8e-6,
    ctau=5e-6,
)

ne = 0.01*nc

sim = Simulation(
    nx=nx,
    ny=ny,
    dx=dx,
    dy=dy,
    npatch_x=10,
    npatch_y=10,
    dt_cfl=0.99,
    sim_time=100e-15
)

ele = Electron(density=density(ne), ppc=10)
proton = Proton(density=density(ne/8*2), ppc=2)
carbon = Species(name="C", charge=6, mass=12*1800, density=density(ne/8), ppc=1)

sim.add_species([ele, carbon, proton])

interval = 10e-15
if __name__ == "__main__":
    sim.run(2001, callbacks=[
            movingwindow,
            laser,
            n_ele := ExtractSpeciesDensity(sim, ele, interval),
            PlotFields(
                [
                    dict(field=n_ele.density, scale=1/nc, cmap='Grays', vmin=0, vmax=ne/nc*2), 
                    dict(field='ey',  scale=e/(m_e*c*omega0), cmap='bwr_alpha', vmin=-laser.a0, vmax=laser.a0)
                ],
                prefix='lwfa', interval=interval,
            ),
            SaveFieldsToHDF5('lwfa/fields', interval, ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'jx', 'jy', 'rho']),
            SaveSpeciesDensityToHDF5(proton, 'lwfa/', interval),
        ]
    )
