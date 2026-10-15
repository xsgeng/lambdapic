from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from lambdapic import (
    Electron,
    ExtractSpeciesDensity,
    MovingWindow,
    Photon,
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
from lambdapic.core.utils.logger import logger

um = 1e-6
l0 = 0.8 * um
omega0 = 2 * pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2

nx = 512
ny = 512
dx = l0 / 20
dy = l0 / 20

Lx = nx * dx
Ly = ny * dy


def density(n0):
    def _density(x, y):
        ne = 0.0
        if x > 2*um:
            ne = n0
        return ne
    return _density

laser = SimpleLaser2D(
    a0=300,
    w0=2e-6,
    l0=0.8e-6,
    ctau=5e-6
)

sim = Simulation(
    nx=nx,
    ny=ny,
    dx=dx,
    dy=dy,
    sim_time=100e-15,
)

ele = Electron(density=density(5*nc), ppc=10, radiation="photons")
pho = Photon()
ele.set_photon(pho)

proton = Proton(density=density(5*nc), ppc=10)

sim.add_species([ele, proton, pho])

interval = 10e-15

@callback(interval=interval)
def npho(sim: Simulation):
    npart = 0
    for ipatch, p in enumerate(sim.patches):
        part = p.particles[pho.ispec]
        npart += part.is_alive.sum()
    
    npart = sim.mpi.comm.reduce(npart)
    if sim.mpi.rank == 0:
        logger.info(f"nphoton = {npart}")

@callback("current deposition", interval=interval)
def prune(sim: Simulation):
    for ipatch, p in enumerate(sim.patches):
        p.particles[sim.ispec].prune()
    
if __name__ == "__main__":
    sim.run(callbacks=[
            laser, 
            n_ele := ExtractSpeciesDensity(sim, ele, interval),
            PlotFields(
                [
                    dict(field=n_ele.density, scale=1/nc, cmap='Grays', vmin=0, vmax=10), 
                    dict(field='ey',  scale=e/(m_e*c*omega0), cmap='bwr_alpha', vmin=-laser.a0, vmax=laser.a0)
                ],
                prefix='photons', interval=interval,
            ),
            npho,
            prune,
        ]
    )
