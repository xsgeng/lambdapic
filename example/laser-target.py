from pathlib import Path

import matplotlib.pyplot as plt
import mytools
import numpy as np
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap as _LinearSegmentedColormap
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi

from lambdapic import Electron, Proton, Species, Simulation, callback
from lambdapic.callback.laser import SimpleLaser, GaussianLaser
from lambdapic.callback.utils import ExtractSpeciesDensity, get_fields

logger.remove()

um = 1e-6
l0 = 0.8 * um
t0 = l0 / c
omega0 = 2 * pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2

nx = 1024
ny = 1024
dx = l0 / 50
dy = l0 / 50

Lx = nx * dx
Ly = ny * dy


def density(n0):
    def _density(x, y):
        ne = 0.0
        if x > Lx/2 and x < Lx/2+1*um:
            ne = n0
        return ne
    return _density

laser = GaussianLaser(
    a0=10,
    w0=2e-6,
    l0=0.8e-6,
    ctau=5e-6,
    focus_position=Lx/2,
)

if __name__ == "__main__":
    sim = Simulation(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        npatch_x=16,
        npatch_y=16,
    )

    ele = Electron(density=density(10*nc), ppc=10)
    proton = Proton(density=density(10*nc/8*2), ppc=10)
    carbon = Species(name="C", charge=6, mass=12*1800, density=density(10*nc/8), ppc=10)

    sim.add_species([ele, carbon, proton])

    store_ne = ExtractSpeciesDensity(sim, ele, every=100)
    store_proton = ExtractSpeciesDensity(sim, proton, every=100)

    @callback()
    def plot_results(sim):
        it = sim.itime
        if it % 100 == 0:
            ex, ey, ez, bx, by, bz, jy, rho = get_fields(sim, ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'jy', 'rho'])
            
            ey *= e / (m_e * c * omega0)

            fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")

            h1 = ax.imshow(
                ey.T, 
                extent=[0, Lx, 0, Ly],
                origin='lower',
                cmap="bwr",
                vmax=laser.a0,
                vmin=-laser.a0,
            )
            h2 = ax.imshow(
                store_proton.density.T/nc, 
                extent=[0, Lx, 0, Ly],
                origin='lower',
                cmap=mytools.cmap.grey_alpha,
                vmax=5,
                vmin=0,
            )
            fig.colorbar(h1)
            fig.colorbar(h2)

            figdir = Path('laser-target')
            if not figdir.exists():
                figdir.mkdir()

            fig.savefig(figdir/f'{it:04d}.png', dpi=300)
            plt.close()

    
    sim.run(2001, callbacks=[
            laser, 
            store_ne, 
            store_proton, 
            plot_results,
        ]
    )
