from pathlib import Path

import matplotlib.pyplot as plt
import mytools
import numpy as np
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap as _LinearSegmentedColormap
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi

from lambdapic import Electron, Proton, Photon, Species, Simulation, callback
from lambdapic.callback.laser import SimpleLaser
from lambdapic.callback.utils import ExtractSpeciesDensity, get_fields

import faulthandler
faulthandler.enable()

logger.remove()

um = 1e-6
l0 = 0.8 * um
t0 = l0 / c
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
        if x > 1*um and x < 10*um:
            ne = n0
        return ne
    return _density

laser = SimpleLaser(
    a0=300,
    w0=2e-6,
    l0=0.8e-6,
    ctau=5e-6
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

    ele = Electron(density=density(5*nc), ppc=1, radiation="photons")
    pho = Photon()
    ele.set_photon(pho)

    proton = Proton(density=density(5*nc), ppc=1)

    sim.add_species([ele, proton, pho])

    store_ne = ExtractSpeciesDensity(sim, ele, every=100)

    @callback()
    def plot_results(sim):
        it = sim.itime
        if it % 100 == 0:
            ex, ey, ez, bx, by, bz, jy, rho = get_fields(sim, ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'jy', 'rho'])
            
            ey *= e / (m_e * c * omega0)

            fig, axes = plt.subplots(2, 1, figsize=(5, 5), layout="constrained")

            ax = axes[0]
            h1 = ax.imshow(
                ey.T, 
                extent=[0, Lx, 0, Ly],
                origin='lower',
                cmap="bwr",
                vmax=laser.a0,
                vmin=-laser.a0,
            )
            h2 = ax.imshow(
                store_ne.density.T/nc, 
                extent=[0, Lx, 0, Ly],
                origin='lower',
                cmap=mytools.cmap.grey_alpha,
                # vmax=5,
                # vmin=0,
            )
            fig.colorbar(h1)
            fig.colorbar(h2)

            ax = axes[1]
            x_pho = []
            y_pho = []
            for p in sim.patches:
                pho_ = p.particles[sim.species.index(pho)]
                alive = np.logical_not(pho_.is_dead)
                x_pho.append(pho_.x[alive])
                y_pho.append(pho_.y[alive])
            x_pho = np.concatenate(x_pho)
            y_pho = np.concatenate(y_pho)
            ax.hist2d(x_pho, y_pho, bins=128)

            figdir = Path('laser-target')
            if not figdir.exists():
                figdir.mkdir()

            fig.savefig(figdir/f'{it:04d}.png', dpi=300)
            plt.close()

    @callback()
    def npho(sim: Simulation):
        if sim.itime % 100 != 0:
            return
        npart = 0
        for ipatch, p in enumerate(sim.patches):
            part = p.particles[sim.species.index(pho)]
            npart += np.logical_not(part.is_dead).sum()
        print(f"{npart=}")

    @callback("start")
    def enable_radiation(sim: Simulation):
        from lambdapic.simulation import NonlinearComptonLCFA
        # disable first
        if sim.itime == 0:
            sim.radiation[0] = None
        # enable at itime 200
        if sim.itime == 200:
            sim.radiation[0] = NonlinearComptonLCFA(sim.patches, 0)

    
    sim.run(1001, callbacks=[
            laser, 
            store_ne, 
            plot_results,
            npho,
            # enable_radiation
        ]
    )
