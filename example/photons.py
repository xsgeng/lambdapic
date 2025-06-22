from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi

from lambdapic import Electron, Photon, Proton, Simulation, Species, callback
from lambdapic.callback.laser import SimpleLaser2D
from lambdapic.callback.utils import get_fields
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

if __name__ == "__main__":
    sim = Simulation(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        npatch_x=16,
        npatch_y=16,
    )

    ele = Electron(density=density(5*nc), ppc=10, radiation="photons")
    pho = Photon()
    ele.set_photon(pho)

    proton = Proton(density=density(5*nc), ppc=10)

    sim.add_species([ele, proton, pho])

    @callback()
    def plot_results(sim):
        it = sim.itime
        if it % 100 == 0:
            ex, ey, ez, bx, by, bz, jy, rho = get_fields(sim, ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'jy', 'rho'])
            if sim.mpi.rank > 0:
                return
            ey *= e / (m_e * c * omega0)
            
            bwr_alpha = LinearSegmentedColormap(
                'bwr_alpha', 
                dict( 
                    red=[ (0, 0, 0), (0.5, 1, 1), (1, 1, 1) ], 
                    green=[ (0, 0.5, 0), (0.5, 1, 1), (1, 0, 0) ], 
                    blue=[ (0, 1, 1), (0.5, 1, 1), (1, 0, 0) ], 
                    alpha = [ (0, 1, 1), (0.5, 0, 0), (1, 1, 1) ]
                )
            )

            fig, axes = plt.subplots(2, 1, figsize=(5, 5), layout="constrained")
            
            ax = axes[0]
            
            h2 = ax.imshow(
                -rho.T/e/nc, 
                extent=[0, Lx, 0, Ly],
                origin='lower',
                cmap='Grays',
                vmax=10,
                vmin=0,
            )
            h1 = ax.imshow(
                ey.T, 
                extent=[0, Lx, 0, Ly],
                origin='lower',
                cmap=bwr_alpha,
                vmax=laser.a0,
                vmin=-laser.a0,
            )
            fig.colorbar(h1)
            fig.colorbar(h2)

            figdir = Path('qed')
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
            part = p.particles[pho.ispec]
            npart += part.is_alive.sum()
        
        npart = sim.mpi.comm.reduce(npart)
        if sim.mpi.rank == 0:
            logger.info(f"nphoton = {npart}")

    @callback("current deposition")
    def prune(sim: Simulation):
        if sim.itime % 100 != 0:
            return
        for ipatch, p in enumerate(sim.patches):
            p.particles[sim.ispec].prune()

    @callback("start")
    def enable_radiation(sim: Simulation):
        from lambdapic.simulation import NonlinearComptonLCFA
        # disable first
        if sim.itime == 0:
            sim.radiation[0] = None
        # enable at itime 200
        if sim.itime == 200:
            sim.radiation[0] = NonlinearComptonLCFA(sim.patches, ele.ispec)

    
    sim.run(1001, callbacks=[
            laser, 
            plot_results,
            # npho, 
            prune,
            # enable_radiation
        ]
    )
