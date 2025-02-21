from pathlib import Path

import matplotlib.pyplot as plt
import mytools
import numpy as np
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap as _LinearSegmentedColormap
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi

from lambdapic import Electron, Positron, Proton, Photon, Species, Simulation, callback
from lambdapic.callback.laser import SimpleLaser
from lambdapic.callback.utils import ExtractSpeciesDensity, get_fields
import os

import faulthandler
faulthandler.enable()

logger.remove()
f = open('log.txt', 'w')
logger.add(f, level='Timer', format="{level}: {message}")

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
        if abs(x - Lx/2) < 0.05*um:
            ne = n0
        return ne
    return _density

laser1 = SimpleLaser(
    a0=1000,
    w0=2e-6,
    l0=0.8e-6,
    ctau=10e-6,
    side="xmin"
)
laser2 = SimpleLaser(
    a0=1000,
    w0=2e-6,
    l0=0.8e-6,
    ctau=10e-6,
    side="xmax"
)

if __name__ == "__main__":
    sim = Simulation(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        npatch_x=32,
        npatch_y=32,
    )

    ele = Electron(density=density(0.1*nc), ppc=10, radiation="photons")
    pho = Photon()
    pos = Positron()

    ele.set_photon(pho)
    pho.set_bw_pair(electron=ele, positron=pos)

    sim.add_species([ele, pho, pos])

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
                vmax=laser1.a0,
                vmin=-laser1.a0,
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

            figdir = Path('cascade')
            if not figdir.exists():
                figdir.mkdir()

            fig.savefig(figdir/f'{it:04d}.png', dpi=300)
            plt.close()

    @callback()
    def npho(sim: Simulation):
        if sim.itime % 100 != 0:
            return
        npho = 0
        npos = 0
        nevent = 0
        for ipatch, p in enumerate(sim.patches):
            pho_ = p.particles[sim.species.index(pho)]
            npho += pho_.is_alive.sum()
            nevent += pho_.event.sum()
            pos_ = p.particles[sim.species.index(pos)]
            npos += pos_.is_alive.sum()
            
        print(f"{npho=}, {npos=}, {nevent=}")

    
    time = 0
    @callback("start")
    def tic(sim: Simulation):
        from time import perf_counter_ns
        if sim.itime % 100 != 0:
            return
        global time
        time = perf_counter_ns()

    @callback()
    def toc(sim: Simulation):
        from time import perf_counter_ns
        if sim.itime % 100 != 0:
            return
        dt = perf_counter_ns() - time

        npart = 0
        for ipatch, p in enumerate(sim.patches):
            for ispec in range(len(sim.species)):
                part = p.particles[ispec]
                npart += part.is_alive.sum()
        print(f"time: {dt/1e6:.0f} ms, npart: {npart:g}, push time: {dt/npart*os.cpu_count():.0f} ns")

    @callback("start")
    def enable_radiation(sim: Simulation):
        from lambdapic.simulation import NonlinearComptonLCFA
        # disable first
        if sim.itime == 0:
            sim.radiation[0] = None
        # enable at itime 200
        if sim.itime == 200:
            sim.radiation[0] = NonlinearComptonLCFA(sim.patches, 0)

    @callback()
    def prune(sim: Simulation):
        if sim.itime % 100 != 0:
            return
        for ipatch, p in enumerate(sim.patches):
            for ispec in range(len(sim.species)):
                part = p.particles[ispec]
                part.prune()
        sim.update_lists()
        sim.patches.update_lists()
    
    sim.run(1501, callbacks=[
            tic,
            toc,
            laser1, 
            laser2, 
            store_ne, 
            npho,
            plot_results,
            # prune,
            # enable_radiation
        ]
    )
