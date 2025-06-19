from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi

from lambdapic import Electron, Proton, Simulation, Species, callback
from lambdapic.callback.hdf5 import SaveFieldsToHDF5, SaveSpeciesDensityToHDF5
from lambdapic.callback.laser import GaussianLaser2D
from lambdapic.callback.utils import get_fields

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

laser = GaussianLaser2D(
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

    @callback()
    def plot_results(sim: Simulation):
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
            fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")

            h1 = ax.imshow(
                -rho.T/e/nc, 
                extent=[0, Lx, 0, Ly],
                origin='lower',
                cmap='Grays',
                vmax=20,
                vmin=0,
            )
            h2 = ax.imshow(
                ey.T, 
                extent=[0, Lx, 0, Ly],
                origin='lower',
                cmap=bwr_alpha,
                vmax=laser.a0,
                vmin=-laser.a0,
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
            plot_results,
            SaveFieldsToHDF5('laser-target/fields', 100, ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'rho']),
            SaveSpeciesDensityToHDF5(carbon, 'laser-target/density', 100),
            SaveSpeciesDensityToHDF5(ele, 'laser-target/density', 100),
        ]
    )
