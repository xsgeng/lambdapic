import faulthandler
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numpy import vectorize
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi

from lambdapic import Electron, Proton, Simulation, Species, callback
from lambdapic.callback.hdf5 import SaveFieldsToHDF5, SaveSpeciesDensityToHDF5
from lambdapic.callback.laser import GaussianLaser2D, SimpleLaser2D
from lambdapic.callback.utils import MovingWindow, get_fields

faulthandler.enable()

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

if __name__ == "__main__":
    sim = Simulation(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        npatch_x=10,
        npatch_y=10,
    )

    ele = Electron(density=density(ne), ppc=10)
    proton = Proton(density=density(ne/8*2), ppc=2)
    carbon = Species(name="C", charge=6, mass=12*1800, density=density(ne/8), ppc=1)

    sim.add_species([ele, carbon, proton])

    @callback()
    def plot_results(sim: Simulation):
        it = sim.itime
        if it % 100 == 0:
            if sim.mpi.rank > 0:
                return
            
            nx_moved = int(movingwindow.patch_this_shift // dx)
            s = np.s_[nx_moved:nx_moved+(sim.npatch_x-1)*sim.nx_per_patch, :]
            
            with h5py.File(f'lwfa/{ele.name}_t{sim.itime:06d}.h5', 'r', locking=False) as f:
                nele = f['density'][s]

            with h5py.File(f'lwfa/fields_t{sim.itime:06d}.h5', 'r', locking=False) as f:
                ey = f['ey'][s] * e / (m_e * c * omega0)
                
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

            extent = [
                movingwindow.total_shift,
                movingwindow.total_shift + Lx*(sim.npatch_x-1)/sim.npatch_x,
                0,
                Ly
            ]

            h1 = ax.imshow(
                nele.T/nc, 
                extent=extent,
                origin='lower',
                cmap='Grays',
                vmax=ne/nc*2,
                vmin=0,
            )
            h2 = ax.imshow(
                ey.T, 
                extent=extent,
                origin='lower',
                cmap=bwr_alpha,
                vmax=laser.a0,
                vmin=-laser.a0,
            )
            fig.colorbar(h1)
            fig.colorbar(h2)

            figdir = Path('lwfa')
            if not figdir.exists():
                figdir.mkdir()

            fig.savefig(figdir/f'{it:04d}.png', dpi=300)
            plt.close()

    sim.run(2001, callbacks=[
            movingwindow,
            laser, 
            SaveFieldsToHDF5('lwfa/fields', 100, ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'rho']),
            SaveSpeciesDensityToHDF5(ele, 'lwfa/', 100),
            plot_results, # plot after saving hdf5
        ]
    )
