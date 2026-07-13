from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi

from lambdapic import Electron, Proton, Simulation, Species, callback, SaveFieldsToHDF5
from lambdapic.callback.utils import ExtractSpeciesDensity, SetMomentum
from lambdapic.callback.callback import Callback


import h5py


um = 1e-6
l0 = 0.8 * um
t0 = l0 / c
omega0 = 2 * pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2

nx = 2048
ny = 2048
dx = l0 / 50
dy = l0 / 50

Lx = nx * dx
Ly = ny * dy

class ExtractSpeciesCurrent(Callback):
    stage = "current_deposition"
    def __init__(self, sim: Simulation, species: Species, which: str, every):
        self.sim = sim
        self.species = species
        self.every = every
        self.ispec_target = sim.species.index(species)
        self.j = np.zeros((sim.nx, sim.ny))
        
        self.nx_per_patch = sim.nx_per_patch
        self.ny_per_patch = sim.ny_per_patch
        self.n_guard = sim.n_guard

        self.which = which

    def _get_patch_slice(self, patch):
        return np.s_[
            patch.ipatch_x*self.nx_per_patch:(patch.ipatch_x+1)*self.nx_per_patch,
            patch.ipatch_y*self.ny_per_patch:(patch.ipatch_y+1)*self.ny_per_patch
        ]

    def __call__(self, sim: Simulation):
        if callable(self.every):
            if not self.every(sim):
                return
        elif sim.itime % self.every != 0:
            return

        ispec = sim.ispec
        if self.ispec_target == 0:
            if ispec == 0:
                for p in sim.patches:
                    s = self._get_patch_slice(p)
                    self.j[s] = getattr(p.fields, self.which)[:-2*self.n_guard, :-2*self.n_guard]
        else:
            if ispec == self.ispec_target - 1:
                for p in sim.patches:
                    s = self._get_patch_slice(p)
                    # store previous rho
                    self.j[s] = getattr(p.fields, self.which)[:-2*self.n_guard, :-2*self.n_guard]
            if ispec == self.ispec_target:
                for p in sim.patches:
                    s = self._get_patch_slice(p)
                    # subtract previous rho
                    self.j[s] = getattr(p.fields, self.which)[:-2*self.n_guard, :-2*self.n_guard] - self.j[s]

def density(n0):
    def _density(x, y):
        ne = 0.0
        if x > Lx/2 and x < Lx/2 + 1*um:
            ne = n0
        return ne
    return _density

def dens_bunch(n0):
    def _density(x, y):
        ne = 0.0
        if x > Lx/2 - 2*um and x < Lx/2 - 1*um and abs(y - Ly/2) < 0.5*um:
            ne = n0
        return ne
    return _density

@njit(parallel=True)
def step_maxwell_2d_first(ex, ey, bz, jx, jy, dx, dy, dt):
    """
    Advance 2D TE Maxwell equations by one full dt with zero boundary.
    B is stored at half time steps (leapfrog).
    """
    nx, ny = ex.shape
    bfactor = dt * c**2
    jfactor = dt / epsilon_0

    # E^n -> E^(n+1) using B^(n+1/2)
    for i in prange(nx):
        for j in range(ny):
            dbz_dy = (bz[i, j] - bz[i, j-1]) / dy if j > 0 else 0.0
            dbz_dx = (bz[i, j] - bz[i-1, j]) / dx if i > 0 else 0.0
            ex[i, j] += bfactor * dbz_dy - jfactor * jx[i, j]
            ey[i, j] += bfactor * -dbz_dx - jfactor * jy[i, j]

    # B^(n-1/2) -> B^(n+1/2) using E^n
    for i in prange(nx):
        for j in range(ny):
            dey_dx = (ey[i+1, j] - ey[i, j]) / dx if i < nx - 1 else 0.0
            dex_dy = (ex[i, j+1] - ex[i, j]) / dy if j < ny - 1 else 0.0
            bz[i, j] -= dt * (dey_dx - dex_dy)

@njit(parallel=True)
def step_maxwell_2d_second(ex, ey, bz, jx, jy, dx, dy, dt):
    """
    Advance 2D TE Maxwell equations by one full dt with zero boundary.
    B is stored at half time steps (leapfrog).
    """
    nx, ny = ex.shape
    bfactor = dt * c**2
    jfactor = dt / epsilon_0

    # B^(n-1/2) -> B^(n+1/2) using E^n
    for i in prange(nx):
        for j in range(ny):
            dey_dx = (ey[i+1, j] - ey[i, j]) / dx if i < nx - 1 else 0.0
            dex_dy = (ex[i, j+1] - ex[i, j]) / dy if j < ny - 1 else 0.0
            bz[i, j] -= dt * (dey_dx - dex_dy)

    # E^n -> E^(n+1) using B^(n+1/2)
    for i in prange(nx):
        for j in range(ny):
            dbz_dy = (bz[i, j] - bz[i, j-1]) / dy if j > 0 else 0.0
            dbz_dx = (bz[i, j] - bz[i-1, j]) / dx if i > 0 else 0.0
            ex[i, j] += bfactor * dbz_dy - jfactor * jx[i, j]
            ey[i, j] += bfactor * -dbz_dx - jfactor * jy[i, j]

sim = Simulation(
    nx=nx,
    ny=ny,
    dx=dx,
    dy=dy,
)

ele = Electron(density=density(10*nc), ppc=10)
proton = Proton(density=density(10*nc), ppc=10)

ele_bunch = Electron(density=dens_bunch(0.1*nc), ppc=10)
proton_bunch = Proton(density=dens_bunch(0.1*nc), ppc=10)

sim.add_species([ele, proton, ele_bunch, proton_bunch])

@callback('init')
def copy_positions(sim: Simulation):
    for p in sim.patches:
        p.particles[proton.ispec].x[:] = p.particles[ele.ispec].x[:]
        p.particles[proton.ispec].y[:] = p.particles[ele.ispec].y[:]
        p.particles[proton_bunch.ispec].x[:] = p.particles[ele_bunch.ispec].x[:]
        p.particles[proton_bunch.ispec].y[:] = p.particles[ele_bunch.ispec].y[:]

set_momentum_bunch = SetMomentum(ele_bunch, [100, 0, 0])


if __name__ == "__main__":
    outdir = Path(Path(__file__).stem)
    outdir.mkdir(exist_ok=True)

    store_jx_ele = ExtractSpeciesCurrent(sim, ele, 'jx', 1)
    store_jx_proton = ExtractSpeciesCurrent(sim, proton, 'jx', 1)
    store_jx_ele_bunch = ExtractSpeciesCurrent(sim, ele_bunch, 'jx', 1)
    store_jx_proton_bunch = ExtractSpeciesCurrent(sim, proton_bunch, 'jx', 1)

    store_jy_ele = ExtractSpeciesCurrent(sim, ele, 'jy', 1)
    store_jy_proton = ExtractSpeciesCurrent(sim, proton, 'jy', 1)
    store_jy_ele_bunch = ExtractSpeciesCurrent(sim, ele_bunch, 'jy', 1)
    store_jy_proton_bunch = ExtractSpeciesCurrent(sim, proton_bunch, 'jy', 1)

    ex_target = np.zeros((nx, ny))
    ey_target = np.zeros((nx, ny))
    bz_target = np.zeros((nx, ny))

    ex_bunch = np.zeros((nx, ny))
    ey_bunch = np.zeros((nx, ny))
    bz_bunch = np.zeros((nx, ny))

    h5py.File('ex-target.h5', 'w').close()
    h5py.File('ex-bunch.h5', 'w').close()

    @callback(stage='maxwell_1')
    def solve_maxwell_first(sim: Simulation):
        it = sim.itime
        dt = sim.dt/2

        jx_target = store_jx_ele.j + store_jx_proton.j
        jy_target = store_jy_ele.j + store_jy_proton.j

        step_maxwell_2d_first(ex_target, ey_target, bz_target, jx_target, jy_target, dx, dy, dt)

        jx_bunch = store_jx_ele_bunch.j + store_jx_proton_bunch.j
        jy_bunch = store_jy_ele_bunch.j + store_jy_proton_bunch.j

        step_maxwell_2d_first(ex_bunch, ey_bunch, bz_bunch, jx_bunch, jy_bunch, dx, dy, dt)

    @callback(stage='maxwell_2')
    def solve_maxwell_second(sim: Simulation):
        it = sim.itime
        dt = sim.dt/2

        jx_target = store_jx_ele.j + store_jx_proton.j
        jy_target = store_jy_ele.j + store_jy_proton.j

        step_maxwell_2d_second(ex_target, ey_target, bz_target, jx_target, jy_target, dx, dy, dt)

        jx_bunch = store_jx_ele_bunch.j + store_jx_proton_bunch.j
        jy_bunch = store_jy_ele_bunch.j + store_jy_proton_bunch.j

        step_maxwell_2d_second(ex_bunch, ey_bunch, bz_bunch, jx_bunch, jy_bunch, dx, dy, dt)

        if it % 100 == 0:
            with h5py.File(outdir/'ex-target.h5', 'a') as f:
                g = f.create_group(f'{it:04d}')
                g.create_dataset('ex', data=ex_target)
            with h5py.File(outdir/'ex-bunch.h5', 'a') as f:
                g = f.create_group(f'{it:04d}')
                g.create_dataset('ex', data=ex_bunch)


    sim.run(501, callbacks=[
            copy_positions,
            store_jx_ele,
            store_jx_proton,
            store_jy_ele,
            store_jy_proton,
            store_jx_ele_bunch,
            store_jx_proton_bunch,
            store_jy_ele_bunch,
            store_jy_proton_bunch,
            solve_maxwell_first,
            solve_maxwell_second,
            set_momentum_bunch,
            SaveFieldsToHDF5(outdir/"fields", 100, ["ex"])
        ]
    )
