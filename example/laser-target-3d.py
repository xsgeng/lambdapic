import h5py
import numpy as np

from lambdapic import (
    Electron,
    GaussianLaser3D,
    PlotFields,
    Proton,
    SaveFieldsToHDF5,
    SaveSpeciesDensityToHDF5,
    Simulation3D,
    c,
    callback,
    e,
    epsilon_0,
    m_e,
    pi,
)

um = 1e-6
l0 = 0.8 * um
t0 = l0 / c
omega0 = 2 * pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2

nx = 512
ny = 256
nz = 256
dx = l0 / 20
dy = l0 / 10
dz = l0 / 10

Lx = nx * dx
Ly = ny * dy
Lz = nz * dz

def density(n0):
    def _density(x, y, z):
        if x > 1*um:
            return n0
        return 0.0
    return _density

laser = GaussianLaser3D(
    a0=10,
    w0=2e-6,
    l0=0.8e-6,
    ctau=5e-6,
    focus_position=Lx/2,
    x0=10e-6,
)
sim = Simulation3D(
    nx=nx, ny=ny, nz=nz,
    dx=dx, dy=dy, dz=dz,
    nsteps=1001,
    log_file='laser-target-3d.log',
)

ele = Electron(density=density(1*nc), ppc=2)
proton = Proton(density=density(1*nc), ppc=2)

sim.add_species([ele, proton])

ne_data = np.zeros((nx, ny))
ey_data = np.zeros((nx, ny))

@callback(interval=100)
def read_density(sim: Simulation3D):
    if sim.mpi.rank > 0:
        return
    
    with h5py.File(f'laser-target-3d/{sim.itime:06d}.h5', 'r', locking=False) as f:
        ey_data[:, :] = f['ey'][..., nz//2]

    with h5py.File(f'laser-target-3d/{ele.name}_{sim.itime:06d}.h5', 'r', locking=False) as f:
        ne_data[:, :] = f['density'][..., nz//2]

diag_interval = 100
if __name__ == "__main__":
    sim.run(
        1001, 
        callbacks=[
            laser,
            SaveFieldsToHDF5('laser-target-3d', diag_interval, ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'rho']),
            SaveSpeciesDensityToHDF5(ele, 'laser-target-3d', diag_interval),
            read_density,
            PlotFields(
                [
                    dict(field=ne_data, scale=1.0/nc, cmap='Grays', vmin=0, vmax=2), 
                    dict(field=ey_data,  scale=e/(m_e*c*omega0), cmap='bwr_alpha', vmin=-laser.a0, vmax=laser.a0)
                ],
                'laser-target-3d', diag_interval
            ),
        ]
    )
