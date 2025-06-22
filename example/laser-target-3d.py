from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi

from lambdapic import Electron, Proton, Species, Simulation3D, callback
from lambdapic.callback.laser import  GaussianLaser3D

from lambdapic.callback.hdf5 import SaveFieldsToHDF5, SaveSpeciesDensityToHDF5

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
)
if __name__ == "__main__":
    sim = Simulation3D(
        nx=nx, ny=ny, nz=nz,
        dx=dx, dy=dy, dz=dz,
        npatch_x=16,
        npatch_y=8,
        npatch_z=8,
        cpml_thickness=6,
    )

    ele = Electron(density=density(1*nc), ppc=10)
    proton = Proton(density=density(1*nc), ppc=10)

    sim.add_species([ele, proton])

    sim.run(
        1001, 
        callbacks=[
            laser,
            SaveFieldsToHDF5('laser-target-3d', 100, ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'rho']),
            SaveSpeciesDensityToHDF5(ele, 'laser-target-3d/density', 100),
        ]
    )
