from lambdapic import (
    Electron,
    ExtractSpeciesDensity,
    GaussianLaser2D,
    PlotFields,
    Proton,
    SaveFieldsToHDF5,
    SaveSpeciesDensityToHDF5,
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
    x0=10e-6
)

sim = Simulation(
    nx=nx,
    ny=ny,
    dx=dx,
    dy=dy,
    nsteps=2000,
    log_file='laser-target.log',
)

ele = Electron(density=density(10*nc), ppc=10)
proton = Proton(density=density(10*nc/8*2), ppc=10)
carbon = Species(name="C", charge=6, mass=12*1800, density=density(10*nc/8), ppc=10)

sim.add_species([ele, carbon, proton])
    
if __name__ == "__main__":
    sim.run(2001, callbacks=[
            laser, 
            n_ele := ExtractSpeciesDensity(sim, ele, 500),
            PlotFields(
                [
                    dict(field=n_ele.density, scale=1/nc, cmap='Grays', vmin=0, vmax=20), 
                    dict(field='ey',  scale=e/(m_e*c*omega0), cmap='bwr_alpha', vmin=-laser.a0, vmax=laser.a0)
                ],
                prefix='laser-target', interval=10e-15,
            ),
            SaveFieldsToHDF5('laser-target/fields', 500, ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'rho']),
            SaveSpeciesDensityToHDF5(carbon, 'laser-target/density', 500),
            SaveSpeciesDensityToHDF5(ele, 'laser-target/density', 500),
        ]
    )
