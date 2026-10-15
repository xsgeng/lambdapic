import numpy as np

from lambdapic import (
    Electron,
    ExtractSpeciesDensity,
    ParticleInjector,
    PlotFields,
    Positron,
    SaveFieldsToHDF5,
    SaveSpeciesDensityToHDF5,
    Simulation,
    c,
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

nx = 384
ny = 128
dx = l0 / 20
dy = l0 / 20

Lx = nx * dx
Ly = ny * dy

n0 = 0.5 * nc
u0 = 1.0
ppc = 10

# Gaussian time envelope of the injected streams: each stream is a finite
# pulse of length ~ v * tau rather than a continuous flow
tp = 35e-15
tau = 25e-15

omega_pe = np.sqrt(n0 * e**2 / (epsilon_0 * m_e))

sim = Simulation(
    nx=nx,
    ny=ny,
    dx=dx,
    dy=dy,
    npatch_x=8,
    npatch_y=8,
    boundary_conditions={
        'xmin': 'pml',
        'xmax': 'pml',
        'ymin': 'periodic',
        'ymax': 'periodic',
    },
    sim_time=120e-15,
    log_file='counter-streaming.log',
)

# the species start empty: everything inside comes from the injectors
ele = Electron(density=None)
pos = Positron(density=None)

sim.add_species([ele, pos])


def density(y, t):
    """Stream density, uniform across y, with a Gaussian time envelope."""
    return n0 * np.exp(-((t - tp) / tau)**2)


# two neutral co-moving e-/e+ streams entering from opposite faces: each
# stream is charge- and current-free on its own, so the counter-streaming
# (two-stream / filamentation) instability is the only instability seeded.
# Profiles take the transverse coordinate (y for x-sides) plus optional time.
stream_left_ele = ParticleInjector(ele, density=density, side='xmin', ux=u0, ppc=ppc)
stream_left_pos = ParticleInjector(pos, density=density, side='xmin', ux=u0, ppc=ppc)
stream_right_ele = ParticleInjector(ele, density=density, side='xmax', ux=-u0, ppc=ppc)
stream_right_pos = ParticleInjector(pos, density=density, side='xmax', ux=-u0, ppc=ppc)

if __name__ == "__main__":
    sim.run(
        callbacks=[
            stream_left_ele,
            stream_left_pos,
            stream_right_ele,
            stream_right_pos,
            n_ele := ExtractSpeciesDensity(sim, ele, 100),
            PlotFields(
                [
                    # electron density: the two streams and the two-stream
                    # bunching that grows after they start to interpenetrate
                    dict(field=n_ele.density, scale=1/nc, cmap='Grays', vmin=0, vmax=1.2),
                    # out-of-plane magnetic field B_z: the Weibel filaments.
                    # Counter-streaming along x magnetizes through B_z in 2D
                    # (B_y stays zero by symmetry).
                    dict(field='bz', scale=e/(m_e*omega_pe), cmap='bwr_alpha', vmin=-1.5, vmax=1.5),
                ],
                prefix='counter-streaming/fields', interval=100,
            ),
            SaveFieldsToHDF5('counter-streaming/fields', 100, ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'rho']),
            SaveSpeciesDensityToHDF5(ele, 'counter-streaming/ele', 100),
            SaveSpeciesDensityToHDF5(pos, 'counter-streaming/pos', 100),
        ]
    )
