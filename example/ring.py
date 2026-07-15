"""Ring (annulus) domain simulation example.

This example demonstrates the `_MaskSimulation` class which creates
a simulation domain from a boolean mask function. Only patches whose
centers evaluate to True are included; every open face automatically
gets a PML absorbing boundary.

The target geometry is a ring (annulus) with inner radius `r_inner`
and outer radius `r_outer`. Both the inner hole edge and the outer
domain edge absorb outgoing waves via PML.
"""

import numpy as np

from lambdapic import (
    Electron,
    ExtractSpeciesDensity,
    SetTemperature,
    PlotFields,
    Proton,
    SaveFieldsToHDF5,
    SaveSpeciesDensityToHDF5,
    c,
    e,
    epsilon_0,
    m_e,
    pi,
)
from lambdapic.simulation._mask_simulation import _MaskSimulation


# --- physical constants ---
um = 1e-6
l0 = 0.8 * um
omega0 = 2 * pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2

# --- grid ---
nx = ny = 512
dx = dy = l0 / 20
Lx = nx * dx
Ly = ny * dy

# --- ring geometry ---
r_inner = 0.2 * Lx
r_outer = 0.45 * Lx
cx = Lx / 2
cy = Ly / 2


def ring_mask(r_inner: float, r_outer: float, cx: float, cy: float):
    """Return a mask function for an annular (ring) domain."""
    def _mask(x: float, y: float) -> bool:
        r = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        return r_inner <= r <= r_outer
    return _mask


# density is non-zero only inside the ring, away from the PML edges
def density(n0):
    def _density(x, y):
        r = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        if (r_inner+r_outer)/2 - 0.5*um < r < (r_inner+r_outer)/2 + 0.5*um:
            return n0
        return 0.0
    return _density

# --- mask-driven simulation ---
sim = _MaskSimulation(
    nx=nx,
    ny=ny,
    dx=dx,
    dy=dy,
    npatch_x=32,
    npatch_y=32,
    dt_cfl=0.99,
    nsteps=2001,
    log_file="ring.log",
    mask=ring_mask(r_inner, r_outer, cx, cy),
)

ne = 10 * nc

ele = Electron(density=density(ne), ppc=10)
proton = Proton(density=density(ne), ppc=2)

sim.add_species([ele, proton])

if __name__ == "__main__":
    sim.run(
        callbacks=[
            SetTemperature(ele, 1e5),
            n_ele := ExtractSpeciesDensity(sim, ele, 500),
            PlotFields(
                [
                    dict(
                        field=n_ele.density,
                        scale=1 / nc,
                        cmap="Grays",
                        vmin=0,
                        vmax=ne / nc * 2,
                    ),
                    dict(
                        field="ey",
                        scale=e / (m_e * c * omega0),
                        cmap="bwr_alpha",
                    ),
                ],
                prefix="ring/ey",
                interval=500,
            ),
            SaveFieldsToHDF5(
                "ring/fields", 500, ["ex", "ey", "ez", "bx", "by", "bz", "rho"]
            ),
            SaveSpeciesDensityToHDF5(ele, "ring/density", 500),
            SaveSpeciesDensityToHDF5(proton, "ring/density", 500),
        ]
    )
