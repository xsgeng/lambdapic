"""
Hybrid fluid-PIC simulation: relativistic electron beam transport
in a high-density plasma background.

The background electrons are treated as a cold fluid (no macro-particles)
evolved by the relativistic momentum equation (Boris push) and the
continuity equation (upwind scheme with sub-cycling). The beam electrons
and background ions are fully kinetic PIC species. The fluid current is
injected via a callback at the current_deposition stage.

The fluid pusher and current injector are point operations, so they read
E/B and write J directly from per-patch field arrays via typed.List
references, avoiding gather/scatter copies. The continuity equation uses
an upwind stencil and stays on a single global grid, avoiding inter-patch
guard-cell synchronization.

Output:
    data/fields/       -- E, B, J fields (HDF5)
    data/fluid/        -- fluid momentum, velocity, and density (HDF5)
    data/beam_density/ -- beam electron density (HDF5)
"""

from pathlib import Path

import h5py
import numpy as np
from numba import njit, prange, typed
from scipy.constants import c, e, epsilon_0, m_e, pi

from lambdapic import (
    Electron,
    Proton,
    Simulation,
    SaveFieldsToHDF5,
    SaveSpeciesDensityToHDF5,
    SaveParticlesToHDF5
)
from lambdapic.callback.callback import Callback
from lambdapic.callback.utils import SetMomentum


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
um = 1e-6
l0 = 0.8 * um
omega0 = 2 * pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2

n_bg = 100 * nc       # background plasma density
n_beam = 0.5 * nc     # beam density
ux_beam = 50         # beam normalized momentum (gamma ~ 100)

# Domain
Lx = 20.0 * um
Ly = 10.0 * um
dx = l0 / 100
nx = int(round(Lx / dx))
ny = int(round(Ly / dx))

# Plasma region
plasma_xmin = 1.0 * um
plasma_xmax = Lx - 1.0 * um

# Beam region
beam_xmin = 1.0 * um
beam_xmax = 5.0 * um
beam_half_width = 0.2 * um

# Time
sim_time = (Lx-beam_xmax) / c


# ---------------------------------------------------------------------------
# Numba kernels
#
# The fluid pusher and current injector are point operations (no spatial
# stencil), so they read E/B and write J directly from per-patch field
# arrays via typed.List references -- no gather/scatter copies. The
# continuity equation uses an upwind stencil and stays on the global grid.
# ---------------------------------------------------------------------------

@njit(cache=True, inline='always')
def boris_fluid_2d(ux, uy, uz, ex, ey, ez, bx, by, bz, efactor, bfactor):
    """Relativistic Boris push for one cell (u = gamma*v/c, dimensionless).

    Same scheme as the PIC particle pusher, guaranteeing |v| < c.
    Returns (ux_new, uy_new, uz_new).
    """
    # Half acceleration by E
    ux_m = ux + efactor * ex
    uy_m = uy + efactor * ey
    uz_m = uz + efactor * ez

    # Rotation by B (t = bfactor * B / gamma)
    gamma_m = np.sqrt(1.0 + ux_m * ux_m + uy_m * uy_m + uz_m * uz_m)
    inv_gamma_m = 1.0 / gamma_m
    tx = bfactor * bx * inv_gamma_m
    ty = bfactor * by * inv_gamma_m
    tz = bfactor * bz * inv_gamma_m
    t2 = tx * tx + ty * ty + tz * tz
    sx = 2.0 * tx / (1.0 + t2)
    sy = 2.0 * ty / (1.0 + t2)
    sz = 2.0 * tz / (1.0 + t2)

    # u' = u_m + u_m x t
    ux_p = ux_m + (uy_m * tz - uz_m * ty)
    uy_p = uy_m + (uz_m * tx - ux_m * tz)
    uz_p = uz_m + (ux_m * ty - uy_m * tx)

    # u+ = u_m + u' x s, then half acceleration by E
    ux_new = ux_m + (uy_p * sz - uz_p * sy) + efactor * ex
    uy_new = uy_m + (uz_p * sx - ux_p * sz) + efactor * ey
    uz_new = uz_m + (ux_p * sy - uy_p * sx) + efactor * ez

    return ux_new, uy_new, uz_new


@njit(parallel=True, cache=True)
def push_fluid_velocity_patches(
    uex, uey, uez, vex, vey, vez, ne,
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    ix0_arr, iy0_arr,
    npatches, nx_pp, ny_pp,
    dt
):
    """Push fluid momentum u and derive velocity v, reading E/B per patch.

    Fluid arrays (uex, vex, ne, ...) are global (sim.nx, sim.ny).
    Field lists are typed.List of per-patch arrays including guard cells;
    only the interior [0:nx_pp, 0:ny_pp] is read. The u -> v conversion
    is fused with the Boris push to save a grid sweep.
    """
    qm = -e / m_e
    efactor = qm * dt / (2.0 * c)
    bfactor = qm * dt / 2.0

    for _ipatch in prange(npatches):
        ipatch = np.int64(_ipatch)
        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]
        ix0 = ix0_arr[ipatch]
        iy0 = iy0_arr[ipatch]

        for i in range(nx_pp):
            for j in range(ny_pp):
                ig = ix0 + i
                jg = iy0 + j
                n = ne[ig, jg]
                if n == 0.0:
                    continue
                ux_new, uy_new, uz_new = boris_fluid_2d(
                    uex[ig, jg], uey[ig, jg], uez[ig, jg],
                    ex[i, j], ey[i, j], ez[i, j],
                    bx[i, j], by[i, j], bz[i, j],
                    efactor, bfactor,
                )
                uex[ig, jg] = ux_new
                uey[ig, jg] = uy_new
                uez[ig, jg] = uz_new
                gamma = np.sqrt(1.0 + ux_new * ux_new + uy_new * uy_new + uz_new * uz_new)
                vex[ig, jg] = ux_new * c / gamma
                vey[ig, jg] = uy_new * c / gamma
                vez[ig, jg] = uz_new * c / gamma


@njit(parallel=True, cache=True)
def inject_fluid_current_patches(
    jx_list, jy_list, jz_list,
    vex, vey, vez, ne,
    ix0_arr, iy0_arr,
    npatches, nx_pp, ny_pp
):
    """Inject fluid current J = -e * n * v directly into per-patch j arrays.

    Reads global ne/v, accumulates into patch interior current cells.
    """
    for _ipatch in prange(npatches):
        ipatch = np.int64(_ipatch)
        jx = jx_list[ipatch]
        jy = jy_list[ipatch]
        jz = jz_list[ipatch]
        ix0 = ix0_arr[ipatch]
        iy0 = iy0_arr[ipatch]

        for i in range(nx_pp):
            for j in range(ny_pp):
                ig = ix0 + i
                jg = iy0 + j
                n = ne[ig, jg]
                if n == 0.0:
                    continue
                jx[i, j] += -e * n * vex[ig, jg]
                jy[i, j] += -e * n * vey[ig, jg]
                jz[i, j] += -e * n * vez[ig, jg]


@njit(parallel=True)
def update_fluid_density_subcycle(ne, vex, vey, dt, dx, dy):
    """Advance fluid density by dt via continuity equation with upwind scheme.

    dn/dt + d(n*vx)/dx + d(n*vy)/dy = 0

    First-order upwind with zero-gradient (outflow) boundaries.
    The fluid step dt is sub-cycled so that the local CFL number
    v*dt_sub/dx < 0.5, ensuring stability. A single scratch array
    is reused across sub-steps to avoid allocation overhead.
    """
    nx, ny = ne.shape
    vmax = 0.0
    for i in range(nx):
        for j in range(ny):
            v = abs(vex[i, j])
            if v > vmax:
                vmax = v
            v = abs(vey[i, j])
            if v > vmax:
                vmax = v

    if vmax == 0.0:
        return

    n_sub = int(vmax * dt / min(dx, dy) / 0.5) + 1
    if n_sub > 1000:
        n_sub = 1000
    dt_sub = dt / n_sub

    ne_new = np.empty_like(ne)
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    for _ in range(n_sub):
        for i in prange(nx):
            for j in range(ny):
                n = ne[i, j]
                vx = vex[i, j]
                vy = vey[i, j]

                # Upwind flux in x: F = n * vx
                if i == 0:
                    n_up = n
                else:
                    n_up = ne[i - 1, j] if vx >= 0.0 else n
                F_im = n_up * vx

                if i == nx - 1:
                    n_dn = n
                else:
                    n_dn = n if vx >= 0.0 else ne[i + 1, j]
                F_ip = n_dn * vx

                # Upwind flux in y: G = n * vy
                if j == 0:
                    n_up = n
                else:
                    n_up = ne[i, j - 1] if vy >= 0.0 else n
                G_jm = n_up * vy

                if j == ny - 1:
                    n_dn = n
                else:
                    n_dn = n if vy >= 0.0 else ne[i, j + 1]
                G_jp = n_dn * vy

                dndx = (F_ip - F_im) * inv_dx
                dndy = (G_jp - G_jm) * inv_dy
                n_new = n - dt_sub * (dndx + dndy)
                if n_new < 0.0:
                    n_new = 0.0
                ne_new[i, j] = n_new
        ne[:, :] = ne_new


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
class FluidBackground(Callback):
    """Cold-fluid background electrons coupled to PIC via current injection.

    Fluid quantities (uex, vex, ne, ...) live on a single global grid,
    avoiding inter-patch guard-cell synchronization for the continuity
    equation. The Boris pusher reads E/B directly from per-patch field
    arrays via typed.List references, and the current injector writes J
    directly into per-patch current arrays, eliminating gather/scatter
    copies.

    On the first species (ispec == 0) the fluid momentum is pushed by a
    full dt and the density is advanced by the continuity equation. On
    the last species the fluid current J = -e * n_e * v_e is injected
    into the per-patch grid currents.
    """

    stage = "current_deposition"

    def __init__(self):
        self.interval = 1
        self._initialized = False

    def _init_fluid(self, sim):
        self.uex = np.zeros((sim.nx, sim.ny))
        self.uey = np.zeros((sim.nx, sim.ny))
        self.uez = np.zeros((sim.nx, sim.ny))
        self.vex = np.zeros((sim.nx, sim.ny))
        self.vey = np.zeros((sim.nx, sim.ny))
        self.vez = np.zeros((sim.nx, sim.ny))
        self.ne = np.zeros((sim.nx, sim.ny))

        # Fill density from x-axis
        x = np.arange(sim.nx) * sim.dx
        in_plasma = (x >= plasma_xmin) & (x <= plasma_xmax)
        self.ne[in_plasma, :] = n_bg

        # Cache per-patch field references and offsets for zero-copy access
        self.ex_list = typed.List([p.fields.ex for p in sim.patches])
        self.ey_list = typed.List([p.fields.ey for p in sim.patches])
        self.ez_list = typed.List([p.fields.ez for p in sim.patches])
        self.bx_list = typed.List([p.fields.bx for p in sim.patches])
        self.by_list = typed.List([p.fields.by for p in sim.patches])
        self.bz_list = typed.List([p.fields.bz for p in sim.patches])
        self.jx_list = typed.List([p.fields.jx for p in sim.patches])
        self.jy_list = typed.List([p.fields.jy for p in sim.patches])
        self.jz_list = typed.List([p.fields.jz for p in sim.patches])
        self.ix0_arr = np.array(
            [p.ipatch_x * sim.nx_per_patch for p in sim.patches], dtype=np.int64
        )
        self.iy0_arr = np.array(
            [p.ipatch_y * sim.ny_per_patch for p in sim.patches], dtype=np.int64
        )
        self.npatches = sim.patches.npatches
        self.nx_pp = sim.nx_per_patch
        self.ny_pp = sim.ny_per_patch

        self._initialized = True

    def _call(self, sim):
        if not self._initialized:
            self._init_fluid(sim)

        n_species = len(sim.species)

        if sim.ispec == 0:
            # Push fluid momentum (relativistic Boris) and derive velocity,
            # reading E/B directly from per-patch arrays
            push_fluid_velocity_patches(
                self.uex, self.uey, self.uez,
                self.vex, self.vey, self.vez, self.ne,
                self.ex_list, self.ey_list, self.ez_list,
                self.bx_list, self.by_list, self.bz_list,
                self.ix0_arr, self.iy0_arr,
                self.npatches, self.nx_pp, self.ny_pp,
                sim.dt,
            )
            # Advance density via continuity equation (sub-cycled upwind)
            update_fluid_density_subcycle(
                self.ne, self.vex, self.vey, sim.dt, sim.dx, sim.dy
            )

        elif sim.ispec == n_species - 1:
            # Inject fluid current directly into per-patch j arrays
            inject_fluid_current_patches(
                self.jx_list, self.jy_list, self.jz_list,
                self.vex, self.vey, self.vez, self.ne,
                self.ix0_arr, self.iy0_arr,
                self.npatches, self.nx_pp, self.ny_pp,
            )


class SaveFluidFields(Callback):
    """Save fluid velocity and background density to HDF5."""

    stage = "end"

    def __init__(self, fluid_cb, prefix, interval=200):
        self.fluid_cb = fluid_cb
        self.interval = interval
        self.prefix = Path(prefix)
        self.prefix.mkdir(parents=True, exist_ok=True)

    def _call(self, sim):
        filename = self.prefix / f"{sim.itime:06d}.h5"
        with h5py.File(filename, "w") as h5f:
            h5f.create_dataset("uex", data=self.fluid_cb.uex)
            h5f.create_dataset("uey", data=self.fluid_cb.uey)
            h5f.create_dataset("uez", data=self.fluid_cb.uez)
            h5f.create_dataset("vex", data=self.fluid_cb.vex)
            h5f.create_dataset("vey", data=self.fluid_cb.vey)
            h5f.create_dataset("vez", data=self.fluid_cb.vez)
            h5f.create_dataset("ne", data=self.fluid_cb.ne)


# ---------------------------------------------------------------------------
# Density profiles
# ---------------------------------------------------------------------------
def density_bg(n0):
    def _density(x, y):
        if x >= plasma_xmin and x <= plasma_xmax:
            return n0
        return 0.0
    return _density


def density_beam(n0):
    def _density(x, y):
        if x >= beam_xmin and x <= beam_xmax and abs(y - Ly / 2) < beam_half_width:
            return n0
        return 0.0
    return _density


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    outdir = Path(Path(__file__).stem)
    outdir.mkdir(exist_ok=True)

    sim = Simulation(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dx,
        sim_time=sim_time,
        random_seed=42,
    )

    beam_ele = Electron(density=density_beam(n_beam), ppc=2)
    bg_proton = Proton(density=density_bg(n_bg), ppc=1)
    sim.add_species([beam_ele, bg_proton])

    fluid_cb = FluidBackground()

    sim.run(callbacks=[
        SetMomentum(beam_ele, [ux_beam, 0, 0]),
        fluid_cb,
        SaveFluidFields(fluid_cb, str(outdir / "fluid"), interval=200),
        SaveFieldsToHDF5(
            str(outdir / "fields"), 200, ["ex", "ey", "bz", "jx", "jy"]
        ),
        SaveSpeciesDensityToHDF5(beam_ele, prefix=str(outdir / "beam_density"), interval=200),
        SaveParticlesToHDF5(beam_ele, prefix=str(outdir / "beam_particles"), interval=200)
    ])
