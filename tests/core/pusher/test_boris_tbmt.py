"""Tests for the Boris pusher with TBMT spin dynamics.

Covers:
- the numba fallback kernels (boris + lambdapic.core.pusher.tbmt.tbmt_spin)
- the unified C kernels (unified_boris_tbmt_pusher_cpu_2d/3d)
- species/particle plumbing (polarization, anomalous magnetic moment)
- a full 2D Simulation on both pusher dispatch paths
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.constants import c, e, m_e

from lambdapic.core.fields import Fields2D, Fields3D
from lambdapic.core.particles import ParticlesBase, SpinParticles
from lambdapic.core.pusher.boris import boris
from lambdapic.core.pusher.tbmt import tbmt_spin
from lambdapic.core.species import Electron, Species
from lambdapic.core.pusher.unified.unified_pusher_2d import (
    unified_boris_tbmt_pusher_cpu_2d,
)
from lambdapic.core.pusher.unified.unified_pusher_3d import (
    unified_boris_tbmt_pusher_cpu_3d,
)
from lambdapic import Simulation
from lambdapic.callback.callback import callback


@pytest.fixture(autouse=True)
def _isolate_species_registry(isolated_registry):
    """Keep this module's species out of the global ``_ALL_SPECIES`` registry.

    The TBMT tests register spin species (including error-path species with
    ``pusher='boris+tbmt'`` but no polarization); leaking them into the
    shared registry breaks unrelated species-less simulations on the same
    xdist worker (they auto-register every polluted species at init).
    """

# ---------------------------------------------------------------------------
# Parameters (realistic PIC scales)
# ---------------------------------------------------------------------------
AE_SCHWINGER = 0.00115965218059  # electron a_e = (g-2)/2
DT = 1e-17  # s
Q = -e
M = m_e

NX, NY, NZ = 16, 16, 16
DX = DY = DZ = 1e-8
N_GUARD = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_spin_particles(n: int, **arrays) -> SpinParticles:
    """Construct SpinParticles with given arrays."""
    p = SpinParticles(ipatch=0, rank=0)
    p.initialize(n)
    for name, arr in arrays.items():
        getattr(p, name)[:] = arr
    return p


def analytic_spin_angle(T_axis: float) -> float:
    """Rotation angle of one Boris spin rotation around a signed axis value.

    The Boris map s' = s + s x T, sf = s + s' x S with S = 2T/(1+T^2)
    is an exact rotation by -2*atan(T) around the T axis.
    """
    return -2.0 * np.arctan(T_axis)


# ---------------------------------------------------------------------------
# 1. Numba fallback kernel: analytic spin precession
# ---------------------------------------------------------------------------
class TestTbmtNumbaKernel:
    """lambdapic.core.pusher.tbmt.tbmt_spin physics tests."""

    def _run_kernel(self, part: SpinParticles, Ex, Ey, Ez, Bx, By, Bz,
                    q, m, ae, dt, nsteps) -> None:
        n = part.npart
        e = [np.full(n, v) for v in (Ex, Ey, Ez)]
        b = [np.full(n, v) for v in (Bx, By, Bz)]
        for _ in range(nsteps):
            boris(part.ux, part.uy, part.uz, part.inv_gamma,
                  *e, *b, q, m, n, part.is_dead, dt)
            tbmt_spin(part.sx, part.sy, part.sz,
                      part.ux, part.uy, part.uz,
                      *e, *b, q, m, ae, n, part.is_dead, dt)

    def test_rest_particle_spin_precession(self):
        """Electron at rest in uniform Bz: exact per-step spin rotation.

        With beta = 0 the effective field is B_eff = (a+1) B, so each step
        rotates the spin by theta = -2*atan(q*dt*(a+1)*B/2m) around z.
        The momentum must stay exactly zero.
        """
        B0 = 1000.0  # T
        for ae in (0.0, AE_SCHWINGER, 0.5):
            part = make_spin_particles(1, ux=[0.0], uy=[0.0], uz=[0.0],
                                       sx=[1.0], sy=[0.0], sz=[0.0])
            nsteps = 50

            self._run_kernel(part, 0, 0, 0, 0, 0, B0, Q, M, ae, DT, nsteps)

            theta = analytic_spin_angle(Q * DT * (ae + 1.0) * B0 / (2 * M))
            assert_allclose(part.sx, np.cos(nsteps * theta), rtol=1e-12)
            assert_allclose(part.sy, np.sin(nsteps * theta), rtol=1e-12)
            assert_allclose(part.sz, 0.0, atol=1e-15)
            # momentum unchanged
            assert_allclose(part.ux, 0.0, atol=1e-16)
            assert_allclose(part.uy, 0.0, atol=1e-16)
            assert_allclose(part.uz, 0.0, atol=1e-16)
            assert_allclose(part.inv_gamma, 1.0, rtol=1e-15)

    def test_relativistic_pure_b_precession(self):
        """Relativistic electron in uniform Bz: precession with 1/gamma.

        For beta along x and B along z, beta.B = 0 and gamma is constant,
        so B_eff = (a + 1/gamma) B and the spin rotates by a constant
        angle per step while the momentum gyrates.
        """
        B0 = 1000.0  # T
        u0 = 2.0  # gamma = sqrt(5)
        ae = AE_SCHWINGER
        gamma = np.sqrt(1 + u0 * u0)

        part = make_spin_particles(1, ux=[u0], uy=[0.0], uz=[0.0],
                                   sx=[1.0], sy=[0.0], sz=[0.0])
        nsteps = 40

        self._run_kernel(part, 0, 0, 0, 0, 0, B0, Q, M, ae, DT, nsteps)

        theta = analytic_spin_angle(Q * DT * (ae + 1.0 / gamma) * B0 / (2 * M))
        assert_allclose(part.sx, np.cos(nsteps * theta), rtol=1e-12)
        assert_allclose(part.sy, np.sin(nsteps * theta), rtol=1e-12)
        # energy conserved
        assert_allclose(np.sqrt(part.ux**2 + part.uy**2 + part.uz**2), u0,
                        rtol=1e-12)

    def test_e_cross_beta_term(self):
        """Single step with E only: spin precession from the beta x E/c term.

        For u along x, E along y and B = 0 the mid-step momentum is
        (u0, p, 0) with p = q*dt*Ey/(2mc), gamma = sqrt(1+u0^2+p^2), and
        B_eff = -(a + 1/(gamma+1)) * (beta x E)/c points along z.
        """
        u0 = 1.5
        Ey = 5e11  # V/m
        ae = AE_SCHWINGER

        part = make_spin_particles(1, ux=[u0], uy=[0.0], uz=[0.0],
                                   sx=[1.0], sy=[0.0], sz=[0.0])

        self._run_kernel(part, 0, Ey, 0, 0, 0, 0, Q, M, ae, DT, 1)

        efactor = Q * DT / (2 * M * c)
        p = efactor * Ey
        gamma = np.sqrt(1 + u0 * u0 + p * p)
        Tz = Q * DT / (2 * M) * (-(ae + 1.0 / (gamma + 1.0))
                                 * (u0 / gamma) * Ey / c)
        theta = analytic_spin_angle(Tz)
        assert_allclose(part.sx, np.cos(theta), rtol=1e-12)
        assert_allclose(part.sy, np.sin(theta), rtol=1e-12)
        assert_allclose(part.sz, 0.0, atol=1e-15)
        # momentum: two half kicks along y, no rotation
        assert_allclose(part.ux, u0, rtol=1e-13)
        assert_allclose(part.uy, 2 * p, rtol=1e-12)

    def test_orbit_matches_plain_boris(self):
        """The spin pass must not change the orbit: compare against plain Boris."""
        rng = np.random.default_rng(42)
        n = 64
        ux = rng.uniform(-2, 2, n)
        uy = rng.uniform(-2, 2, n)
        uz = rng.uniform(-2, 2, n)
        Ex = rng.uniform(-1e11, 1e11, n)
        Ey = rng.uniform(-1e11, 1e11, n)
        Ez = rng.uniform(-1e11, 1e11, n)
        Bx = rng.uniform(-10, 10, n)
        By = rng.uniform(-10, 10, n)
        Bz = rng.uniform(-10, 10, n)

        part_tbmt = make_spin_particles(
            n, ux=ux, uy=uy, uz=uz, sx=rng.uniform(-1, 1, n),
            sy=rng.uniform(-1, 1, n), sz=rng.uniform(-1, 1, n))
        part_boris = ParticlesBase(ipatch=0, rank=0)
        part_boris.initialize(n)
        part_boris.ux[:] = ux
        part_boris.uy[:] = uy
        part_boris.uz[:] = uz

        boris(
            part_boris.ux, part_boris.uy, part_boris.uz,
            part_boris.inv_gamma, Ex, Ey, Ez, Bx, By, Bz,
            Q, M, n, part_boris.is_dead, DT)
        boris(
            part_tbmt.ux, part_tbmt.uy, part_tbmt.uz,
            part_tbmt.inv_gamma, Ex, Ey, Ez, Bx, By, Bz,
            Q, M, n, part_tbmt.is_dead, DT)
        tbmt_spin(
            part_tbmt.sx, part_tbmt.sy, part_tbmt.sz,
            part_tbmt.ux, part_tbmt.uy, part_tbmt.uz,
            Ex, Ey, Ez, Bx, By, Bz, Q, M, AE_SCHWINGER, n,
            part_tbmt.is_dead, DT)

        assert_allclose(part_tbmt.ux, part_boris.ux, rtol=1e-13)
        assert_allclose(part_tbmt.uy, part_boris.uy, rtol=1e-13)
        assert_allclose(part_tbmt.uz, part_boris.uz, rtol=1e-13)
        assert_allclose(part_tbmt.inv_gamma, part_boris.inv_gamma, rtol=1e-13)

    def test_spin_norm_conserved(self):
        """|s| is conserved by the Boris rotation to machine precision."""
        rng = np.random.default_rng(7)
        n = 8
        s0 = rng.normal(size=(n, 3))
        s0 /= np.linalg.norm(s0, axis=1)[:, None]

        part = make_spin_particles(
            n, ux=rng.uniform(-3, 3, n), uy=rng.uniform(-3, 3, n),
            uz=rng.uniform(-3, 3, n),
            sx=s0[:, 0], sy=s0[:, 1], sz=s0[:, 2])

        for _ in range(100):
            self._run_kernel(
                part,
                float(rng.uniform(-1e12, 1e12)), float(rng.uniform(-1e12, 1e12)),
                float(rng.uniform(-1e12, 1e12)),
                float(rng.uniform(-20, 20)), float(rng.uniform(-20, 20)),
                float(rng.uniform(-20, 20)),
                Q, M, AE_SCHWINGER, DT, 1)

        norm = np.sqrt(part.sx**2 + part.sy**2 + part.sz**2)
        assert_allclose(norm, 1.0, rtol=1e-12)

    def test_dead_particles_skipped(self):
        """Dead particles are not touched."""
        part = make_spin_particles(
            4, ux=[1.0, 0.5, 0.2, 0.1], uy=[0.0] * 4, uz=[0.0] * 4,
            sx=[1.0, 0.6, 0.0, 0.8], sy=[0.0, 0.8, 0.0, -0.6],
            sz=[0.0, 0.0, 0.0, 0.0])
        part.is_dead[:] = [True, False, True, False]

        ux0 = part.ux.copy()
        sx0 = part.sx.copy()

        self._run_kernel(part, 1e10, 0, 0, 0, 0, 5.0, Q, M, AE_SCHWINGER, DT, 1)

        assert_allclose(part.ux[[0, 2]], ux0[[0, 2]])
        assert_allclose(part.sx[[0, 2]], sx0[[0, 2]])
        # alive particles did move
        assert not np.allclose(part.ux[[1, 3]], ux0[[1, 3]])
        assert not np.allclose(part.sx[[1, 3]], sx0[[1, 3]])


# ---------------------------------------------------------------------------
# 2. Unified C kernel vs numba kernel
# ---------------------------------------------------------------------------
def _set_field_pattern(fields) -> None:
    """Fill fields with a nontrivial deterministic pattern."""
    x = fields.xaxis / DX
    y = fields.yaxis / DY
    X, Y = np.meshgrid(x, y, indexing="ij")
    fields.ex[:, :] = np.sin(X) * np.cos(Y) * 1e10 + 2e9
    fields.ey[:, :] = np.cos(X) * np.sin(Y) * 1e10 - 3e9
    fields.ez[:, :] = np.cos(X + Y) * 5e10 + 1e9
    fields.bx[:, :] = np.sin(0.5 * X) * 3.0 + 1.0
    fields.by[:, :] = np.cos(0.7 * Y) * 2.0 - 0.5
    fields.bz[:, :] = np.sin(X * Y * 0.01) * 4.0 + 2.5


def _make_fields2d():
    return Fields2D(nx=NX, ny=NY, dx=DX, dy=DY, x0=0.0, y0=0.0,
                    n_guard=N_GUARD)


def _make_fields3d():
    return Fields3D(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ,
                   x0=0.0, y0=0.0, z0=0.0, n_guard=N_GUARD)


class TestUnifiedTbmtCpu2D:
    """unified_boris_tbmt_pusher_cpu_2d vs the numba kernel."""

    def _run_and_compare(self, part: SpinParticles, fields: Fields2D,
                          is_clean: bool) -> None:
        # snapshot
        snap = {name: getattr(part, name).copy()
                for name in ("x", "y", "ux", "uy", "uz",
                             "sx", "sy", "sz", "inv_gamma")}

        unified_boris_tbmt_pusher_cpu_2d([part], [fields], 1, DT, Q, M,
                                         AE_SCHWINGER)

        # interpolated fields used by the kernel are stored on the particles
        ex_i, ey_i, ez_i = part.ex_part[0], part.ey_part[0], part.ez_part[0]
        bx_i, by_i, bz_i = part.bx_part[0], part.by_part[0], part.bz_part[0]

        # replay with the numba kernels: Boris, then the standalone spin
        ref = SpinParticles(ipatch=0, rank=0)
        ref.initialize(1)
        for name in ("ux", "uy", "uz", "sx", "sy", "sz", "inv_gamma"):
            getattr(ref, name)[:] = snap[name][0]

        # half position push (old momentum)
        xh = snap["x"][0] + c * 0.5 * DT * snap["inv_gamma"][0] * snap["ux"][0]
        yh = snap["y"][0] + c * 0.5 * DT * snap["inv_gamma"][0] * snap["uy"][0]

        boris(
            ref.ux, ref.uy, ref.uz, ref.inv_gamma,
            np.array([ex_i]), np.array([ey_i]), np.array([ez_i]),
            np.array([bx_i]), np.array([by_i]), np.array([bz_i]),
            Q, M, 1, ref.is_dead, DT)
        tbmt_spin(
            ref.sx, ref.sy, ref.sz,
            ref.ux, ref.uy, ref.uz,
            np.array([ex_i]), np.array([ey_i]), np.array([ez_i]),
            np.array([bx_i]), np.array([by_i]), np.array([bz_i]),
            Q, M, AE_SCHWINGER, 1, ref.is_dead, DT)

        # half position push (new momentum)
        x_ref = xh + c * 0.5 * DT * ref.inv_gamma[0] * ref.ux[0]
        y_ref = yh + c * 0.5 * DT * ref.inv_gamma[0] * ref.uy[0]

        assert_allclose(part.ux[0], ref.ux[0], rtol=1e-13)
        assert_allclose(part.uy[0], ref.uy[0], rtol=1e-13)
        assert_allclose(part.uz[0], ref.uz[0], rtol=1e-13)
        assert_allclose(part.sx[0], ref.sx[0], rtol=1e-13)
        assert_allclose(part.sy[0], ref.sy[0], rtol=1e-13)
        assert_allclose(part.sz[0], ref.sz[0], rtol=1e-13)
        assert_allclose(part.inv_gamma[0], ref.inv_gamma[0], rtol=1e-13)
        assert_allclose(part.x[0], x_ref, rtol=1e-12)
        assert_allclose(part.y[0], y_ref, rtol=1e-12)

    def test_clean_strip_matches_numba(self):
        """All particles alive: clean strip path."""
        fields = _make_fields2d()
        _set_field_pattern(fields)
        x_center = (NX / 2) * DX
        y_center = (NY / 2) * DY
        part = make_spin_particles(
            1, x=[x_center], y=[y_center], ux=[0.3], uy=[-0.2], uz=[0.5],
            sx=[0.0], sy=[1.0], sz=[0.0])
        self._run_and_compare(part, fields, is_clean=True)

    def test_dirty_strip_matches_numba(self):
        """A dead particle forces the dirty (scalar) strip path."""
        fields = _make_fields2d()
        _set_field_pattern(fields)
        x_center = (NX / 2) * DX
        y_center = (NY / 2) * DY
        part = make_spin_particles(
            2, x=[x_center, x_center], y=[y_center, y_center],
            ux=[0.3, 0.0], uy=[-0.2, 0.0], uz=[0.5, 0.0],
            sx=[0.0, 1.0], sy=[1.0, 0.0], sz=[0.0, 0.0])
        part.is_dead[1] = True

        snap_dead = (part.ux[1], part.sx[1])
        self._run_and_compare(part, fields, is_clean=False)
        # dead particle untouched
        assert part.ux[1] == snap_dead[0]
        assert part.sx[1] == snap_dead[1]

    def test_multiple_particles(self):
        """Several particles with different momenta/spins in one patch."""
        fields = _make_fields2d()
        _set_field_pattern(fields)
        rng = np.random.default_rng(11)
        n = 7
        xs = rng.uniform(2 * DX, (NX - 2) * DX, n)
        ys = rng.uniform(2 * DY, (NY - 2) * DY, n)
        part = make_spin_particles(
            n, x=xs, y=ys,
            ux=rng.uniform(-1, 1, n), uy=rng.uniform(-1, 1, n),
            uz=rng.uniform(-1, 1, n),
            sx=rng.uniform(-1, 1, n), sy=rng.uniform(-1, 1, n),
            sz=rng.uniform(-1, 1, n))

        snap = {name: getattr(part, name).copy()
                for name in ("x", "y", "ux", "uy", "uz",
                             "sx", "sy", "sz", "inv_gamma")}

        unified_boris_tbmt_pusher_cpu_2d([part], [fields], 1, DT, Q, M,
                                         AE_SCHWINGER)

        for ip in range(n):
            ex_i = np.array([part.ex_part[ip]])
            ey_i = np.array([part.ey_part[ip]])
            ez_i = np.array([part.ez_part[ip]])
            bx_i = np.array([part.bx_part[ip]])
            by_i = np.array([part.by_part[ip]])
            bz_i = np.array([part.bz_part[ip]])

            ref = SpinParticles(ipatch=0, rank=0)
            ref.initialize(1)
            ref.ux[:] = snap["ux"][ip]
            ref.uy[:] = snap["uy"][ip]
            ref.uz[:] = snap["uz"][ip]
            ref.sx[:] = snap["sx"][ip]
            ref.sy[:] = snap["sy"][ip]
            ref.sz[:] = snap["sz"][ip]
            ref.inv_gamma[:] = snap["inv_gamma"][ip]

            xh = snap["x"][ip] + c * 0.5 * DT * snap["inv_gamma"][ip] * snap["ux"][ip]
            yh = snap["y"][ip] + c * 0.5 * DT * snap["inv_gamma"][ip] * snap["uy"][ip]

            boris(
                ref.ux, ref.uy, ref.uz, ref.inv_gamma,
                ex_i, ey_i, ez_i, bx_i, by_i, bz_i,
                Q, M, 1, ref.is_dead, DT)
            tbmt_spin(
                ref.sx, ref.sy, ref.sz,
                ref.ux, ref.uy, ref.uz,
                ex_i, ey_i, ez_i, bx_i, by_i, bz_i,
                Q, M, AE_SCHWINGER, 1, ref.is_dead, DT)

            x_ref = xh + c * 0.5 * DT * ref.inv_gamma[0] * ref.ux[0]
            y_ref = yh + c * 0.5 * DT * ref.inv_gamma[0] * ref.uy[0]

            assert_allclose(part.ux[ip], ref.ux[0], rtol=1e-13)
            assert_allclose(part.uy[ip], ref.uy[0], rtol=1e-13)
            assert_allclose(part.uz[ip], ref.uz[0], rtol=1e-13)
            assert_allclose(part.sx[ip], ref.sx[0], rtol=1e-13)
            assert_allclose(part.sy[ip], ref.sy[0], rtol=1e-13)
            assert_allclose(part.sz[ip], ref.sz[0], rtol=1e-13)
            assert_allclose(part.inv_gamma[ip], ref.inv_gamma[0], rtol=1e-13)
            assert_allclose(part.x[ip], x_ref, rtol=1e-12)
            assert_allclose(part.y[ip], y_ref, rtol=1e-12)


class TestUnifiedTbmtCpu3D:
    """unified_boris_tbmt_pusher_cpu_3d vs the numba kernel."""

    def test_matches_numba(self):
        fields = _make_fields3d()
        # 3D pattern
        x = fields.xaxis / DX
        y = fields.yaxis / DY
        z = fields.zaxis / DZ
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        fields.ex[:, :, :] = np.sin(X) * np.cos(Y) * 1e10 + 2e9
        fields.ey[:, :, :] = np.cos(Y) * np.sin(Z) * 1e10 - 3e9
        fields.ez[:, :, :] = np.cos(X + Y + Z) * 5e10 + 1e9
        fields.bx[:, :, :] = np.sin(0.5 * X) * 3.0 + 1.0
        fields.by[:, :, :] = np.cos(0.7 * Y) * 2.0 - 0.5
        fields.bz[:, :, :] = np.sin(X * Y * Z * 0.001) * 4.0 + 2.5

        part = make_spin_particles(
            1, x=[(NX / 2) * DX], y=[(NY / 2) * DY], z=[(NZ / 2) * DZ],
            ux=[0.3], uy=[-0.2], uz=[0.5],
            sx=[0.0], sy=[1.0], sz=[0.0])

        snap = {name: getattr(part, name).copy()
                for name in ("x", "y", "z", "ux", "uy", "uz",
                             "sx", "sy", "sz", "inv_gamma")}

        unified_boris_tbmt_pusher_cpu_3d([part], [fields], 1, DT, Q, M,
                                         AE_SCHWINGER)

        ref = SpinParticles(ipatch=0, rank=0)
        ref.initialize(1)
        for name in ("ux", "uy", "uz", "sx", "sy", "sz", "inv_gamma"):
            getattr(ref, name)[:] = snap[name][0]

        cdt_half = c * 0.5 * DT
        xh = snap["x"][0] + cdt_half * snap["inv_gamma"][0] * snap["ux"][0]
        yh = snap["y"][0] + cdt_half * snap["inv_gamma"][0] * snap["uy"][0]
        zh = snap["z"][0] + cdt_half * snap["inv_gamma"][0] * snap["uz"][0]

        boris(
            ref.ux, ref.uy, ref.uz, ref.inv_gamma,
            np.array([part.ex_part[0]]), np.array([part.ey_part[0]]),
            np.array([part.ez_part[0]]),
            np.array([part.bx_part[0]]), np.array([part.by_part[0]]),
            np.array([part.bz_part[0]]),
            Q, M, 1, ref.is_dead, DT)
        tbmt_spin(
            ref.sx, ref.sy, ref.sz,
            ref.ux, ref.uy, ref.uz,
            np.array([part.ex_part[0]]), np.array([part.ey_part[0]]),
            np.array([part.ez_part[0]]),
            np.array([part.bx_part[0]]), np.array([part.by_part[0]]),
            np.array([part.bz_part[0]]),
            Q, M, AE_SCHWINGER, 1, ref.is_dead, DT)

        x_ref = xh + cdt_half * ref.inv_gamma[0] * ref.ux[0]
        y_ref = yh + cdt_half * ref.inv_gamma[0] * ref.uy[0]
        z_ref = zh + cdt_half * ref.inv_gamma[0] * ref.uz[0]

        assert_allclose(part.ux[0], ref.ux[0], rtol=1e-13)
        assert_allclose(part.uy[0], ref.uy[0], rtol=1e-13)
        assert_allclose(part.uz[0], ref.uz[0], rtol=1e-13)
        assert_allclose(part.sx[0], ref.sx[0], rtol=1e-13)
        assert_allclose(part.sy[0], ref.sy[0], rtol=1e-13)
        assert_allclose(part.sz[0], ref.sz[0], rtol=1e-13)
        assert_allclose(part.inv_gamma[0], ref.inv_gamma[0], rtol=1e-13)
        assert_allclose(part.x[0], x_ref, rtol=1e-12)
        assert_allclose(part.y[0], y_ref, rtol=1e-12)
        assert_allclose(part.z[0], z_ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. Species / particles plumbing
# ---------------------------------------------------------------------------
class TestSpeciesSpinPlumbing:

    def test_electron_default_anomalous_moment(self):
        ele = Electron(density=lambda x, y: 1.0, ppc=1)
        assert_allclose(ele.anomalous_magnetic_moment, AE_SCHWINGER)

    def test_species_default_anomalous_moment(self):
        ion = Species(name="ion", charge=1, mass=1836.0)
        assert ion.anomalous_magnetic_moment == 0.0

    def test_anomalous_moment_override(self):
        ele = Electron(density=lambda x, y: 1.0, ppc=1,
                       anomalous_magnetic_moment=0.25)
        assert ele.anomalous_magnetic_moment == 0.25

    def test_base_species_with_polarization_creates_spin_particles(self):
        ion = Species(name="ion", charge=1, mass=1836.0,
                      polarization=(0.0, 0.0, 1.0))
        part = ion.create_particles(ipatch=0, rank=0)
        assert isinstance(part, SpinParticles)
        assert "sx" in part.attrs

    def test_electron_with_polarization_creates_spin_particles(self):
        ele = Electron(density=lambda x, y: 1.0, ppc=1,
                       polarization=(1.0, 0.0, 0.0))
        part = ele.create_particles(ipatch=0, rank=0)
        assert isinstance(part, SpinParticles)

    def test_tbmt_without_polarization_raises(self):
        from lambdapic.core.species import _ALL_SPECIES
        sim = Simulation(nx=16, ny=16, dx=0.1, dy=0.1,
                         npatch_x=2, npatch_y=2, dt_cfl=0.95)
        ele = Electron(density=lambda x, y: 1.0, ppc=2,
                       pusher="boris+tbmt")  # no polarization
        try:
            sim.add_species([ele])
            with pytest.raises(ValueError, match="polarization"):
                sim.initialize()
        finally:
            # do not leak into the global registry: later simulations
            # auto-register leftover species (see _add_default_species_if_empty)
            if ele in _ALL_SPECIES:
                _ALL_SPECIES.remove(ele)


# ---------------------------------------------------------------------------
# 4. Full 2D simulation (both dispatch paths)
# ---------------------------------------------------------------------------
B0 = 1000.0  # T
N_STEPS = 50


def _make_tbmt_sim(pusher: str | None = "boris+tbmt") -> Simulation:
    sim = Simulation(nx=32, ny=32, dx=0.1, dy=0.1,
                     npatch_x=2, npatch_y=2, dt_cfl=0.95)
    kwargs = dict(density=lambda x, y: 1.0, ppc=4,
                  polarization=(1.0, 0.0, 0.0))
    if pusher is not None:
        kwargs["pusher"] = pusher
    ele = Electron(**kwargs)
    sim.add_species([ele])
    sim.initialize()
    # uniform magnetic field, electrons start at rest: no current, no E
    for p in sim.patches:
        p.fields.bz[:] = B0
    return sim


def _check_spin_precession(sim: Simulation) -> None:
    """Compare spin of all particles to the analytic TBMT rotation."""
    dt = sim.dt
    theta = analytic_spin_angle(-e * dt * (AE_SCHWINGER + 1.0) * B0 / (2 * m_e))
    total = N_STEPS * theta
    for p in sim.patches:
        part = p.particles[0]
        alive = part.is_alive
        assert alive.any()
        assert_allclose(part.sx[alive], np.cos(total), rtol=1e-6)
        assert_allclose(part.sy[alive], np.sin(total), rtol=1e-6,
                        atol=1e-10)
        assert_allclose(part.sz[alive], 0.0, atol=1e-10)
        # momentum still zero
        assert_allclose(part.ux[alive], 0.0, atol=1e-14)
        assert_allclose(part.uy[alive], 0.0, atol=1e-14)


class TestTbmtSimulation2D:

    def test_unified_path_spin_precession(self):
        """Electrons at rest in uniform B: spin precesses analytically.

        The species has no callbacks in pusher stages, so the run uses the
        unified C kernel path.
        """
        sim = _make_tbmt_sim()
        from lambdapic.core.pusher.pusher import BorisTBMTPusher
        assert isinstance(sim.pusher[0], BorisTBMTPusher)
        sim.run(N_STEPS)
        _check_spin_precession(sim)

    def test_fallback_path_spin_precession(self):
        """Same physics through the numba fallback path.

        A callback registered on a pusher stage disables the unified path.
        """
        @callback(stage="_push_momentum", interval=1)
        def noop(sim):
            pass

        sim = _make_tbmt_sim()
        sim.run(N_STEPS, callbacks=[noop])
        _check_spin_precession(sim)

    def test_spin_initialized_from_polarization(self):
        """fill_particles initializes sx/sy/sz from the species polarization."""
        sim = _make_tbmt_sim()
        for p in sim.patches:
            part = p.particles[0]
            alive = part.is_alive
            assert alive.any()
            assert_allclose(part.sx[alive], 1.0)
            assert_allclose(part.sy[alive], 0.0)
            assert_allclose(part.sz[alive], 0.0)


class TestSpinAutoUpgrade:
    """A polarized species with the default 'boris' pusher auto-upgrades to boris+tbmt."""

    def test_upgraded_dispatch(self):
        """Forgetting pusher='boris+tbmt' still yields the TBMT pusher."""
        from lambdapic.core.pusher.pusher import BorisTBMTPusher

        sim = _make_tbmt_sim(pusher=None)
        assert isinstance(sim.pusher[0], BorisTBMTPusher)
        # the species config is updated to match the actual pusher
        assert sim.patches.species[0].pusher == "boris+tbmt"

    def test_upgrade_warning_logged(self):
        """The upgrade emits a warning (rank 0)."""
        from loguru import logger

        sim = Simulation(nx=32, ny=32, dx=0.1, dy=0.1,
                         npatch_x=2, npatch_y=2, dt_cfl=0.95)
        ele = Electron(density=lambda x, y: 1.0, ppc=4,
                       polarization=(1.0, 0.0, 0.0))
        sim.add_species([ele])

        # the sink must be added after Simulation.__init__ (configure_logger
        # removes existing handlers) and before initialize() (_init_pushers)
        messages = []
        handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            sim.initialize()
        finally:
            logger.remove(handler_id)

        assert any("boris+tbmt" in m for m in messages)

    def test_explicit_tbmt_no_warning(self):
        """Explicit pusher='boris+tbmt' upgrades silently."""
        from loguru import logger

        sim = Simulation(nx=32, ny=32, dx=0.1, dy=0.1,
                         npatch_x=2, npatch_y=2, dt_cfl=0.95)
        ele = Electron(density=lambda x, y: 1.0, ppc=4,
                       polarization=(1.0, 0.0, 0.0),
                       pusher="boris+tbmt")
        sim.add_species([ele])

        messages = []
        handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            sim.initialize()
        finally:
            logger.remove(handler_id)

        assert not messages

    def test_no_upgrade_without_polarization(self):
        """A plain boris species is untouched."""
        from lambdapic.core.pusher.pusher import BorisPusher

        sim = Simulation(nx=32, ny=32, dx=0.1, dy=0.1,
                         npatch_x=2, npatch_y=2, dt_cfl=0.95)
        ele = Electron(density=lambda x, y: 1.0, ppc=4)
        sim.add_species([ele])
        sim.initialize()
        assert isinstance(sim.pusher[0], BorisPusher)
        assert ele.pusher == "boris"

    def test_spin_precession_with_auto_upgrade(self):
        """End-to-end: the upgraded pusher evolves the spin analytically."""
        sim = _make_tbmt_sim(pusher=None)
        sim.run(N_STEPS)
        _check_spin_precession(sim)


class TestTbmtSimulation3D:
    """3D simulation through the unified C kernel path."""

    def test_unified_path_spin_precession_3d(self):
        from lambdapic import Simulation3D

        sim = Simulation3D(nx=16, ny=16, nz=16, dx=0.1, dy=0.1, dz=0.1,
                           npatch_x=2, npatch_y=2, npatch_z=2, dt_cfl=0.95)
        ele = Electron(density=lambda x, y, z: 1.0, ppc=1,
                       polarization=(1.0, 0.0, 0.0),
                       pusher="boris+tbmt")
        sim.add_species([ele])
        sim.initialize()

        for p in sim.patches:
            p.fields.bz[:] = B0

        nsteps = 10
        sim.run(nsteps)

        theta = analytic_spin_angle(-e * sim.dt * (AE_SCHWINGER + 1.0)
                                    * B0 / (2 * m_e))
        total = nsteps * theta
        for p in sim.patches:
            part = p.particles[0]
            alive = part.is_alive
            if not alive.any():
                continue
            assert_allclose(part.sx[alive], np.cos(total), rtol=1e-6)
            assert_allclose(part.sy[alive], np.sin(total), rtol=1e-6,
                            atol=1e-10)
            assert_allclose(part.uz[alive], 0.0, atol=1e-14)
