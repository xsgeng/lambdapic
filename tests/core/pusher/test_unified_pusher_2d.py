"""Tests for lambdapic.core.pusher.unified.unified_pusher_2d.

Tests basic correctness, dead-particle exclusion, boundary wrapping,
multiple patches, and current deposition.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import e

from lambdapic.core.fields import Fields2D
from lambdapic.core.particles import ParticlesBase
from lambdapic.core.pusher.unified.unified_pusher_2d import (
    unified_boris_pusher_cpu_2d,
)

# ---------------------------------------------------------------------------
# Realistic PIC parameters
# ---------------------------------------------------------------------------
NX, NY = 16, 16
DX, DY = 1e-8, 1e-8
N_GUARD = 3
DENSITY = 1e27  # m^-3
DT = 1e-17  # s
Q = -e
M = 9.1093837e-31  # electron mass
PPC = 10
W = DENSITY * DX * DY / PPC


def make_fields2d(
    nx: int = NX,
    ny: int = NY,
    dx: float = DX,
    dy: float = DY,
    x0: float = 0.0,
    y0: float = 0.0,
    n_guard: int = N_GUARD,
) -> Fields2D:
    """Construct a minimal Fields2D-like object."""
    return Fields2D(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        x0=x0,
        y0=y0,
        n_guard=n_guard,
    )


def make_particles(
    n: int,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    ux: np.ndarray | None = None,
    uy: np.ndarray | None = None,
    uz: np.ndarray | None = None,
    w: np.ndarray | None = None,
    is_dead: np.ndarray | None = None,
    inv_gamma: np.ndarray | None = None,
) -> ParticlesBase:
    """Construct a minimal ParticlesBase-like object with given arrays."""
    p = ParticlesBase(ipatch=0, rank=0)
    p.initialize(n)

    if x is not None:
        p.x[:] = x
    if y is not None:
        p.y[:] = y
    if ux is not None:
        p.ux[:] = ux
    if uy is not None:
        p.uy[:] = uy
    if uz is not None:
        p.uz[:] = uz
    if w is not None:
        p.w[:] = w
    if is_dead is not None:
        p.is_dead[:] = is_dead
    if inv_gamma is not None:
        p.inv_gamma[:] = inv_gamma

    return p


def _set_nonzero_fields(fields: Fields2D) -> None:
    """Fill E/B fields with nonzero values so interpolation changes particle fields."""
    fields.ex[:, :] = np.sin(fields.xaxis / DX) * np.cos(fields.yaxis / DY)
    fields.ey[:, :] = np.cos(fields.xaxis / DX) * np.sin(fields.yaxis / DY)
    fields.ez[:, :] = 1e10
    fields.bx[:, :] = 1.0
    fields.by[:, :] = 2.0
    fields.bz[:, :] = 3.0




# ---------------------------------------------------------------------------
# 1. Basic correctness
# ---------------------------------------------------------------------------

class TestBasicCorrectness:
    """Single particle: pusher runs, modifies particle and field arrays."""

    def test_runs_and_modifies_arrays(self) -> None:
        fields = make_fields2d()
        _set_nonzero_fields(fields)

        x_center = (NX / 2) * DX
        y_center = (NY / 2) * DY

        particles = make_particles(
            n=1,
            x=np.array([x_center]),
            y=np.array([y_center]),
            ux=np.array([0.01]),
            uy=np.array([0.02]),
            uz=np.array([0.005]),
            w=np.array([W]),
            inv_gamma=np.array([1.0 / np.sqrt(1 + 0.01 ** 2 + 0.02 ** 2 + 0.005 ** 2)]),
            is_dead=np.array([False]),
        )

        x_before = particles.x.copy()
        y_before = particles.y.copy()
        ux_before = particles.ux.copy()
        uy_before = particles.uy.copy()
        uz_before = particles.uz.copy()
        inv_gamma_before = particles.inv_gamma.copy()
        rho_before = fields.rho.copy()
        jx_before = fields.jx.copy()
        jy_before = fields.jy.copy()
        jz_before = fields.jz.copy()

        unified_boris_pusher_cpu_2d([particles], [fields], 1, DT, Q, M)

        assert not np.allclose(particles.x, x_before, rtol=0, atol=1e-20), "x should change"
        assert not np.allclose(particles.y, y_before, rtol=0, atol=1e-20), "y should change"
        assert not np.allclose(particles.ux, ux_before, rtol=0, atol=1e-20), "ux should change"
        assert not np.allclose(particles.uy, uy_before, rtol=0, atol=1e-20), "uy should change"
        assert not np.allclose(particles.uz, uz_before, rtol=0, atol=1e-20), "uz should change"
        assert not np.allclose(particles.inv_gamma, inv_gamma_before, rtol=0, atol=1e-20), "inv_gamma should change"

        assert not np.allclose(fields.rho, rho_before, rtol=0, atol=1e-20), "rho should change"
        assert not np.allclose(fields.jx, jx_before, rtol=0, atol=1e-20), "jx should change"
        assert not np.allclose(fields.jy, jy_before, rtol=0, atol=1e-20), "jy should change"
        assert not np.allclose(fields.jz, jz_before, rtol=0, atol=1e-20), "jz should change"


# ---------------------------------------------------------------------------
# 2. Dead particle excluded
# ---------------------------------------------------------------------------

class TestDeadParticleExcluded:
    """Dead particles must not affect the result."""

    def test_dead_particle_excluded(self) -> None:
        fields = make_fields2d()
        _set_nonzero_fields(fields)

        x_center = (NX / 2) * DX
        y_center = (NY / 2) * DY

        particles = make_particles(
            n=2,
            x=np.array([x_center, x_center]),
            y=np.array([y_center, y_center]),
            ux=np.array([0.1, 0.1]),
            uy=np.array([0.2, 0.2]),
            uz=np.array([0.05, 0.05]),
            w=np.array([W, W]),
            inv_gamma=np.array([1.0, 1.0]),
            is_dead=np.array([False, True]),
        )

        unified_boris_pusher_cpu_2d([particles], [fields], 1, DT, Q, M)

        single_fields = make_fields2d()
        _set_nonzero_fields(single_fields)
        single_particles = make_particles(
            n=1,
            x=np.array([x_center]),
            y=np.array([y_center]),
            ux=np.array([0.1]),
            uy=np.array([0.2]),
            uz=np.array([0.05]),
            w=np.array([W]),
            inv_gamma=np.array([1.0]),
            is_dead=np.array([False]),
        )

        unified_boris_pusher_cpu_2d(
            [single_particles], [single_fields], 1, DT, Q, M
        )

        assert np.allclose(fields.rho, single_fields.rho, rtol=1e-10, atol=1e-15)
        assert np.allclose(fields.jx, single_fields.jx, rtol=1e-10, atol=1e-15)
        assert np.allclose(fields.jy, single_fields.jy, rtol=1e-10, atol=1e-15)
        assert np.allclose(fields.jz, single_fields.jz, rtol=1e-10, atol=1e-15)

        assert np.allclose(particles.x[0], single_particles.x[0], rtol=1e-10, atol=1e-15)
        assert np.allclose(particles.y[0], single_particles.y[0], rtol=1e-10, atol=1e-15)
        assert np.allclose(particles.ux[0], single_particles.ux[0], rtol=1e-10, atol=1e-15)
        assert np.allclose(particles.uy[0], single_particles.uy[0], rtol=1e-10, atol=1e-15)
        assert np.allclose(particles.uz[0], single_particles.uz[0], rtol=1e-10, atol=1e-15)
        assert np.allclose(particles.inv_gamma[0], single_particles.inv_gamma[0], rtol=1e-10, atol=1e-15)


# ---------------------------------------------------------------------------
# 3. Boundary wrap
# ---------------------------------------------------------------------------

class TestBoundaryWrap:
    """Particle near edge: push + deposition works with periodic wrapping."""

    def test_boundary_wrap(self) -> None:
        small_nx, small_ny = 8, 8
        dx_small = 1e-8
        dy_small = 1e-8
        n_guard = 3

        fields = make_fields2d(
            nx=small_nx, ny=small_ny, dx=dx_small, dy=dy_small, n_guard=n_guard
        )
        _set_nonzero_fields(fields)

        x_near_edge = 0.1 * dx_small
        y_near_edge = 0.1 * dy_small

        particles = make_particles(
            n=1,
            x=np.array([x_near_edge]),
            y=np.array([y_near_edge]),
            ux=np.array([-1.0]),
            uy=np.array([-1.0]),
            uz=np.array([0.0]),
            w=np.array([W]),
            inv_gamma=np.array([1.0 / np.sqrt(1 + 1.0 + 1.0)]),
            is_dead=np.array([False]),
        )

        unified_boris_pusher_cpu_2d([particles], [fields], 1, DT, Q, M)

        total_rho = np.sum(fields.rho)
        assert total_rho != 0.0, "boundary-wrapped particle should deposit charge"
        nonzero_cells = np.count_nonzero(fields.rho)
        assert nonzero_cells > 0, "deposition should affect some cells"
        # Particle should have moved
        assert not np.allclose(particles.x, [x_near_edge], atol=1e-20)
        assert not np.allclose(particles.y, [y_near_edge], atol=1e-20)


# ---------------------------------------------------------------------------
# 4. Multiple patches
# ---------------------------------------------------------------------------

class TestMultiplePatches:
    """Two patches evolve independently."""

    def test_multiple_patches(self) -> None:
        n_patches = 2
        nx, ny = 8, 8
        dx, dy = 1e-8, 1e-8
        n_guard = 3

        fields_list = []
        particles_list = []

        for _ in range(n_patches):
            f = make_fields2d(nx=nx, ny=ny, dx=dx, dy=dy, n_guard=n_guard)
            _set_nonzero_fields(f)
            fields_list.append(f)

            x_c = (nx / 2) * dx
            y_c = (ny / 2) * dy
            p = make_particles(
                n=1,
                x=np.array([x_c]),
                y=np.array([y_c]),
                ux=np.array([0.05]),
                uy=np.array([0.1]),
                uz=np.array([0.02]),
                w=np.array([W]),
                inv_gamma=np.array([1.0]),
                is_dead=np.array([False]),
            )
            particles_list.append(p)

        unified_boris_pusher_cpu_2d(
            particles_list, fields_list, n_patches, DT, Q, M
        )

        for f in fields_list:
            assert np.sum(f.rho) != 0.0, "each patch should have deposited charge"

        single_f = make_fields2d(nx=nx, ny=ny, dx=dx, dy=dy, n_guard=n_guard)
        _set_nonzero_fields(single_f)
        single_p = make_particles(
            n=1,
            x=np.array([(nx / 2) * dx]),
            y=np.array([(ny / 2) * dy]),
            ux=np.array([0.05]),
            uy=np.array([0.1]),
            uz=np.array([0.02]),
            w=np.array([W]),
            inv_gamma=np.array([1.0]),
            is_dead=np.array([False]),
        )
        unified_boris_pusher_cpu_2d([single_p], [single_f], 1, DT, Q, M)

        for f in fields_list:
            assert np.allclose(f.rho, single_f.rho, rtol=1e-10, atol=1e-15)
            assert np.allclose(f.jx, single_f.jx, rtol=1e-10, atol=1e-15)
            assert np.allclose(f.jy, single_f.jy, rtol=1e-10, atol=1e-15)
            assert np.allclose(f.jz, single_f.jz, rtol=1e-10, atol=1e-15)


# ---------------------------------------------------------------------------
# 5. Current deposition included
# ---------------------------------------------------------------------------

class TestCurrentDepositionIncluded:
    """The unified pusher must update rho and jx/jy/jz."""

    def test_current_deposition_updates_fields(self) -> None:
        fields = make_fields2d()
        _set_nonzero_fields(fields)

        x_center = (NX / 2) * DX
        y_center = (NY / 2) * DY

        particles = make_particles(
            n=1,
            x=np.array([x_center]),
            y=np.array([y_center]),
            ux=np.array([0.01]),
            uy=np.array([0.02]),
            uz=np.array([0.005]),
            w=np.array([W]),
            inv_gamma=np.array([1.0]),
            is_dead=np.array([False]),
        )

        rho_before = fields.rho.copy()
        jx_before = fields.jx.copy()
        jy_before = fields.jy.copy()
        jz_before = fields.jz.copy()

        unified_boris_pusher_cpu_2d([particles], [fields], 1, DT, Q, M)

        assert not np.allclose(fields.rho, rho_before), "rho should be deposited"
        assert not np.allclose(fields.jx, jx_before), "jx should be deposited"
        assert not np.allclose(fields.jy, jy_before), "jy should be deposited"
        assert not np.allclose(fields.jz, jz_before), "jz should be deposited"



