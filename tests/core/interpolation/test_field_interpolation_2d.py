"""Tests for lambdapic.core.interpolation.cpu2d.

Tests field interpolation from Yee grid to particle positions using realistic
PIC parameters with minimal objects (no full Simulation, no mocks).

The extension accesses attributes via PyObject_GetAttrString, so objects
only need the right attributes -- duck typing is sufficient.
"""
from __future__ import annotations

import numpy as np
from scipy.constants import e

from lambdapic.core.fields import Fields2D
from lambdapic.core.particles import ParticlesBase
from lambdapic.core.interpolation.cpu2d import (
    interpolation_patches_2d,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_fields2d(
    nx: int = 16,
    ny: int = 16,
    dx: float = 1e-8,
    dy: float = 1e-8,
    x0: float = 0.0,
    y0: float = 0.0,
    n_guard: int = 3,
) -> Fields2D:
    """Construct a minimal Fields2D-like object.

    Shape includes guard cells: (nx + 2*n_guard, ny + 2*n_guard).
    """
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


# Realistic PIC parameters (scaled for test convenience)
NX, NY = 16, 16
DX = 1e-8          # 0.01 um
DY = 1e-8
N_GUARD = 3
DENSITY = 1e27     # m^-3
DT = 1e-17         # s
Q = -e             # electron charge
PPC = 10           # particles per cell
# Weight: number density * cell volume / ppc
W = DENSITY * DX * DY / PPC


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

class TestBasicCorrectness:
    """Single particle at cell center with uniform fields."""

    def test_uniform_fields_at_cell_center(self) -> None:
        fields = make_fields2d(nx=NX, ny=NY, dx=DX, dy=DY, n_guard=N_GUARD)
        fields.ex[:] = 1.0
        fields.ey[:] = 0.0
        fields.ez[:] = 0.0
        fields.bx[:] = 0.0
        fields.by[:] = 0.0
        fields.bz[:] = 1.0

        # Cell center of the first interior cell
        x_center = 0.5 * DX
        y_center = 0.5 * DY

        particles = make_particles(
            n=1,
            x=np.array([x_center]),
            y=np.array([y_center]),
            ux=np.array([0.0]),
            uy=np.array([0.0]),
            uz=np.array([0.0]),
            w=np.array([W]),
            inv_gamma=np.array([1.0]),
            is_dead=np.array([False]),
        )

        interpolation_patches_2d([particles], [fields], 1)

        assert np.isclose(particles.ex_part[0], 1.0)
        assert np.isclose(particles.ey_part[0], 0.0)
        assert np.isclose(particles.ez_part[0], 0.0)
        assert np.isclose(particles.bx_part[0], 0.0)
        assert np.isclose(particles.by_part[0], 0.0)
        assert np.isclose(particles.bz_part[0], 1.0)


# ---------------------------------------------------------------------------
# Dead particle excluded
# ---------------------------------------------------------------------------

class TestDeadParticleExcluded:
    """Dead particles do not receive interpolated fields."""

    def test_dead_particle_excluded(self) -> None:
        fields = make_fields2d(nx=NX, ny=NY, dx=DX, dy=DY, n_guard=N_GUARD)
        fields.ex[:] = 2.0
        fields.ey[:] = 3.0
        fields.ez[:] = 4.0
        fields.bx[:] = 5.0
        fields.by[:] = 6.0
        fields.bz[:] = 7.0

        x_center = 0.5 * DX
        y_center = 0.5 * DY

        particles = make_particles(
            n=2,
            x=np.array([x_center, x_center]),
            y=np.array([y_center, y_center]),
            ux=np.array([0.0, 0.0]),
            uy=np.array([0.0, 0.0]),
            uz=np.array([0.0, 0.0]),
            w=np.array([W, W]),
            inv_gamma=np.array([1.0, 1.0]),
            is_dead=np.array([False, True]),
        )

        interpolation_patches_2d([particles], [fields], 1)

        # Baseline with a single alive particle
        single_fields = make_fields2d(nx=NX, ny=NY, dx=DX, dy=DY, n_guard=N_GUARD)
        single_fields.ex[:] = 2.0
        single_fields.ey[:] = 3.0
        single_fields.ez[:] = 4.0
        single_fields.bx[:] = 5.0
        single_fields.by[:] = 6.0
        single_fields.bz[:] = 7.0

        single_particles = make_particles(
            n=1,
            x=np.array([x_center]),
            y=np.array([y_center]),
            ux=np.array([0.0]),
            uy=np.array([0.0]),
            uz=np.array([0.0]),
            w=np.array([W]),
            inv_gamma=np.array([1.0]),
            is_dead=np.array([False]),
        )

        interpolation_patches_2d([single_particles], [single_fields], 1)

        # Alive particle should match the single-particle baseline
        assert np.allclose(particles.ex_part[0], single_particles.ex_part[0], rtol=1e-10)
        assert np.allclose(particles.ey_part[0], single_particles.ey_part[0], rtol=1e-10)
        assert np.allclose(particles.ez_part[0], single_particles.ez_part[0], rtol=1e-10)
        assert np.allclose(particles.bx_part[0], single_particles.bx_part[0], rtol=1e-10)
        assert np.allclose(particles.by_part[0], single_particles.by_part[0], rtol=1e-10)
        assert np.allclose(particles.bz_part[0], single_particles.bz_part[0], rtol=1e-10)

        # Dead particle should retain its initialized zero values
        assert np.isclose(particles.ex_part[1], 0.0)
        assert np.isclose(particles.ey_part[1], 0.0)
        assert np.isclose(particles.ez_part[1], 0.0)
        assert np.isclose(particles.bx_part[1], 0.0)
        assert np.isclose(particles.by_part[1], 0.0)
        assert np.isclose(particles.bz_part[1], 0.0)


# ---------------------------------------------------------------------------
# Boundary wrap
# ---------------------------------------------------------------------------

class TestBoundaryWrap:
    """Particles near grid boundaries interpolate with periodic wrapping."""

    def test_boundary_wrap(self) -> None:
        fields = make_fields2d(nx=8, ny=8, dx=DX, dy=DY, n_guard=N_GUARD)
        fields.ex[:] = 1.0
        fields.ey[:] = 2.0
        fields.ez[:] = 3.0
        fields.bx[:] = 4.0
        fields.by[:] = 5.0
        fields.bz[:] = 6.0

        x_near_edge = 0.1 * DX
        y_near_edge = 0.1 * DY

        particles = make_particles(
            n=1,
            x=np.array([x_near_edge]),
            y=np.array([y_near_edge]),
            ux=np.array([0.0]),
            uy=np.array([0.0]),
            uz=np.array([0.0]),
            w=np.array([W]),
            inv_gamma=np.array([1.0]),
            is_dead=np.array([False]),
        )

        interpolation_patches_2d([particles], [fields], 1)

        # Should get valid interpolated values (not nan)
        assert not np.isnan(particles.ex_part[0])
        assert not np.isnan(particles.ey_part[0])
        assert not np.isnan(particles.ez_part[0])
        assert not np.isnan(particles.bx_part[0])
        assert not np.isnan(particles.by_part[0])
        assert not np.isnan(particles.bz_part[0])

        # For uniform fields, should recover the constant values
        assert np.isclose(particles.ex_part[0], 1.0)
        assert np.isclose(particles.ey_part[0], 2.0)
        assert np.isclose(particles.ez_part[0], 3.0)
        assert np.isclose(particles.bx_part[0], 4.0)
        assert np.isclose(particles.by_part[0], 5.0)
        assert np.isclose(particles.bz_part[0], 6.0)


# ---------------------------------------------------------------------------
# Multiple patches
# ---------------------------------------------------------------------------

class TestMultiplePatches:
    """Multiple patches interpolate independently without cross-contamination."""

    def test_multiple_patches(self) -> None:
        n_patches = 2
        nx, ny = 8, 8
        dx, dy = DX, DY
        n_guard = N_GUARD

        fields_list = []
        particles_list = []

        for _ip in range(n_patches):
            f = make_fields2d(nx=nx, ny=ny, dx=dx, dy=dy, n_guard=n_guard)
            f.ex[:] = 1.0
            f.ey[:] = 2.0
            f.ez[:] = 3.0
            f.bx[:] = 4.0
            f.by[:] = 5.0
            f.bz[:] = 6.0
            fields_list.append(f)

            x_c = 0.5 * dx
            y_c = 0.5 * dy
            p = make_particles(
                n=1,
                x=np.array([x_c]),
                y=np.array([y_c]),
                ux=np.array([0.0]),
                uy=np.array([0.0]),
                uz=np.array([0.0]),
                w=np.array([W]),
                inv_gamma=np.array([1.0]),
                is_dead=np.array([False]),
            )
            particles_list.append(p)

        interpolation_patches_2d(particles_list, fields_list, n_patches)

        for p in particles_list:
            assert np.isclose(p.ex_part[0], 1.0)
            assert np.isclose(p.ey_part[0], 2.0)
            assert np.isclose(p.ez_part[0], 3.0)
            assert np.isclose(p.bx_part[0], 4.0)
            assert np.isclose(p.by_part[0], 5.0)
            assert np.isclose(p.bz_part[0], 6.0)


# ---------------------------------------------------------------------------
# Precision / Staggered grid
# ---------------------------------------------------------------------------

class TestPrecisionStaggeredGrid:
    """Constant fields are preserved exactly at various positions despite Yee staggering."""

    def test_staggered_constant_fields(self) -> None:
        fields = make_fields2d(nx=NX, ny=NY, dx=DX, dy=DY, n_guard=N_GUARD)
        fields.ex[:] = 1.0
        fields.ey[:] = 2.0
        fields.ez[:] = 3.0
        fields.bx[:] = 4.0
        fields.by[:] = 5.0
        fields.bz[:] = 6.0

        # Test multiple representative positions
        positions = [
            (0.5 * DX, 0.5 * DY),                # cell center
            (DX, DY),                            # cell corner
            (0.25 * DX, 0.75 * DY),              # arbitrary offset
            ((NX - 0.5) * DX, (NY - 0.5) * DY),  # far interior cell center
        ]

        for x, y in positions:
            particles = make_particles(
                n=1,
                x=np.array([x]),
                y=np.array([y]),
                ux=np.array([0.0]),
                uy=np.array([0.0]),
                uz=np.array([0.0]),
                w=np.array([W]),
                inv_gamma=np.array([1.0]),
                is_dead=np.array([False]),
            )

            interpolation_patches_2d([particles], [fields], 1)

            assert np.isclose(particles.ex_part[0], 1.0), f"ex_part at ({x}, {y})"
            assert np.isclose(particles.ey_part[0], 2.0), f"ey_part at ({x}, {y})"
            assert np.isclose(particles.ez_part[0], 3.0), f"ez_part at ({x}, {y})"
            assert np.isclose(particles.bx_part[0], 4.0), f"bx_part at ({x}, {y})"
            assert np.isclose(particles.by_part[0], 5.0), f"by_part at ({x}, {y})"
            assert np.isclose(particles.bz_part[0], 6.0), f"bz_part at ({x}, {y})"


# ---------------------------------------------------------------------------
# Non-uniform field correctness (regression tests with structured data)
# ---------------------------------------------------------------------------

class TestNonUniformFields:
    """Interpolation on non-uniform fields with varied particle positions."""

    def test_single_patch_non_uniform(self) -> None:
        np.random.seed(42)
        nx, ny = 16, 16
        dx, dy = 1e-8, 1e-8
        n_guard = 3
        npart = 10

        fields = make_fields2d(nx=nx, ny=ny, dx=dx, dy=dy, n_guard=n_guard)

        # Fill with structured non-uniform data so staggered offsets matter
        i_idx = np.arange(fields.ex.shape[0])[:, None]
        j_idx = np.arange(fields.ex.shape[1])[None, :]

        fields.ex[:] = np.sin(2 * np.pi * i_idx / nx) * np.cos(2 * np.pi * j_idx / ny)
        fields.ey[:] = np.cos(2 * np.pi * i_idx / nx) * np.sin(2 * np.pi * j_idx / ny)
        fields.ez[:] = np.sin(2 * np.pi * (i_idx + j_idx) / nx)
        fields.bx[:] = np.cos(2 * np.pi * (i_idx - j_idx) / ny)
        fields.by[:] = np.sin(2 * np.pi * i_idx / nx) + np.cos(2 * np.pi * j_idx / ny)
        fields.bz[:] = np.cos(2 * np.pi * i_idx / nx) * np.cos(2 * np.pi * j_idx / ny)

        # Particles at varied positions within the interior
        x = np.random.uniform(0.5 * dx, (nx - 0.5) * dx, npart)
        y = np.random.uniform(0.5 * dy, (ny - 0.5) * dy, npart)

        particles = make_particles(
            n=npart,
            x=x.copy(),
            y=y.copy(),
            ux=np.random.uniform(-0.1, 0.1, npart),
            uy=np.random.uniform(-0.1, 0.1, npart),
            uz=np.random.uniform(-0.1, 0.1, npart),
            w=np.full(npart, W),
            inv_gamma=np.full(npart, 1.0),
            is_dead=np.full(npart, False),
        )

        interpolation_patches_2d([particles], [fields], 1)

        # Should produce valid (non-nan) results
        assert not np.any(np.isnan(particles.ex_part))
        assert not np.any(np.isnan(particles.ey_part))
        assert not np.any(np.isnan(particles.ez_part))
        assert not np.any(np.isnan(particles.bx_part))
        assert not np.any(np.isnan(particles.by_part))
        assert not np.any(np.isnan(particles.bz_part))

    def test_multiple_patches_non_uniform(self) -> None:
        np.random.seed(123)
        n_patches = 2
        nx, ny = 8, 8
        dx, dy = 1e-8, 1e-8
        n_guard = 3
        npart = 5

        fields_list = []
        particles_list = []

        for ip in range(n_patches):
            f = make_fields2d(nx=nx, ny=ny, dx=dx, dy=dy, n_guard=n_guard)

            i_idx = np.arange(f.ex.shape[0])[:, None]
            j_idx = np.arange(f.ex.shape[1])[None, :]
            phase = ip * 0.5

            f.ex[:] = np.sin(2 * np.pi * i_idx / nx + phase) * np.cos(2 * np.pi * j_idx / ny)
            f.ey[:] = np.cos(2 * np.pi * i_idx / nx) * np.sin(2 * np.pi * j_idx / ny + phase)
            f.ez[:] = np.sin(2 * np.pi * (i_idx + j_idx) / nx + phase)
            f.bx[:] = np.cos(2 * np.pi * (i_idx - j_idx) / ny + phase)
            f.by[:] = np.sin(2 * np.pi * i_idx / nx + phase) + np.cos(2 * np.pi * j_idx / ny)
            f.bz[:] = np.cos(2 * np.pi * i_idx / nx) * np.cos(2 * np.pi * j_idx / ny + phase)

            fields_list.append(f)

            x = np.random.uniform(0.5 * dx, (nx - 0.5) * dx, npart)
            y = np.random.uniform(0.5 * dy, (ny - 0.5) * dy, npart)

            p = make_particles(
                n=npart,
                x=x.copy(),
                y=y.copy(),
                ux=np.random.uniform(-0.1, 0.1, npart),
                uy=np.random.uniform(-0.1, 0.1, npart),
                uz=np.random.uniform(-0.1, 0.1, npart),
                w=np.full(npart, W),
                inv_gamma=np.full(npart, 1.0),
                is_dead=np.full(npart, False),
            )
            particles_list.append(p)

        interpolation_patches_2d(particles_list, fields_list, n_patches)

        for p in particles_list:
            # Should produce valid (non-nan) results
            assert not np.any(np.isnan(p.ex_part))
            assert not np.any(np.isnan(p.ey_part))
            assert not np.any(np.isnan(p.ez_part))
            assert not np.any(np.isnan(p.bx_part))
            assert not np.any(np.isnan(p.by_part))
            assert not np.any(np.isnan(p.bz_part))
