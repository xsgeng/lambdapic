"""Tests for lambdapic.core.interpolation.cpu3d.

Tests 3D field interpolation using realistic PIC parameters with minimal
objects (no full Simulation, no mocks).

The extension accesses attributes via PyObject_GetAttrString, so objects
only need the right attributes -- duck typing is sufficient.
"""
from __future__ import annotations

import numpy as np

from lambdapic.core.fields import Fields3D
from lambdapic.core.interpolation.cpu3d import interpolation_patches_3d
from lambdapic.core.particles import ParticlesBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fields3d(
    nx: int = 16,
    ny: int = 16,
    nz: int = 16,
    dx: float = 1e-8,
    dy: float = 1e-8,
    dz: float = 1e-8,
    x0: float = 0.0,
    y0: float = 0.0,
    z0: float = 0.0,
    n_guard: int = 3,
) -> Fields3D:
    """Construct a minimal Fields3D-like object."""
    return Fields3D(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx,
        dy=dy,
        dz=dz,
        x0=x0,
        y0=y0,
        z0=z0,
        n_guard=n_guard,
    )


def make_particles(
    n: int,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    z: np.ndarray | None = None,
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
    if z is not None:
        p.z[:] = z
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


# Realistic PIC parameters
NX, NY, NZ = 16, 16, 16
DX, DY, DZ = 1e-8, 1e-8, 1e-8
N_GUARD = 3


# ---------------------------------------------------------------------------
# 1. Basic correctness
# ---------------------------------------------------------------------------

def test_basic_correctness_uniform_fields() -> None:
    """Single particle in uniform fields -> all 6 components correct."""
    fields = make_fields3d(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ, n_guard=N_GUARD)
    val = 3.14
    fields.ex[:] = val
    fields.ey[:] = val
    fields.ez[:] = val
    fields.bx[:] = val
    fields.by[:] = val
    fields.bz[:] = val

    x_c = (NX / 2) * DX
    y_c = (NY / 2) * DY
    z_c = (NZ / 2) * DZ

    particles = make_particles(
        n=1,
        x=np.array([x_c]),
        y=np.array([y_c]),
        z=np.array([z_c]),
        is_dead=np.array([False]),
    )

    interpolation_patches_3d([particles], [fields], 1)

    assert np.isclose(particles.ex_part[0], val)
    assert np.isclose(particles.ey_part[0], val)
    assert np.isclose(particles.ez_part[0], val)
    assert np.isclose(particles.bx_part[0], val)
    assert np.isclose(particles.by_part[0], val)
    assert np.isclose(particles.bz_part[0], val)


# ---------------------------------------------------------------------------
# 2. Dead particle excluded
# ---------------------------------------------------------------------------

def test_dead_particle_excluded() -> None:
    """Dead particles do not receive interpolated fields."""
    fields = make_fields3d(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ, n_guard=N_GUARD)
    val = 2.71
    fields.ex[:] = val
    fields.ey[:] = val
    fields.ez[:] = val
    fields.bx[:] = val
    fields.by[:] = val
    fields.bz[:] = val

    x_c = (NX / 2) * DX
    y_c = (NY / 2) * DY
    z_c = (NZ / 2) * DZ

    particles = make_particles(
        n=2,
        x=np.array([x_c, x_c]),
        y=np.array([y_c, y_c]),
        z=np.array([z_c, z_c]),
        is_dead=np.array([False, True]),
    )

    interpolation_patches_3d([particles], [fields], 1)

    assert np.isclose(particles.ex_part[0], val)
    assert np.isclose(particles.ey_part[0], val)
    assert np.isclose(particles.ez_part[0], val)
    assert np.isclose(particles.bx_part[0], val)
    assert np.isclose(particles.by_part[0], val)
    assert np.isclose(particles.bz_part[0], val)

    assert particles.ex_part[1] == 0.0
    assert particles.ey_part[1] == 0.0
    assert particles.ez_part[1] == 0.0
    assert particles.bx_part[1] == 0.0
    assert particles.by_part[1] == 0.0
    assert particles.bz_part[1] == 0.0


# ---------------------------------------------------------------------------
# 3. Boundary wrap
# ---------------------------------------------------------------------------

def test_boundary_wrap() -> None:
    """Particle near grid boundary interpolates correctly with periodic wrapping."""
    fields = make_fields3d(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ, n_guard=N_GUARD)
    val = 1.23
    # Fill entire arrays (including guard cells) so wrapping always sees the same value
    fields.ex[:] = val
    fields.ey[:] = val
    fields.ez[:] = val
    fields.bx[:] = val
    fields.by[:] = val
    fields.bz[:] = val

    x_near = 0.1 * DX
    y_near = 0.1 * DY
    z_near = 0.1 * DZ

    particles = make_particles(
        n=1,
        x=np.array([x_near]),
        y=np.array([y_near]),
        z=np.array([z_near]),
        is_dead=np.array([False]),
    )

    interpolation_patches_3d([particles], [fields], 1)

    assert np.isclose(particles.ex_part[0], val)
    assert np.isclose(particles.ey_part[0], val)
    assert np.isclose(particles.ez_part[0], val)
    assert np.isclose(particles.bx_part[0], val)
    assert np.isclose(particles.by_part[0], val)
    assert np.isclose(particles.bz_part[0], val)


# ---------------------------------------------------------------------------
# 4. Multiple patches
# ---------------------------------------------------------------------------

def test_multiple_patches() -> None:
    """Two patches interpolate independently without cross-contamination."""
    n_patches = 2
    fields_list = []
    particles_list = []

    for _ in range(n_patches):
        f = make_fields3d(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ, n_guard=N_GUARD)
        f.ex[:] = 1.0
        f.ey[:] = 2.0
        f.ez[:] = 3.0
        f.bx[:] = 4.0
        f.by[:] = 5.0
        f.bz[:] = 6.0
        fields_list.append(f)

        x_c = (NX / 2) * DX
        y_c = (NY / 2) * DY
        z_c = (NZ / 2) * DZ
        p = make_particles(
            n=1,
            x=np.array([x_c]),
            y=np.array([y_c]),
            z=np.array([z_c]),
            is_dead=np.array([False]),
        )
        particles_list.append(p)

    interpolation_patches_3d(particles_list, fields_list, n_patches)

    for p in particles_list:
        assert np.isclose(p.ex_part[0], 1.0)
        assert np.isclose(p.ey_part[0], 2.0)
        assert np.isclose(p.ez_part[0], 3.0)
        assert np.isclose(p.bx_part[0], 4.0)
        assert np.isclose(p.by_part[0], 5.0)
        assert np.isclose(p.bz_part[0], 6.0)


# ---------------------------------------------------------------------------
# 5. Precision / staggered grid verification
# ---------------------------------------------------------------------------

def test_staggered_grid_precision() -> None:
    """Each field component is read from its own staggered grid.

    By assigning a unique constant to every component we guarantee that
    cross-contamination between staggered grids would be caught immediately.
    """
    fields = make_fields3d(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ, n_guard=N_GUARD)
    fields.ex[:] = 1.0
    fields.ey[:] = 2.0
    fields.ez[:] = 3.0
    fields.bx[:] = 4.0
    fields.by[:] = 5.0
    fields.bz[:] = 6.0

    x_c = (NX / 2) * DX
    y_c = (NY / 2) * DY
    z_c = (NZ / 2) * DZ

    particles = make_particles(
        n=1,
        x=np.array([x_c]),
        y=np.array([y_c]),
        z=np.array([z_c]),
        is_dead=np.array([False]),
    )

    interpolation_patches_3d([particles], [fields], 1)

    assert np.isclose(particles.ex_part[0], 1.0)
    assert np.isclose(particles.ey_part[0], 2.0)
    assert np.isclose(particles.ez_part[0], 3.0)
    assert np.isclose(particles.bx_part[0], 4.0)
    assert np.isclose(particles.by_part[0], 5.0)
    assert np.isclose(particles.bz_part[0], 6.0)



