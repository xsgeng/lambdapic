"""Tests for lambdapic.core.pusher.unified.unified_pusher_3d.

Tests unified_boris_pusher_cpu_3d using realistic PIC parameters with minimal
objects (no full Simulation, no mocks).

The extension accesses attributes via PyObject_GetAttrString, so objects only
need the right attributes -- duck typing is sufficient.
"""
from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import e

from lambdapic.core.fields import Fields3D
from lambdapic.core.particles import ParticlesBase
from lambdapic.core.pusher.unified.unified_pusher_3d import (
    unified_boris_pusher_cpu_3d,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NX, NY, NZ = 16, 16, 16
DX = DY = DZ = 1e-8  # 0.01 um
N_GUARD = 3
DT = 1e-17  # s
Q = -e  # electron charge
M = 9.1093837e-31  # kg
DENSITY = 1e27  # m^-3
PPC = 10
# Weight: number density * cell volume / ppc
W = DENSITY * DX * DY * DZ / PPC


def make_fields3d(
    nx: int = NX,
    ny: int = NY,
    nz: int = NZ,
    dx: float = DX,
    dy: float = DY,
    dz: float = DZ,
    x0: float = 0.0,
    y0: float = 0.0,
    z0: float = 0.0,
    n_guard: int = N_GUARD,
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
    else:
        # Compute correct inv_gamma from momentum if momentum was provided
        if ux is not None and uy is not None and uz is not None:
            p.inv_gamma[:] = 1.0 / np.sqrt(1.0 + ux**2 + uy**2 + uz**2)

    return p


def set_constant_fields(fields: Fields3D) -> None:
    """Set E and B fields to constant nonzero values for interpolation."""
    fields.ex[:] = 1e10
    fields.ey[:] = 2e10
    fields.ez[:] = 3e10
    fields.bx[:] = 0.1
    fields.by[:] = 0.2
    fields.bz[:] = 0.3


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicCorrectness:
    """A single particle is pushed, interpolated, and deposits current."""

    def test_single_particle_modified(self) -> None:
        fields = make_fields3d()
        set_constant_fields(fields)
        # Zero current arrays so we can detect deposition
        fields.rho[:] = 0.0
        fields.jx[:] = 0.0
        fields.jy[:] = 0.0
        fields.jz[:] = 0.0

        x_center = (NX / 2) * DX
        y_center = (NY / 2) * DY
        z_center = (NZ / 2) * DZ

        ux0, uy0, uz0 = 0.1, 0.2, 0.05
        inv_gamma0 = 1.0 / np.sqrt(1.0 + ux0**2 + uy0**2 + uz0**2)

        particles = make_particles(
            n=1,
            x=np.array([x_center]),
            y=np.array([y_center]),
            z=np.array([z_center]),
            ux=np.array([ux0]),
            uy=np.array([uy0]),
            uz=np.array([uz0]),
            w=np.array([W]),
            inv_gamma=np.array([inv_gamma0]),
            is_dead=np.array([False]),
        )

        unified_boris_pusher_cpu_3d([particles], [fields], 1, DT, Q, M)

        # Position should have moved (total push of dt)
        assert particles.x[0] != x_center, "x should have changed"
        assert particles.y[0] != y_center, "y should have changed"
        assert particles.z[0] != z_center, "z should have changed"

        # Momentum should have changed due to Boris push
        assert particles.ux[0] != ux0, "ux should have changed"
        assert particles.uy[0] != uy0, "uy should have changed"
        assert particles.uz[0] != uz0, "uz should have changed"

        # inv_gamma should have been updated
        new_inv_gamma = 1.0 / np.sqrt(
            1.0
            + particles.ux[0] ** 2
            + particles.uy[0] ** 2
            + particles.uz[0] ** 2
        )
        assert np.isclose(particles.inv_gamma[0], new_inv_gamma), "inv_gamma should be consistent with new momentum"
        assert particles.inv_gamma[0] != inv_gamma0, "inv_gamma should have changed"

        # Interpolated fields should have been stored on particle
        assert particles.ex_part[0] != 0.0, "ex_part should be set"
        assert particles.ey_part[0] != 0.0, "ey_part should be set"
        assert particles.ez_part[0] != 0.0, "ez_part should be set"
        assert particles.bx_part[0] != 0.0, "bx_part should be set"
        assert particles.by_part[0] != 0.0, "by_part should be set"
        assert particles.bz_part[0] != 0.0, "bz_part should be set"

        # Current should have been deposited
        assert np.sum(fields.rho) != 0.0, "rho should have been deposited"
        assert np.sum(fields.jx) != 0.0, "jx should have been deposited"
        assert np.sum(fields.jy) != 0.0, "jy should have been deposited"
        assert np.sum(fields.jz) != 0.0, "jz should have been deposited"


class TestDeadParticleExcluded:
    """Dead particles do not contribute to deposition or get pushed."""

    def test_dead_particle_excluded(self) -> None:
        fields = make_fields3d()
        set_constant_fields(fields)
        fields.rho[:] = 0.0
        fields.jx[:] = 0.0
        fields.jy[:] = 0.0
        fields.jz[:] = 0.0

        x_center = (NX / 2) * DX
        y_center = (NY / 2) * DY
        z_center = (NZ / 2) * DZ

        ux0, uy0, uz0 = 0.1, 0.2, 0.05
        inv_gamma0 = 1.0 / np.sqrt(1.0 + ux0**2 + uy0**2 + uz0**2)

        particles_two = make_particles(
            n=2,
            x=np.array([x_center, x_center]),
            y=np.array([y_center, y_center]),
            z=np.array([z_center, z_center]),
            ux=np.array([ux0, ux0]),
            uy=np.array([uy0, uy0]),
            uz=np.array([uz0, uz0]),
            w=np.array([W, W]),
            inv_gamma=np.array([inv_gamma0, inv_gamma0]),
            is_dead=np.array([False, True]),
        )

        unified_boris_pusher_cpu_3d([particles_two], [fields], 1, DT, Q, M)

        # The alive particle should have been pushed
        assert particles_two.x[0] != x_center
        assert particles_two.ux[0] != ux0

        # The dead particle should be untouched
        assert particles_two.x[1] == x_center, "dead particle x should be unchanged"
        assert particles_two.y[1] == y_center, "dead particle y should be unchanged"
        assert particles_two.z[1] == z_center, "dead particle z should be unchanged"
        assert particles_two.ux[1] == ux0, "dead particle ux should be unchanged"
        assert particles_two.uy[1] == uy0, "dead particle uy should be unchanged"
        assert particles_two.uz[1] == uz0, "dead particle uz should be unchanged"

        # Compare with a single alive particle
        fields_single = make_fields3d()
        set_constant_fields(fields_single)
        fields_single.rho[:] = 0.0
        fields_single.jx[:] = 0.0
        fields_single.jy[:] = 0.0
        fields_single.jz[:] = 0.0

        particles_single = make_particles(
            n=1,
            x=np.array([x_center]),
            y=np.array([y_center]),
            z=np.array([z_center]),
            ux=np.array([ux0]),
            uy=np.array([uy0]),
            uz=np.array([uz0]),
            w=np.array([W]),
            inv_gamma=np.array([inv_gamma0]),
            is_dead=np.array([False]),
        )

        unified_boris_pusher_cpu_3d([particles_single], [fields_single], 1, DT, Q, M)

        assert np.allclose(fields.rho, fields_single.rho, rtol=1e-10)
        assert np.allclose(fields.jx, fields_single.jx, rtol=1e-10)
        assert np.allclose(fields.jy, fields_single.jy, rtol=1e-10)
        assert np.allclose(fields.jz, fields_single.jz, rtol=1e-10)


class TestBoundaryWrap:
    """Particles near grid boundaries deposit with periodic wrapping (INDEX3)."""

    def test_boundary_wrap(self) -> None:
        small_n = 8
        dx_small = 1e-8
        dy_small = 1e-8
        dz_small = 1e-8
        n_guard = 3

        fields = make_fields3d(
            nx=small_n,
            ny=small_n,
            nz=small_n,
            dx=dx_small,
            dy=dy_small,
            dz=dz_small,
            n_guard=n_guard,
        )
        set_constant_fields(fields)
        fields.rho[:] = 0.0
        fields.jx[:] = 0.0
        fields.jy[:] = 0.0
        fields.jz[:] = 0.0

        x_near_edge = 0.1 * dx_small
        y_near_edge = 0.1 * dy_small
        z_near_edge = 0.1 * dz_small

        particles = make_particles(
            n=1,
            x=np.array([x_near_edge]),
            y=np.array([y_near_edge]),
            z=np.array([z_near_edge]),
            ux=np.array([0.1]),
            uy=np.array([0.1]),
            uz=np.array([0.1]),
            w=np.array([W]),
            is_dead=np.array([False]),
        )

        unified_boris_pusher_cpu_3d([particles], [fields], 1, DT, Q, M)

        total_rho = np.sum(fields.rho)
        assert total_rho != 0.0, "boundary-wrapped particle should deposit charge"
        nonzero_cells = np.count_nonzero(fields.rho)
        assert nonzero_cells > 0, "deposition should affect some cells"


class TestMultiplePatches:
    """Multiple patches process particles independently without cross-contamination."""

    def test_multiple_patches(self) -> None:
        n_patches = 2
        nx, ny, nz = 8, 8, 8
        dx, dy, dz = 1e-8, 1e-8, 1e-8
        n_guard = 3

        fields_list = []
        particles_list = []

        for _ip in range(n_patches):
            f = make_fields3d(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, n_guard=n_guard)
            set_constant_fields(f)
            f.rho[:] = 0.0
            f.jx[:] = 0.0
            f.jy[:] = 0.0
            f.jz[:] = 0.0
            fields_list.append(f)

            x_c = (nx / 2) * dx
            y_c = (ny / 2) * dy
            z_c = (nz / 2) * dz
            p = make_particles(
                n=1,
                x=np.array([x_c]),
                y=np.array([y_c]),
                z=np.array([z_c]),
                ux=np.array([0.05]),
                uy=np.array([0.1]),
                uz=np.array([0.02]),
                w=np.array([W]),
                is_dead=np.array([False]),
            )
            particles_list.append(p)

        unified_boris_pusher_cpu_3d(particles_list, fields_list, n_patches, DT, Q, M)

        for f in fields_list:
            assert np.sum(f.rho) != 0.0, "each patch should have deposited charge"

        # Each patch should see the same result
        assert np.allclose(fields_list[0].rho, fields_list[1].rho, rtol=1e-10)
        assert np.allclose(fields_list[0].jx, fields_list[1].jx, rtol=1e-10)
        assert np.allclose(fields_list[0].jy, fields_list[1].jy, rtol=1e-10)
        assert np.allclose(fields_list[0].jz, fields_list[1].jz, rtol=1e-10)


class TestCurrentDepositionIncluded:
    """The unified pusher deposits current onto the field arrays."""

    def test_fields_updated(self) -> None:
        fields = make_fields3d()
        set_constant_fields(fields)
        fields.rho[:] = 0.0
        fields.jx[:] = 0.0
        fields.jy[:] = 0.0
        fields.jz[:] = 0.0

        x_center = (NX / 2) * DX
        y_center = (NY / 2) * DY
        z_center = (NZ / 2) * DZ

        particles = make_particles(
            n=1,
            x=np.array([x_center]),
            y=np.array([y_center]),
            z=np.array([z_center]),
            ux=np.array([0.1]),
            uy=np.array([0.2]),
            uz=np.array([0.05]),
            w=np.array([W]),
            is_dead=np.array([False]),
        )

        unified_boris_pusher_cpu_3d([particles], [fields], 1, DT, Q, M)

        assert not np.allclose(fields.rho, 0.0), "rho should be nonzero after deposition"
        assert not np.allclose(fields.jx, 0.0), "jx should be nonzero after deposition"
        assert not np.allclose(fields.jy, 0.0), "jy should be nonzero after deposition"
        assert not np.allclose(fields.jz, 0.0), "jz should be nonzero after deposition"


class TestMultiParticleCorrectness:
    """Multi-particle correctness with mixed alive/dead states."""

    def test_multi_particle_correctness(self) -> None:
        n_patches = 2
        nx, ny, nz = 8, 8, 8
        dx, dy, dz = 1e-8, 1e-8, 1e-8
        n_guard = 3
        dt = 1e-17
        q = -e
        m = 9.1093837e-31

        fields_list = []
        particles_list = []
        for ip in range(n_patches):
            f = make_fields3d(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, n_guard=n_guard)
            f.ex[:] = 1e10 + ip * 1e9
            f.ey[:] = 2e10 + ip * 1e9
            f.ez[:] = 3e10 + ip * 1e9
            f.bx[:] = 0.1 + ip * 0.01
            f.by[:] = 0.2 + ip * 0.01
            f.bz[:] = 0.3 + ip * 0.01
            f.rho[:] = 0.0
            f.jx[:] = 0.0
            f.jy[:] = 0.0
            f.jz[:] = 0.0
            fields_list.append(f)

            np.random.seed(ip + 42)
            n_part = 5
            x = np.random.uniform(2 * dx, (nx - 2) * dx, n_part)
            y = np.random.uniform(2 * dy, (ny - 2) * dy, n_part)
            z = np.random.uniform(2 * dz, (nz - 2) * dz, n_part)
            ux = np.random.uniform(-0.2, 0.2, n_part)
            uy = np.random.uniform(-0.2, 0.2, n_part)
            uz = np.random.uniform(-0.2, 0.2, n_part)
            inv_gamma = 1.0 / np.sqrt(1.0 + ux**2 + uy**2 + uz**2)
            w = np.full(n_part, W)
            is_dead = np.array([False, False, True, False, False])

            p = make_particles(
                n=n_part,
                x=x,
                y=y,
                z=z,
                ux=ux,
                uy=uy,
                uz=uz,
                w=w,
                inv_gamma=inv_gamma,
                is_dead=is_dead,
            )
            particles_list.append(p)

        original_x = [p.x.copy() for p in particles_list]

        unified_boris_pusher_cpu_3d(particles_list, fields_list, n_patches, dt, q, m)

        for i in range(n_patches):
            assert np.sum(fields_list[i].rho) != 0.0
            assert np.sum(fields_list[i].jx) != 0.0
            assert np.sum(fields_list[i].jy) != 0.0
            assert np.sum(fields_list[i].jz) != 0.0

        for i in range(n_patches):
            for j in range(n_part):
                if not particles_list[i].is_dead[j]:
                    assert particles_list[i].x[j] != original_x[i][j]
                    assert not np.isnan(particles_list[i].x[j])
                else:
                    assert particles_list[i].x[j] == original_x[i][j]
