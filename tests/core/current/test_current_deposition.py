"""Tests for lambdapic.core.current.cpu2d and cpu3d.

Tests reset_current_cpu_2d, current_deposition_cpu_2d, and
current_deposition_cpu_3d using realistic PIC parameters with minimal
objects (no full Simulation, no mocks).

The extension accesses attributes via PyObject_GetAttrString, so objects
only need the right attributes -- duck typing is sufficient.
"""
from __future__ import annotations

import os
from time import perf_counter_ns

import numpy as np
from scipy.constants import c, e

from lambdapic.core.current.cpu2d import (
    current_deposition_cpu_2d,
    reset_current_cpu_2d,
)
from lambdapic.core.current.cpu3d import current_deposition_cpu_3d
from lambdapic.core.current.deposition import CurrentDeposition2D, CurrentDeposition3D
from lambdapic.core.fields import Fields2D, Fields3D
from lambdapic.core.particles import ParticlesBase
from lambdapic.core.patch.patch import Patch2D, Patch3D, Patches
from lambdapic.core.species import Electron, Proton


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
# Old coverage: reset & basic 2D deposition behaviour
# ---------------------------------------------------------------------------

class TestResetZeroesAllArrays:
    """reset_current_cpu_2d sets jx, jy, jz, rho to zero."""

    def test_reset_zeroes_all_arrays(self) -> None:
        fields = make_fields2d(nx=NX, ny=NY, dx=DX, dy=DY, n_guard=N_GUARD)
        fields.rho[:] = 1.23
        fields.jx[:] = 4.56
        fields.jy[:] = 7.89
        fields.jz[:] = -0.12

        reset_current_cpu_2d([fields], 1)

        assert np.allclose(fields.rho, 0.0), "rho not zeroed"
        assert np.allclose(fields.jx, 0.0), "jx not zeroed"
        assert np.allclose(fields.jy, 0.0), "jy not zeroed"
        assert np.allclose(fields.jz, 0.0), "jz not zeroed"


class TestResetIdempotent:
    """Calling reset_current_cpu_2d twice is harmless."""

    def test_reset_idempotent(self) -> None:
        fields = make_fields2d(nx=NX, ny=NY, dx=DX, dy=DY, n_guard=N_GUARD)
        fields.rho[:] = 1.0
        fields.jx[:] = 1.0
        fields.jy[:] = 1.0
        fields.jz[:] = 1.0

        reset_current_cpu_2d([fields], 1)
        reset_current_cpu_2d([fields], 1)

        assert np.allclose(fields.rho, 0.0)
        assert np.allclose(fields.jx, 0.0)
        assert np.allclose(fields.jy, 0.0)
        assert np.allclose(fields.jz, 0.0)


class TestResetEmptyList:
    """reset_current_cpu_2d handles empty list / npatches=0."""

    def test_reset_empty_list(self) -> None:
        reset_current_cpu_2d([], 0)


class TestDepositionUpdatesArrays:
    """A single moving particle deposits nonzero rho and current."""

    def test_deposition_updates_arrays(self) -> None:
        fields = make_fields2d(nx=NX, ny=NY, dx=DX, dy=DY, n_guard=N_GUARD)
        x_center = (NX / 2) * DX
        y_center = (NY / 2) * DY

        npart = 1
        particles = make_particles(
            n=npart,
            x=np.array([x_center]),
            y=np.array([y_center]),
            ux=np.array([0.01]),
            uy=np.array([0.02]),
            uz=np.array([0.005]),
            w=np.array([W]),
            inv_gamma=np.array([1.0]),
            is_dead=np.array([False]),
        )

        current_deposition_cpu_2d([fields], [particles], 1, DT, Q)

        total_rho = np.sum(fields.rho)
        assert total_rho != 0.0, "rho should have been deposited"
        assert np.abs(np.sum(fields.jx)) > 0, "jx should be nonzero"
        assert np.abs(np.sum(fields.jy)) > 0, "jy should be nonzero"
        assert fields.rho.shape == (NX + 2 * N_GUARD, NY + 2 * N_GUARD)


class TestDeadParticleExcluded:
    """Dead particles do not contribute to deposition."""

    def test_dead_particle_excluded(self) -> None:
        fields = make_fields2d(nx=NX, ny=NY, dx=DX, dy=DY, n_guard=N_GUARD)

        npart = 2
        x_center = (NX / 2) * DX
        y_center = (NY / 2) * DY

        particles = make_particles(
            n=npart,
            x=np.array([x_center, x_center]),
            y=np.array([y_center, y_center]),
            ux=np.array([0.1, 0.1]),
            uy=np.array([0.2, 0.2]),
            uz=np.array([0.05, 0.05]),
            w=np.array([W, W]),
            inv_gamma=np.array([1.0, 1.0]),
            is_dead=np.array([False, True]),
        )

        current_deposition_cpu_2d([fields], [particles], 1, DT, Q)

        single_fields = make_fields2d(nx=NX, ny=NY, dx=DX, dy=DY, n_guard=N_GUARD)
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
        current_deposition_cpu_2d([single_fields], [single_particles], 1, DT, Q)

        assert np.allclose(fields.rho, single_fields.rho, rtol=1e-10)
        assert np.allclose(fields.jx, single_fields.jx, rtol=1e-10)
        assert np.allclose(fields.jy, single_fields.jy, rtol=1e-10)
        assert np.allclose(fields.jz, single_fields.jz, rtol=1e-10)


class TestBoundaryWrap:
    """Particles near grid boundaries deposit with periodic wrapping (INDEX2)."""

    def test_boundary_wrap(self) -> None:
        small_nx, small_ny = 8, 8
        dx_small = 1e-8
        dy_small = 1e-8
        n_guard = 3

        fields = make_fields2d(
            nx=small_nx, ny=small_ny, dx=dx_small, dy=dy_small, n_guard=n_guard
        )

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
            inv_gamma=np.array([1.0]),
            is_dead=np.array([False]),
        )

        current_deposition_cpu_2d([fields], [particles], 1, DT, Q)

        total_rho = np.sum(fields.rho)
        assert total_rho != 0.0, "boundary-wrapped particle should deposit charge"
        nonzero_cells = np.count_nonzero(fields.rho)
        assert nonzero_cells > 0, "deposition should affect some cells"


class TestMultiplePatches:
    """Multiple patches deposit independently without cross-contamination."""

    def test_multiple_patches(self) -> None:
        n_patches = 2
        nx, ny = 8, 8
        dx, dy = 1e-8, 1e-8
        n_guard = 3

        fields_list = []
        particles_list = []

        for _ip in range(n_patches):
            f = make_fields2d(nx=nx, ny=ny, dx=dx, dy=dy, n_guard=n_guard)
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

        current_deposition_cpu_2d(fields_list, particles_list, n_patches, DT, Q)

        for f in fields_list:
            assert np.sum(f.rho) != 0.0, "each patch should have deposited charge"

        single_f = make_fields2d(nx=nx, ny=ny, dx=dx, dy=dy, n_guard=n_guard)
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
        current_deposition_cpu_2d([single_f], [single_p], 1, DT, Q)

        for f in fields_list:
            assert np.allclose(f.rho, single_f.rho, rtol=1e-10)
            assert np.allclose(f.jx, single_f.jx, rtol=1e-10)
            assert np.allclose(f.jy, single_f.jy, rtol=1e-10)
            assert np.allclose(f.jz, single_f.jz, rtol=1e-10)


# ---------------------------------------------------------------------------
# New coverage: precision, performance, class integration, 3D
# ---------------------------------------------------------------------------

def test_precision_2d():
    """Verify charge/current conservation for a single particle."""
    nx = 6
    ny = 6
    npart = 1
    dx = dy = 1.0e-6
    x0 = -3 * dx
    y0 = -3 * dy
    dt = dx / c * 0.9
    q = e

    ne = 1e27
    w = ne * dx * dy

    ux = np.random.uniform(-10.0, 10.0, (npart,))
    uy = np.random.uniform(-10.0, 10.0, (npart,))
    uz = np.random.uniform(-10.0, 10.0, (npart,))
    inv_gamma = 1.0 / np.sqrt(1.0 + ux**2 + uy**2 + uz**2)

    particles = ParticlesBase(ipatch=0, rank=0)
    particles.initialize(npart)
    particles.x[:] = np.random.uniform(-dx, dx, (npart,))
    particles.y[:] = np.random.uniform(-dy, dy, (npart,))
    particles.ux[:] = ux
    particles.uy[:] = uy
    particles.uz[:] = uz
    particles.inv_gamma[:] = inv_gamma
    particles.w[:] = w
    particles.is_dead[:] = False

    fields = Fields2D(nx=nx, ny=ny, dx=dx, dy=dy, x0=x0, y0=y0, n_guard=3)

    current_deposition_cpu_2d([fields], [particles], 1, dt, q)

    vx = ux * inv_gamma * c
    vy = uy * inv_gamma * c
    vz = uz * inv_gamma * c

    assert abs(fields.jx.sum() - q * ne * vx) / abs(q * ne * vx) < 1e-10
    assert abs(fields.jy.sum() - q * ne * vy) / abs(q * ne * vy) < 1e-10
    assert abs(fields.jz.sum() - q * ne * vz) / abs(q * ne * vz) < 1e-10
    assert abs(fields.rho.sum() - ne * q) / abs(ne * q) < 1e-10


def test_cpu_deposition_2d():
    """Smoke-test the C extension with many patches/particles and measure timing."""
    npatch = 128
    nx = 100
    ny = 100
    npart = 100_000

    dx = 1e-6
    dy = 1e-6
    dt = dx / 2.0 / c
    q = e

    fields_list = []
    particles_list = []

    for ipatch in range(npatch):
        fields = Fields2D(nx=nx, ny=ny, dx=dx, dy=dy, x0=0.0, y0=0.0, n_guard=3)
        particles = ParticlesBase(ipatch=ipatch, rank=0)
        particles.initialize(npart)
        particles.x[:] = np.random.uniform(10 * dx, (nx - 10) * dx, npart)
        particles.y[:] = np.random.uniform(10 * dy, (ny - 10) * dy, npart)
        particles.ux[:] = 0.0
        particles.uy[:] = 0.0
        particles.uz[:] = 0.0
        particles.inv_gamma[:] = 1.0
        particles.is_dead[:] = False
        particles.w[:] = np.random.rand(npart)

        fields_list.append(fields)
        particles_list.append(particles)

    current_deposition_cpu_2d(fields_list, particles_list, npatch, dt, q)

    for fields in fields_list:
        assert not np.isnan(fields.rho).any()
        assert not np.isnan(fields.jx).any()
        assert not np.isnan(fields.jy).any()
        assert not np.isnan(fields.jz).any()

    tic = perf_counter_ns()
    current_deposition_cpu_2d(fields_list, particles_list, npatch, dt, q)
    toc = perf_counter_ns()

    nthreads = int(os.getenv("OMP_NUM_THREADS", os.cpu_count() or 1))
    nthreads = min(nthreads, npatch)
    print(
        f"current_deposit_2d {(toc - tic) / (npart * npatch) * nthreads:.0f} ns per particle"
    )


def test_current_deposition_class_2d():
    """Test CurrentDeposition2D integration with Patches."""
    nc = 1.74e27

    dx = 1e-6
    dy = 1e-6

    nx = 128
    ny = 128

    npatch_x = 4
    npatch_y = 4

    nx_per_patch = nx // npatch_x
    ny_per_patch = ny // npatch_y

    Lx = nx * dx
    Ly = ny * dy

    n_guard = 3
    patches = Patches(dimension=2)
    for j in range(npatch_y):
        for i in range(npatch_x):
            index = i + j * npatch_x
            p = Patch2D(
                rank=0,
                index=index,
                ipatch_x=i,
                ipatch_y=j,
                x0=i * Lx / npatch_x,
                y0=j * Ly / npatch_y,
                nx=nx_per_patch,
                ny=ny_per_patch,
                dx=dx,
                dy=dy,
            )
            f = Fields2D(
                nx=nx_per_patch,
                ny=ny_per_patch,
                dx=dx,
                dy=dy,
                x0=i * Lx / npatch_x,
                y0=j * Ly / npatch_y,
                n_guard=n_guard,
            )
            p.set_fields(f)

            if i > 0:
                p.set_neighbor_index(xmin=(i - 1) + j * npatch_x)
            if i < npatch_x - 1:
                p.set_neighbor_index(xmax=(i + 1) + j * npatch_x)
            if j > 0:
                p.set_neighbor_index(ymin=i + (j - 1) * npatch_x)
            if j < npatch_y - 1:
                p.set_neighbor_index(ymax=i + (j + 1) * npatch_x)

            patches.append(p)

    def density(x, y):
        return 2 * nc

    ele = Electron(density=density, ppc=8)
    ion = Proton(density=density, ppc=2)

    patches.add_species(ele)
    patches.add_species(ion)

    rand_gen = np.random.default_rng(42)
    patches.fill_particles(rand_gen)

    for patch in patches:
        p = patch.particles[0]
        p.ux[:] = np.random.normal(0, 1, p.npart)
        p.uy[:] = np.random.normal(0, 1, p.npart)
        p.uz[:] = np.random.normal(0, 1, p.npart)
        p.inv_gamma[:] = (1.0 + (p.ux**2 + p.uy**2 + p.uz**2)) ** -0.5

    current_dep = CurrentDeposition2D(patches)
    current_dep(0, dt=1e-15)

    current_deposition_cpu_2d(
        [patch.fields for patch in patches],
        [patch.particles[1] for patch in patches],
        npatch_x * npatch_y,
        1e-15,
        e,
    )

    for patch in patches:
        assert not np.isnan(patch.fields.rho).any()
        assert not np.isnan(patch.fields.jx).any()
        assert not np.isnan(patch.fields.jy).any()
        assert not np.isnan(patch.fields.jz).any()


def test_precision_3d():
    """Verify charge/current conservation for a single particle in 3D."""
    nx = ny = nz = 6
    npart = 1
    dx = dy = dz = 1.0e-6
    x0 = y0 = z0 = -3 * dx
    dt = dx / c * 0.9
    q = e

    ne = 1e27
    w = ne * dx * dy * dz

    ux = np.random.uniform(-10.0, 10.0, (npart,))
    uy = np.random.uniform(-10.0, 10.0, (npart,))
    uz = np.random.uniform(-10.0, 10.0, (npart,))
    inv_gamma = 1.0 / np.sqrt(1.0 + ux**2 + uy**2 + uz**2)

    particles = ParticlesBase(ipatch=0, rank=0)
    particles.initialize(npart)
    particles.x[:] = np.random.uniform(-dx, dx, (npart,))
    particles.y[:] = np.random.uniform(-dy, dy, (npart,))
    particles.z[:] = np.random.uniform(-dz, dz, (npart,))
    particles.ux[:] = ux
    particles.uy[:] = uy
    particles.uz[:] = uz
    particles.inv_gamma[:] = inv_gamma
    particles.w[:] = w
    particles.is_dead[:] = False

    fields = Fields3D(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, x0=x0, y0=y0, z0=z0, n_guard=3)

    current_deposition_cpu_3d([fields], [particles], 1, dt, q)

    vx = ux * inv_gamma * c
    vy = uy * inv_gamma * c
    vz = uz * inv_gamma * c

    assert abs(fields.jx.sum() - q * ne * vx) / abs(q * ne * vx) < 1e-10
    assert abs(fields.jy.sum() - q * ne * vy) / abs(q * ne * vy) < 1e-10
    assert abs(fields.jz.sum() - q * ne * vz) / abs(q * ne * vz) < 1e-10
    assert abs(fields.rho.sum() - ne * q) / abs(ne * q) < 1e-10


def test_cpu_deposition_3d():
    """Smoke-test the 3D C extension with many patches/particles and measure timing."""
    npatch = 16
    nx = ny = nz = 32
    npart = 10_000

    dx = dy = dz = 1e-6
    dt = dx / 2.0 / c
    q = e

    fields_list = []
    particles_list = []

    for ipatch in range(npatch):
        fields = Fields3D(
            nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, x0=0.0, y0=0.0, z0=0.0, n_guard=3
        )
        particles = ParticlesBase(ipatch=ipatch, rank=0)
        particles.initialize(npart)
        particles.x[:] = np.random.uniform(10 * dx, (nx - 10) * dx, npart)
        particles.y[:] = np.random.uniform(10 * dy, (ny - 10) * dy, npart)
        particles.z[:] = np.random.uniform(10 * dz, (nz - 10) * dz, npart)
        particles.ux[:] = 0.0
        particles.uy[:] = 0.0
        particles.uz[:] = 0.0
        particles.inv_gamma[:] = 1.0
        particles.is_dead[:] = False
        particles.w[:] = np.random.rand(npart)

        fields_list.append(fields)
        particles_list.append(particles)

    current_deposition_cpu_3d(fields_list, particles_list, npatch, dt, q)

    for fields in fields_list:
        assert not np.isnan(fields.rho).any()
        assert not np.isnan(fields.jx).any()
        assert not np.isnan(fields.jy).any()
        assert not np.isnan(fields.jz).any()

    tic = perf_counter_ns()
    current_deposition_cpu_3d(fields_list, particles_list, npatch, dt, q)
    toc = perf_counter_ns()

    nthreads = int(os.getenv("OMP_NUM_THREADS", os.cpu_count() or 1))
    nthreads = min(nthreads, npatch)
    print(
        f"current_deposit_3d {(toc - tic) / (npart * npatch) * nthreads:.0f} ns per particle"
    )


def test_current_deposition_class_3d():
    """Test CurrentDeposition3D integration with Patches."""
    nc = 1.74e27

    dx = dy = dz = 1e-6

    nx = ny = nz = 64

    npatch_x = npatch_y = npatch_z = 2

    nx_per_patch = nx // npatch_x
    ny_per_patch = ny // npatch_y
    nz_per_patch = nz // npatch_z

    Lx = nx * dx
    Ly = ny * dy
    Lz = nz * dz

    n_guard = 3
    patches = Patches(dimension=3)
    for k in range(npatch_z):
        for j in range(npatch_y):
            for i in range(npatch_x):
                index = i + j * npatch_x + k * npatch_x * npatch_y
                p = Patch3D(
                    rank=0,
                    index=index,
                    ipatch_x=i,
                    ipatch_y=j,
                    ipatch_z=k,
                    x0=i * Lx / npatch_x,
                    y0=j * Ly / npatch_y,
                    z0=k * Lz / npatch_z,
                    nx=nx_per_patch,
                    ny=ny_per_patch,
                    nz=nz_per_patch,
                    dx=dx,
                    dy=dy,
                    dz=dz,
                )
                f = Fields3D(
                    nx=nx_per_patch,
                    ny=ny_per_patch,
                    nz=nz_per_patch,
                    dx=dx,
                    dy=dy,
                    dz=dz,
                    x0=i * Lx / npatch_x,
                    y0=j * Ly / npatch_y,
                    z0=k * Lz / npatch_z,
                    n_guard=n_guard,
                )
                p.set_fields(f)

                if i > 0:
                    p.set_neighbor_index(xmin=(i - 1) + j * npatch_x + k * npatch_x * npatch_y)
                if i < npatch_x - 1:
                    p.set_neighbor_index(xmax=(i + 1) + j * npatch_x + k * npatch_x * npatch_y)
                if j > 0:
                    p.set_neighbor_index(ymin=i + (j - 1) * npatch_x + k * npatch_x * npatch_y)
                if j < npatch_y - 1:
                    p.set_neighbor_index(ymax=i + (j + 1) * npatch_x + k * npatch_x * npatch_y)
                if k > 0:
                    p.set_neighbor_index(zmin=i + j * npatch_x + (k - 1) * npatch_x * npatch_y)
                if k < npatch_z - 1:
                    p.set_neighbor_index(zmax=i + j * npatch_x + (k + 1) * npatch_x * npatch_y)

                patches.append(p)

    def density(x, y, z):
        return 2 * nc

    ele = Electron(density=density, ppc=4)
    ion = Proton(density=density, ppc=1)

    patches.add_species(ele)
    patches.add_species(ion)

    rand_gen = np.random.default_rng(42)
    patches.fill_particles(rand_gen)

    for patch in patches:
        p = patch.particles[0]
        p.ux[:] = np.random.normal(0, 1, p.npart)
        p.uy[:] = np.random.normal(0, 1, p.npart)
        p.uz[:] = np.random.normal(0, 1, p.npart)
        p.inv_gamma[:] = (1.0 + (p.ux**2 + p.uy**2 + p.uz**2)) ** -0.5

    current_dep = CurrentDeposition3D(patches)
    current_dep(0, dt=1e-15)

    current_deposition_cpu_3d(
        [patch.fields for patch in patches],
        [patch.particles[1] for patch in patches],
        npatch_x * npatch_y * npatch_z,
        1e-15,
        e,
    )

    for patch in patches:
        assert not np.isnan(patch.fields.rho).any()
        assert not np.isnan(patch.fields.jx).any()
        assert not np.isnan(patch.fields.jy).any()
        assert not np.isnan(patch.fields.jz).any()
