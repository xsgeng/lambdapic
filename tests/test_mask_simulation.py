"""Tests for sparse (mask-driven) patch grids with holes."""

import pytest
import numpy as np
from scipy.constants import c, e, epsilon_0, m_e, pi

from lambdapic import Electron
from lambdapic.core.patch.patch import Patches, Patch2D, Boundary2D
from lambdapic.core.boundary.utils import has_pml
from lambdapic._mask_simulation import _MaskSimulation


def ring_mask(r_inner: float, r_outer: float, cx: float = 0.0, cy: float = 0.0):
    """Return a mask function for an annular (ring) domain."""
    def _mask(x: float, y: float) -> bool:
        r = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        return r_inner <= r <= r_outer
    return _mask


def test_sparse_neighbor_index_2d():
    """Test that init_rect_neighbor_index_2d handles sparse grids (holes) correctly."""
    patches = Patches(dimension=2)

    # Create a 4x4 grid but skip center 2x2 patches (indices 5,6,9,10)
    # to simulate a mask that excludes them
    npatch_x, npatch_y = 4, 4
    nx, ny = 10, 10
    dx, dy = 1.0, 1.0

    # excluded center patches: (1,1)=5, (2,1)=6, (1,2)=9, (2,2)=10
    excluded = {(1, 1), (2, 1), (1, 2), (2, 2)}

    index = 0
    for j in range(npatch_y):
        for i in range(npatch_x):
            if (i, j) in excluded:
                continue
            p = Patch2D(
                rank=None, index=index,
                ipatch_x=i, ipatch_y=j,
                x0=i * dx * nx, y0=j * dy * ny,
                nx=nx, ny=ny, dx=dx, dy=dy,
            )
            patches.append(p)
            index += 1

    # Build patch_index_map for sparse grid
    patch_index_map = {(p.ipatch_x, p.ipatch_y): p.index for p in patches.patches}

    # Call init with sparse grid
    patches.init_rect_neighbor_index_2d(
        npatch_x=npatch_x,
        npatch_y=npatch_y,
        boundary_conditions={
            "xmin": "pml",
            "xmax": "pml",
            "ymin": "pml",
            "ymax": "pml",
        },
        patch_index_map=patch_index_map,
    )

    # Test 1: Patch at (0,0) - corner, should have -1 on XMIN and YMIN
    p00 = next(p for p in patches.patches if p.ipatch_x == 0 and p.ipatch_y == 0)
    assert p00.neighbor_index[Boundary2D.XMIN] == -1
    assert p00.neighbor_index[Boundary2D.YMIN] == -1
    assert p00.neighbor_index[Boundary2D.XMAX] >= 0  # neighbor at (1,0)
    assert p00.neighbor_index[Boundary2D.YMAX] >= 0  # neighbor at (0,1)

    # Test 2: Patch at (1,0) - adjacent to hole on YMAX (1,1 is hole)
    p10 = next(p for p in patches.patches if p.ipatch_x == 1 and p.ipatch_y == 0)
    assert p10.neighbor_index[Boundary2D.YMAX] == -1  # (1,1) is hole
    assert p10.neighbor_index[Boundary2D.XMAX] >= 0  # (2,0) exists

    # Test 3: Patch at (3,3) - outer corner, should have -1 on XMAX and YMAX
    p33 = next(p for p in patches.patches if p.ipatch_x == 3 and p.ipatch_y == 3)
    assert p33.neighbor_index[Boundary2D.XMAX] == -1
    assert p33.neighbor_index[Boundary2D.YMAX] == -1

    # Test 4: Diagonal neighbor into hole should be -1
    # Patch at (0,1) has diagonal (1,2) which is hole
    p01 = next(p for p in patches.patches if p.ipatch_x == 0 and p.ipatch_y == 1)
    assert p01.neighbor_index[Boundary2D.XMAXYMAX] == -1  # (1,2) is hole


def test_rect_neighbor_index_2d_regression():
    """Verify backward compatibility: full 4x4 grid still works correctly."""
    patches = Patches(dimension=2)
    npatch_x, npatch_y = 4, 4
    nx, ny = 10, 10
    dx, dy = 1.0, 1.0

    for j in range(npatch_y):
        for i in range(npatch_x):
            index = i + j * npatch_x
            p = Patch2D(
                rank=None, index=index,
                ipatch_x=i, ipatch_y=j,
                x0=i * dx * nx, y0=j * dy * ny,
                nx=nx, ny=ny, dx=dx, dy=dy,
            )
            patches.append(p)

    patches.init_rect_neighbor_index_2d(
        npatch_x=npatch_x,
        npatch_y=npatch_y,
        boundary_conditions={
            "xmin": "pml",
            "xmax": "pml",
            "ymin": "pml",
            "ymax": "pml",
        },
    )

    # Corner patch should have 2 face neighbors and 1 corner neighbor
    p00 = patches.patches[0]
    assert p00.neighbor_index[Boundary2D.XMIN] == -1
    assert p00.neighbor_index[Boundary2D.YMIN] == -1
    assert p00.neighbor_index[Boundary2D.XMAX] == 1  # (1,0)
    assert p00.neighbor_index[Boundary2D.YMAX] == 4  # (0,1)
    assert p00.neighbor_index[Boundary2D.XMAXYMAX] == 5  # (1,1)
    assert p00.neighbor_index[Boundary2D.XMINYMIN] == -1
    assert p00.neighbor_index[Boundary2D.XMAXYMIN] == -1
    assert p00.neighbor_index[Boundary2D.XMINYMAX] == -1

    # Interior patch should have all neighbors
    p11 = patches.patches[5]  # (1,1)
    assert p11.neighbor_index[Boundary2D.XMIN] == 4  # (0,1)
    assert p11.neighbor_index[Boundary2D.XMAX] == 6  # (2,1)
    assert p11.neighbor_index[Boundary2D.YMIN] == 1  # (1,0)
    assert p11.neighbor_index[Boundary2D.YMAX] == 9  # (1,2)
    assert p11.neighbor_index[Boundary2D.XMINYMIN] == 0  # (0,0)
    assert p11.neighbor_index[Boundary2D.XMAXYMIN] == 2  # (2,0)
    assert p11.neighbor_index[Boundary2D.XMINYMAX] == 8  # (0,2)
    assert p11.neighbor_index[Boundary2D.XMAXYMAX] == 10  # (2,2)


def _ring_sim():
    """Return a _MaskSimulation with a centered annular mask."""
    nx = ny = 64
    dx = dy = 1e-8
    npatch_x = npatch_y = 8
    Lx = nx * dx
    Ly = ny * dy
    mask = ring_mask(
        r_inner=0.2 * Lx,
        r_outer=0.45 * Lx,
        cx=Lx / 2,
        cy=Ly / 2,
    )
    return _MaskSimulation(
        nx=nx, ny=ny, dx=dx, dy=dy,
        npatch_x=npatch_x, npatch_y=npatch_y,
        mask=mask,
    )


def test_mask_simulation_create_patches():
    """Patches are created only where the mask is True at patch centers."""
    sim = _ring_sim()
    patches = sim.create_patches()

    expected = 0
    for j in range(sim.npatch_y):
        for i in range(sim.npatch_x):
            xc = (i + 0.5) * sim.Lx / sim.npatch_x
            yc = (j + 0.5) * sim.Ly / sim.npatch_y
            if sim.mask(xc, yc):
                expected += 1

    assert patches.npatches == expected
    assert patches.indices == list(range(patches.npatches))
    for p in patches.patches:
        assert 0 <= p.ipatch_x < sim.npatch_x
        assert 0 <= p.ipatch_y < sim.npatch_y


def test_mask_simulation_neighbor_indices():
    """Face neighbor indices are >=0 exactly when the neighbor patch exists."""
    sim = _ring_sim()
    patches = sim.create_patches()
    selected = {(p.ipatch_x, p.ipatch_y) for p in patches.patches}
    face_dirs = {
        Boundary2D.XMIN: (-1, 0),
        Boundary2D.XMAX: (1, 0),
        Boundary2D.YMIN: (0, -1),
        Boundary2D.YMAX: (0, 1),
    }

    for p in patches.patches:
        for bound, (di, dj) in face_dirs.items():
            ni, nj = p.ipatch_x + di, p.ipatch_y + dj
            neighbor_exists = (ni, nj) in selected
            assert (p.neighbor_index[bound] >= 0) == neighbor_exists

    edge_patches = [
        p for p in patches.patches
        if any(p.neighbor_index[b] < 0 for b in face_dirs)
    ]
    assert len(edge_patches) > 0


def test_mask_simulation_pml_attachment():
    """PML is attached exactly on faces without neighbors."""
    sim = _ring_sim()
    sim.initialize()
    faces = (
        Boundary2D.XMIN,
        Boundary2D.XMAX,
        Boundary2D.YMIN,
        Boundary2D.YMAX,
    )

    for p in sim.patches:
        missing_count = sum(1 for b in faces if p.neighbor_index[b] < 0)
        assert len(p.pml_boundary) == missing_count
        assert len(p.pml_boundary) <= 2
        if len(p.pml_boundary) == 2:
            x_count = sum(has_pml(p.pml_boundary, b) for b in ("xmin", "xmax"))
            y_count = sum(has_pml(p.pml_boundary, b) for b in ("ymin", "ymax"))
            assert x_count + y_count == 2


def test_ring_mask_helper():
    """The ring_mask helper returns True only inside the annulus."""
    m = ring_mask(1.0, 2.0, 0.0, 0.0)
    assert m(1.5, 0.0) is True
    assert m(0.5, 0.0) is False
    assert m(2.5, 0.0) is False
    assert m(0.0, 1.5) is True


def test_empty_mask_raises():
    """A mask that excludes every patch must raise an assertion."""
    sim = _MaskSimulation(
        nx=64, ny=64, dx=1e-8, dy=1e-8,
        npatch_x=8, npatch_y=8,
        mask=lambda x, y: False,
    )
    with pytest.raises(AssertionError, match="mask produced no patches"):
        sim.create_patches()


def test_periodic_boundary_ignored():
    """Periodic wrapping is disabled; mask class forces all-open PML neighbors."""
    nx = ny = 64
    dx = dy = 1e-8
    npatch_x = npatch_y = 8
    Lx = nx * dx
    Ly = ny * dy
    sim = _MaskSimulation(
        nx=nx, ny=ny, dx=dx, dy=dy,
        npatch_x=npatch_x, npatch_y=npatch_y,
        boundary_conditions={
            "xmin": "periodic",
            "xmax": "periodic",
            "ymin": "periodic",
            "ymax": "periodic",
        },
        mask=lambda x, y: x < Lx / 2,
    )
    patches = sim.create_patches()
    right_edge_patches = [p for p in patches.patches if p.ipatch_x == npatch_x // 2 - 1]
    assert len(right_edge_patches) > 0
    for p in right_edge_patches:
        assert p.neighbor_index[Boundary2D.XMAX] == -1


def test_ring_field_damping():
    """Field-only test: PML should damp energy near the inner ring boundary."""
    um = 1e-6
    l0 = 0.8 * um
    nx = ny = 160
    dx = dy = l0 / 20
    npatch_x = npatch_y = 16
    Lx = nx * dx
    Ly = ny * dy
    cpml_thickness = 6

    mask = ring_mask(
        r_inner=0.2 * Lx,
        r_outer=0.45 * Lx,
        cx=Lx / 2,
        cy=Ly / 2,
    )

    sim = _MaskSimulation(
        nx=nx, ny=ny, dx=dx, dy=dy,
        npatch_x=npatch_x, npatch_y=npatch_y,
        cpml_thickness=cpml_thickness,
        mask=mask,
    )

    ele = Electron(density=lambda x, y: 0.0, ppc=1)
    sim.add_species([ele])
    sim.initialize()

    r_inj = 0.2 * Lx + cpml_thickness * dx + 2 * dx
    angle = pi / 4
    xc0 = Lx / 2 + r_inj * np.cos(angle)
    yc0 = Ly / 2 + r_inj * np.sin(angle)
    sigma = 3 * dx
    A = 1e12

    for p in sim.patches:
        xaxis = p.fields.xaxis
        yaxis = p.fields.yaxis
        x, y = np.meshgrid(xaxis, yaxis, indexing='ij')
        p.fields.ey[:, :] = A * np.exp(-((x - xc0)**2 + (y - yc0)**2) / sigma**2)
        p.fields.bz[:, :] = p.fields.ey[:, :] / c

    def pml_energy():
        total = 0.0
        for p in sim.patches:
            ng = p.fields.n_guard
            nx_p = p.nx
            ny_p = p.ny
            ex = p.fields.ex[ng:ng + nx_p, ng:ng + ny_p]
            ey = p.fields.ey[ng:ng + nx_p, ng:ng + ny_p]
            ez = p.fields.ez[ng:ng + nx_p, ng:ng + ny_p]
            bx = p.fields.bx[ng:ng + nx_p, ng:ng + ny_p]
            by = p.fields.by[ng:ng + nx_p, ng:ng + ny_p]
            bz = p.fields.bz[ng:ng + nx_p, ng:ng + ny_p]
            in_pml = np.zeros((nx_p, ny_p), dtype=bool)
            if p.neighbor_index[Boundary2D.XMIN] < 0:
                in_pml[:cpml_thickness, :] = True
            if p.neighbor_index[Boundary2D.XMAX] < 0:
                in_pml[nx_p - cpml_thickness:, :] = True
            if p.neighbor_index[Boundary2D.YMIN] < 0:
                in_pml[:, :cpml_thickness] = True
            if p.neighbor_index[Boundary2D.YMAX] < 0:
                in_pml[:, ny_p - cpml_thickness:] = True
            if np.any(in_pml):
                energy = ex**2 + ey**2 + ez**2 + c**2 * (bx**2 + by**2 + bz**2)
                total += np.sum(energy[in_pml])
        return total

    E0 = pml_energy()
    sim.run(nsteps=50)
    E1 = pml_energy()

    assert E1 < E0 * 0.99, f"PML did not damp energy: E0={E0}, E1={E1}"
    assert sim.itime == 50


def test_ring_run_no_crash():
    """Full simulation with particles: verify no crash and finite arrays."""
    um = 1e-6
    l0 = 0.8 * um
    omega0 = 2 * pi * c / l0
    nc = epsilon_0 * m_e * omega0**2 / e**2
    nx = ny = 160
    dx = dy = l0 / 20
    npatch_x = npatch_y = 16
    Lx = nx * dx
    Ly = ny * dy

    r_inner = 0.2 * Lx
    r_outer = 0.45 * Lx

    mask = ring_mask(r_inner, r_outer, cx=Lx / 2, cy=Ly / 2)

    sim = _MaskSimulation(
        nx=nx, ny=ny, dx=dx, dy=dy,
        npatch_x=npatch_x, npatch_y=npatch_y,
        mask=mask,
    )

    def density(n0):
        def _density(x, y):
            r = ((x - Lx / 2)**2 + (y - Ly / 2)**2) ** 0.5
            if r_inner * 1.2 < r < r_outer * 0.8:
                return n0
            return 0.0
        return _density

    ele = Electron(density=density(0.01 * nc), ppc=2)
    sim.add_species([ele])
    sim.initialize()
    sim.run(nsteps=10)

    assert sim.itime == 10
    for p in sim.patches:
        particles = p.particles[0]
        alive = ~particles.is_dead
        assert np.all(np.isfinite(particles.x[alive]))
        assert np.all(np.isfinite(particles.y[alive]))
        assert np.all(np.isfinite(particles.ux[alive]))
        assert np.all(np.isfinite(particles.uy[alive]))


def test_ring_too_thin_raises():
    """A ring thinner than 2 patches violates the PML constraint."""
    nx = ny = 64
    dx = dy = 1e-8
    npatch_x = npatch_y = 8
    Lx = nx * dx
    Ly = ny * dy

    patch_width = Lx / npatch_x
    mask = ring_mask(
        r_inner=0.25 * Lx,
        r_outer=0.25 * Lx + 0.5 * patch_width,
        cx=Lx / 2,
        cy=Ly / 2,
    )

    sim = _MaskSimulation(
        nx=nx, ny=ny, dx=dx, dy=dy,
        npatch_x=npatch_x, npatch_y=npatch_y,
        mask=mask,
    )

    ele = Electron(density=lambda x, y: 0.01, ppc=1)
    sim.add_species([ele])

    with pytest.raises(AssertionError):
        sim.initialize()


def test_masked_regions_are_nan():
    from pathlib import Path
    import tempfile
    import h5py

    from lambdapic.callback.utils import get_fields
    from lambdapic.callback.hdf5 import SaveFieldsToHDF5

    um = 1e-6
    l0 = 0.8 * um
    nx = ny = 64
    dx = dy = l0 / 20
    npatch_x = npatch_y = 8
    Lx = nx * dx
    Ly = ny * dy

    mask = ring_mask(
        r_inner=0.2 * Lx,
        r_outer=0.45 * Lx,
        cx=Lx / 2,
        cy=Ly / 2,
    )

    sim = _MaskSimulation(
        nx=nx, ny=ny, dx=dx, dy=dy,
        npatch_x=npatch_x, npatch_y=npatch_y,
        mask=mask,
    )

    ele = Electron(density=lambda x, y: 0.0, ppc=1)
    sim.add_species([ele])
    sim.initialize()

    assert hasattr(sim, 'domain_mask')
    assert not sim.domain_mask.all()
    assert sim.domain_mask.any()

    fields = get_fields(sim, ['ex'])
    ex = fields[0]
    assert ex is not None
    assert np.all(np.isnan(ex[~sim.domain_mask]))
    assert np.all(np.isfinite(ex[sim.domain_mask]))

    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_cb = SaveFieldsToHDF5(prefix=tmpdir, interval=1, components=['ex'])
        hdf5_cb(sim)

        filename = Path(tmpdir) / f"{sim.itime:06d}.h5"
        with h5py.File(filename, 'r') as f:
            h5_ex = f['ex'][()]
        assert np.all(np.isnan(h5_ex[~sim.domain_mask]))
        assert np.all(np.isfinite(h5_ex[sim.domain_mask]))
