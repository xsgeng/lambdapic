"""Tests for sparse (mask-driven) patch grids with holes."""

import pytest
import numpy as np
from lambdapic.core.patch.patch import Patches, Patch2D, Boundary2D


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
