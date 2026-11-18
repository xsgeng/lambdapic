"""SimulationConfig validation: PML boundaries need >= 2 patches per direction.

The PML of a direction lives on its two edge patches (ipatch 0 and
npatch-1); with a single patch there both PMLs collapse onto it and the
run fails deep inside Simulation.run with no actionable diagnostics
(observed as an aborted MPI job with no traceback). Construction must
fail fast instead. Auto patching already enforces the minimum
(``npatch_*_min = 2``), so only manual values need the guard.
"""
import pytest

from lambdapic import Simulation, Simulation3D

PERIODIC_2D = {
    'xmin': 'periodic', 'xmax': 'periodic',
    'ymin': 'periodic', 'ymax': 'periodic',
}
PERIODIC_3D = {
    **PERIODIC_2D,
    'zmin': 'periodic', 'zmax': 'periodic',
}


def test_pml_with_single_patch_in_y_rejected():
    """The exact combo that failed at runtime: 2x1 patches, PML on ymin/ymax."""
    with pytest.raises(ValueError, match=r"npatch_y \(1\) must be >= 2"):
        Simulation(
            nx=32, ny=32, dx=1e-8, dy=1e-8,
            npatch_x=2, npatch_y=1, dt_cfl=0.95,
        )


def test_pml_with_single_patch_in_x_rejected():
    with pytest.raises(ValueError, match=r"npatch_x \(1\) must be >= 2"):
        Simulation(
            nx=32, ny=32, dx=1e-8, dy=1e-8,
            npatch_x=1, npatch_y=2, dt_cfl=0.95,
        )


def test_pml_on_one_side_only_still_rejected():
    """Either PML side of the direction triggers the guard, not just both."""
    with pytest.raises(ValueError, match=r"npatch_y \(1\) must be >= 2"):
        Simulation(
            nx=32, ny=32, dx=1e-8, dy=1e-8,
            npatch_x=2, npatch_y=1, dt_cfl=0.95,
            boundary_conditions={
                'xmin': 'periodic', 'xmax': 'periodic',
                'ymin': 'pml', 'ymax': 'periodic',
            },
        )


def test_single_patch_grid_with_periodic_accepted():
    """A 1x1 patch grid is valid as long as every boundary is periodic."""
    sim = Simulation(
        nx=32, ny=32, dx=1e-8, dy=1e-8,
        npatch_x=1, npatch_y=1, dt_cfl=0.95,
        boundary_conditions=PERIODIC_2D,
    )
    assert (sim.npatch_x, sim.npatch_y) == (1, 1)


def test_pml_with_two_patches_per_direction_accepted():
    """Default all-PML boundaries with 2x2 patches construct normally."""
    sim = Simulation(
        nx=32, ny=32, dx=1e-8, dy=1e-8,
        npatch_x=2, npatch_y=2, dt_cfl=0.95,
    )
    assert sim.boundary_conditions['ymin'] == 'pml'


def test_pml_with_single_patch_in_z_rejected_3d():
    with pytest.raises(ValueError, match=r"npatch_z \(1\) must be >= 2"):
        Simulation3D(
            nx=16, ny=16, nz=16, dx=1e-8, dy=1e-8, dz=1e-8,
            npatch_x=2, npatch_y=2, npatch_z=1, dt_cfl=0.95,
        )


def test_single_patch_grid_with_periodic_accepted_3d():
    sim = Simulation3D(
        nx=16, ny=16, nz=16, dx=1e-8, dy=1e-8, dz=1e-8,
        npatch_x=1, npatch_y=1, npatch_z=1, dt_cfl=0.95,
        boundary_conditions=PERIODIC_3D,
    )
    assert (sim.npatch_x, sim.npatch_y, sim.npatch_z) == (1, 1, 1)
