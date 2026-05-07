import pytest
import h5py
import os
import numpy as np
from lambdapic.simulation import Simulation, Simulation3D
from lambdapic.core.species import Electron
from lambdapic.callback.hdf5 import (
    SaveFieldsToHDF5,
    SaveSpeciesDensityToHDF5,
    SaveParticlesToHDF5
)
from lambdapic.callback.utils import ExtractSpeciesDensity

def test_hdf5_field_callback_2d(tmp_path):
    """Test saving field data to HDF5 in 2D simulation."""
    # Setup minimal 2D simulation
    sim = Simulation(
        nx=32,
        ny=32,
        dx=0.1,
        dy=0.1,
        npatch_x=2,
        npatch_y=2,
        dt_cfl=0.95
    )
    
    # Create and run with field callback
    output_dir = tmp_path / "field_output"
    callback = SaveFieldsToHDF5(
        prefix=str(output_dir),
        interval=10,
        components=['ex', 'ey']
    )
    
    sim.run(21, callbacks=[callback])
    
    # Verify output files
    assert os.path.exists(output_dir / "000000.h5")
    assert os.path.exists(output_dir / "000010.h5")
    assert os.path.exists(output_dir / "000020.h5")
    
    # Verify file contents
    with h5py.File(output_dir / "000000.h5", 'r') as f:
        assert 'ex' in f
        assert 'ey' in f
        assert f.attrs['nx'] == 32
        assert f.attrs['ny'] == 32

def test_hdf5_species_callback_2d(tmp_path):
    """Test saving species density to HDF5 in 2D simulation."""
    sim = Simulation(
        nx=32,
        ny=32,
        dx=0.1,
        dy=0.1,
        npatch_x=2,
        npatch_y=2,
        dt_cfl=0.95
    )
    
    # Add species
    electrons = Electron(
        name="electrons",
        density=lambda x, y: 1.0,
        ppc=4,
    )
    sim.add_species([electrons])
    
    # Create and run with density callback
    output_dir = tmp_path / "density_output"
    callback = SaveSpeciesDensityToHDF5(
        species=electrons,
        prefix=str(output_dir),
        interval=10
    )
    
    sim.run(21, callbacks=[callback])
    
    # Verify output files
    assert os.path.exists(output_dir / "electrons_000000.h5")
    assert os.path.exists(output_dir / "electrons_000010.h5")
    assert os.path.exists(output_dir / "electrons_000020.h5")
    
    # Verify file contents
    with h5py.File(output_dir / "electrons_000000.h5", 'r') as f:
        assert 'density' in f
        assert f.attrs['species'] == 'electrons'

# =============================================================================
# 2D Slice tests for SaveSpeciesDensityToHDF5
# =============================================================================

def test_hdf5_density_callback_2d_slice_none(tmp_path):
    """Test SaveSpeciesDensityToHDF5 with slice=None (full domain) in 2D."""
    sim = Simulation(nx=32, ny=32, dx=0.1, dy=0.1, npatch_x=2, npatch_y=2, dt_cfl=0.95)
    electrons = Electron(name="electrons", density=lambda x, y: 1.0, ppc=4)
    sim.add_species([electrons])

    cb = SaveSpeciesDensityToHDF5(species=electrons, prefix=str(tmp_path / 'out'), interval=1, slice=None)
    sim.run(1, callbacks=[cb])

    with h5py.File(tmp_path / 'out' / f'electrons_000000.h5', 'r') as f:
        dset = f['density']
        assert dset.shape == (32, 32)
        assert dset[:].size == 32 * 32
        assert 'slice' not in f.attrs
        assert f.attrs['species'] == 'electrons'


def test_hdf5_density_callback_2d_slice_int(tmp_path):
    """Test SaveSpeciesDensityToHDF5 with an integer slice along y in 2D."""
    sim = Simulation(nx=32, ny=32, dx=0.1, dy=0.1, npatch_x=2, npatch_y=2, dt_cfl=0.95)
    electrons = Electron(name="electrons", density=lambda x, y: 1.0, ppc=4)
    sim.add_species([electrons])

    cb = SaveSpeciesDensityToHDF5(species=electrons, prefix=str(tmp_path / 'out'), interval=1, slice=np.s_[:, 5])
    sim.run(1, callbacks=[cb])

    with h5py.File(tmp_path / 'out' / 'electrons_000000.h5', 'r') as f:
        assert f['density'].shape == (32, 1)
        assert f['density'][:].size == 32
        assert f.attrs['slice'] == '[:, 5]'


def test_hdf5_density_callback_2d_slice_stepped(tmp_path):
    """Test SaveSpeciesDensityToHDF5 with stepped slicing in 2D."""
    sim = Simulation(nx=32, ny=32, dx=0.1, dy=0.1, npatch_x=2, npatch_y=2, dt_cfl=0.95)
    electrons = Electron(name="electrons", density=lambda x, y: 1.0, ppc=4)
    sim.add_species([electrons])

    cb = SaveSpeciesDensityToHDF5(species=electrons, prefix=str(tmp_path / 'out'), interval=1, slice=np.s_[::2, ::3])
    sim.run(1, callbacks=[cb])

    with h5py.File(tmp_path / 'out' / 'electrons_000000.h5', 'r') as f:
        assert f['density'].shape == (16, 11)
        assert f['density'][:].size == 176
        assert f.attrs['slice'] == '[::2, ::3]'


# =============================================================================
# 3D Slice tests for SaveSpeciesDensityToHDF5
# =============================================================================

def test_hdf5_density_callback_3d_slice_none(tmp_path):
    """Test SaveSpeciesDensityToHDF5 with slice=None (full domain) in 3D."""
    sim = Simulation3D(nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, npatch_x=2, npatch_y=2, npatch_z=2)
    electrons = Electron(name="electrons", density=lambda x, y, z: 1.0, ppc=4)
    sim.add_species([electrons])

    cb = SaveSpeciesDensityToHDF5(species=electrons, prefix=str(tmp_path / 'out3d'), interval=1, slice=None)
    sim.run(1, callbacks=[cb])

    with h5py.File(tmp_path / 'out3d' / 'electrons_000000.h5', 'r') as f:
        assert f['density'].shape == (32, 32, 32)
        assert f['density'][:].size == 32 * 32 * 32
        assert 'slice' not in f.attrs


def test_hdf5_density_callback_3d_slice_plane(tmp_path):
    """Test SaveSpeciesDensityToHDF5 with a plane slice in 3D (single z index)."""
    sim = Simulation3D(nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, npatch_x=2, npatch_y=2, npatch_z=2)
    electrons = Electron(name="electrons", density=lambda x, y, z: 1.0, ppc=4)
    sim.add_species([electrons])

    cb = SaveSpeciesDensityToHDF5(species=electrons, prefix=str(tmp_path / 'out3d'), interval=1, slice=np.s_[:, :, 10])
    sim.run(1, callbacks=[cb])

    with h5py.File(tmp_path / 'out3d' / 'electrons_000000.h5', 'r') as f:
        assert f['density'].shape == (32, 32, 1)
        assert f['density'][:].size == 32 * 32
        assert f.attrs['slice'] == '[:, :, 10]'


def test_hdf5_density_callback_3d_slice_stepped(tmp_path):
    """Test SaveSpeciesDensityToHDF5 with stepped slicing in 3D."""
    sim = Simulation3D(nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, npatch_x=2, npatch_y=2, npatch_z=2)
    electrons = Electron(name="electrons", density=lambda x, y, z: 1.0, ppc=4)
    sim.add_species([electrons])

    cb = SaveSpeciesDensityToHDF5(species=electrons, prefix=str(tmp_path / 'out3d'), interval=1, slice=np.s_[::2, ::2, ::5])
    sim.run(1, callbacks=[cb])

    with h5py.File(tmp_path / 'out3d' / 'electrons_000000.h5', 'r') as f:
        assert f['density'].shape == (16, 16, 7)
        assert f['density'][:].size == 1792
        assert f.attrs['slice'] == '[::2, ::2, ::5]'


def test_hdf5_particles_callback_2d(tmp_path):
    """Test saving particle data to HDF5 in 2D simulation."""
    sim = Simulation(
        nx=32,
        ny=32,
        dx=0.1,
        dy=0.1,
        npatch_x=2,
        npatch_y=2,
        dt_cfl=0.95,
        boundary_conditions={'xmin': 'periodic', 'xmax': 'periodic', 'ymin': 'periodic', 'ymax': 'periodic'}
    )
    
    # Add electron species
    electrons = Electron(
        name="electrons",
        density=lambda x, y: 1.0,
        ppc=4,
    )
    sim.add_species([electrons])
    
    # Create and run with particle callback
    output_dir = tmp_path / "particles_output"
    callback = SaveParticlesToHDF5(
        species=electrons,
        prefix=str(output_dir),
        interval=10,
        attrs=['x', 'y', 'w']
    )
    
    sim.run(21, callbacks=[callback])
    
    # Verify output files
    assert os.path.exists(output_dir / "electrons_particles_000000.h5")
    assert os.path.exists(output_dir / "electrons_particles_000010.h5")
    assert os.path.exists(output_dir / "electrons_particles_000020.h5")
    
    # Verify file contents and data length
    with h5py.File(output_dir / "electrons_particles_000000.h5", 'r') as f:
        assert 'x' in f
        assert 'y' in f
        assert 'w' in f
        assert 'id' in f
        assert f.attrs['time'] == 0.0
        assert f.attrs['itime'] == 0
        
        # Check all datasets have same length
        x_len = len(f['x'])
        y_len = len(f['y'])
        w_len = len(f['w'])
        id_len = len(f['id'])
        assert x_len == y_len == w_len == id_len
        
        # Verify length matches expected particles per cell
        expected_particles = sim.nx * sim.ny * electrons.ppc
        assert x_len == expected_particles


# =============================================================================
# 2D Slice tests for SaveFieldsToHDF5
# =============================================================================

def test_hdf5_field_callback_2d_slice_none(tmp_path):
    """Test SaveFieldsToHDF5 with no slice (full domain) in 2D."""
    sim = Simulation(nx=32, ny=32, dx=0.1, dy=0.1, npatch_x=2, npatch_y=2, dt_cfl=0.95)
    sim.initialize()
    for p in sim.patches:
        p.fields.ex[:, :] = p.ipatch_x * 10 + p.ipatch_y

    cb = SaveFieldsToHDF5(prefix=str(tmp_path / 'out'), interval=1, slice=None, components=['ex'])
    cb._call(sim)

    with h5py.File(tmp_path / 'out' / '000000.h5', 'r') as f:
        assert f['ex'].shape == (32, 32)
        ref = np.zeros((32, 32))
        for p in sim.patches:
            ix = p.ipatch_x * sim.nx_per_patch
            iy = p.ipatch_y * sim.ny_per_patch
            ref[ix:ix + sim.nx_per_patch, iy:iy + sim.ny_per_patch] = p.fields.ex[:sim.nx_per_patch, :sim.ny_per_patch]
        np.testing.assert_array_equal(f['ex'][:], ref)
        assert 'slice' not in f.attrs


def test_hdf5_field_callback_2d_slice_int(tmp_path):
    """Test SaveFieldsToHDF5 with an integer slice along y in 2D."""
    sim = Simulation(nx=32, ny=32, dx=0.1, dy=0.1, npatch_x=2, npatch_y=2, dt_cfl=0.95)
    sim.initialize()
    for p in sim.patches:
        p.fields.ex[:, :] = p.ipatch_x * 10 + p.ipatch_y

    cb = SaveFieldsToHDF5(prefix=str(tmp_path / 'out'), interval=1, slice=np.s_[:, 5], components=['ex'])
    cb._call(sim)

    with h5py.File(tmp_path / 'out' / '000000.h5', 'r') as f:
        assert f['ex'].shape == (32, 1)
        ref = np.zeros((32, 32))
        for p in sim.patches:
            ix = p.ipatch_x * sim.nx_per_patch
            iy = p.ipatch_y * sim.ny_per_patch
            ref[ix:ix + sim.nx_per_patch, iy:iy + sim.ny_per_patch] = p.fields.ex[:sim.nx_per_patch, :sim.ny_per_patch]
        np.testing.assert_array_equal(f['ex'][:, 0], ref[:, 5])
        assert f.attrs['slice'] == '[:, 5]'


def test_hdf5_field_callback_2d_slice_stepped(tmp_path):
    """Test SaveFieldsToHDF5 with stepped slicing in 2D."""
    sim = Simulation(nx=32, ny=32, dx=0.1, dy=0.1, npatch_x=2, npatch_y=2, dt_cfl=0.95)
    sim.initialize()
    for p in sim.patches:
        p.fields.ex[:, :] = p.ipatch_x * 10 + p.ipatch_y

    cb = SaveFieldsToHDF5(prefix=str(tmp_path / 'out'), interval=1, slice=np.s_[::2, ::3], components=['ex'])
    cb._call(sim)

    with h5py.File(tmp_path / 'out' / '000000.h5', 'r') as f:
        assert f['ex'].shape == (16, 11)
        ref = np.zeros((32, 32))
        for p in sim.patches:
            ix = p.ipatch_x * sim.nx_per_patch
            iy = p.ipatch_y * sim.ny_per_patch
            ref[ix:ix + sim.nx_per_patch, iy:iy + sim.ny_per_patch] = p.fields.ex[:sim.nx_per_patch, :sim.ny_per_patch]
        np.testing.assert_array_equal(f['ex'][:], ref[::2, ::3])
        assert f.attrs['slice'] == '[::2, ::3]'


def test_hdf5_field_callback_2d_slice_tail(tmp_path):
    """Test SaveFieldsToHDF5 with a tail slice (first half omitted) in 2D."""
    sim = Simulation(nx=32, ny=32, dx=0.1, dy=0.1, npatch_x=2, npatch_y=2, dt_cfl=0.95)
    sim.initialize()
    for p in sim.patches:
        p.fields.ex[:, :] = p.ipatch_x * 10 + p.ipatch_y

    cb = SaveFieldsToHDF5(prefix=str(tmp_path / 'out'), interval=1, slice=np.s_[16:, :], components=['ex'])
    cb._call(sim)

    with h5py.File(tmp_path / 'out' / '000000.h5', 'r') as f:
        assert f['ex'].shape == (16, 32)
        ref = np.zeros((32, 32))
        for p in sim.patches:
            ix = p.ipatch_x * sim.nx_per_patch
            iy = p.ipatch_y * sim.ny_per_patch
            ref[ix:ix + sim.nx_per_patch, iy:iy + sim.ny_per_patch] = p.fields.ex[:sim.nx_per_patch, :sim.ny_per_patch]
        np.testing.assert_array_equal(f['ex'][:], ref[16:, :])


# =============================================================================
# 3D Slice tests for SaveFieldsToHDF5
# =============================================================================

def test_hdf5_field_callback_3d_slice_none(tmp_path):
    """Test SaveFieldsToHDF5 with no slice (full domain) in 3D."""
    sim = Simulation3D(nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, npatch_x=2, npatch_y=2, npatch_z=2)
    sim.initialize()
    for p in sim.patches:
        p.fields.ex[:, :, :] = p.ipatch_x * 100 + p.ipatch_y * 10 + p.ipatch_z

    cb = SaveFieldsToHDF5(prefix=str(tmp_path / 'out3d'), interval=1, slice=None, components=['ex'])
    cb._call(sim)

    with h5py.File(tmp_path / 'out3d' / '000000.h5', 'r') as f:
        assert f['ex'].shape == (32, 32, 32)
        ref = np.zeros((32, 32, 32))
        for p in sim.patches:
            ix = p.ipatch_x * sim.nx_per_patch
            iy = p.ipatch_y * sim.ny_per_patch
            iz = p.ipatch_z * sim.nz_per_patch
            ref[ix:ix + sim.nx_per_patch, iy:iy + sim.ny_per_patch, iz:iz + sim.nz_per_patch] = p.fields.ex[:sim.nx_per_patch, :sim.ny_per_patch, :sim.nz_per_patch]
        np.testing.assert_array_equal(f['ex'][:], ref)
        assert 'slice' not in f.attrs


def test_hdf5_field_callback_3d_slice_plane(tmp_path):
    """Test SaveFieldsToHDF5 with a plane slice in 3D (single z index)."""
    sim = Simulation3D(nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, npatch_x=2, npatch_y=2, npatch_z=2)
    sim.initialize()
    for p in sim.patches:
        p.fields.ex[:, :, :] = p.ipatch_x * 100 + p.ipatch_y * 10 + p.ipatch_z

    cb = SaveFieldsToHDF5(prefix=str(tmp_path / 'out3d'), interval=1, slice=np.s_[:, :, 10], components=['ex'])
    cb._call(sim)

    with h5py.File(tmp_path / 'out3d' / '000000.h5', 'r') as f:
        assert f['ex'].shape == (32, 32, 1)
        ref = np.zeros((32, 32, 32))
        for p in sim.patches:
            ix = p.ipatch_x * sim.nx_per_patch
            iy = p.ipatch_y * sim.ny_per_patch
            iz = p.ipatch_z * sim.nz_per_patch
            ref[ix:ix + sim.nx_per_patch, iy:iy + sim.ny_per_patch, iz:iz + sim.nz_per_patch] = p.fields.ex[:sim.nx_per_patch, :sim.ny_per_patch, :sim.nz_per_patch]
        np.testing.assert_array_equal(f['ex'][:, :, 0], ref[:, :, 10])
        assert f.attrs['slice'] == '[:, :, 10]'


def test_hdf5_field_callback_3d_slice_stepped(tmp_path):
    """Test SaveFieldsToHDF5 with stepped slicing in 3D."""
    sim = Simulation3D(nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, npatch_x=2, npatch_y=2, npatch_z=2)
    sim.initialize()
    for p in sim.patches:
        p.fields.ex[:, :, :] = p.ipatch_x * 100 + p.ipatch_y * 10 + p.ipatch_z

    cb = SaveFieldsToHDF5(prefix=str(tmp_path / 'out3d'), interval=1, slice=np.s_[::2, ::2, ::5], components=['ex'])
    cb._call(sim)

    with h5py.File(tmp_path / 'out3d' / '000000.h5', 'r') as f:
        assert f['ex'].shape == (16, 16, 7)
        ref = np.zeros((32, 32, 32))
        for p in sim.patches:
            ix = p.ipatch_x * sim.nx_per_patch
            iy = p.ipatch_y * sim.ny_per_patch
            iz = p.ipatch_z * sim.nz_per_patch
            ref[ix:ix + sim.nx_per_patch, iy:iy + sim.ny_per_patch, iz:iz + sim.nz_per_patch] = p.fields.ex[:sim.nx_per_patch, :sim.ny_per_patch, :sim.nz_per_patch]
        np.testing.assert_array_equal(f['ex'][:], ref[::2, ::2, ::5])
        assert f.attrs['slice'] == '[::2, ::2, ::5]'


def test_hdf5_field_callback_3d_slice_tail(tmp_path):
    """Test SaveFieldsToHDF5 with a tail slice in 3D (first half along x omitted)."""
    sim = Simulation3D(nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, npatch_x=2, npatch_y=2, npatch_z=2)
    sim.initialize()
    for p in sim.patches:
        p.fields.ex[:, :, :] = p.ipatch_x * 100 + p.ipatch_y * 10 + p.ipatch_z

    cb = SaveFieldsToHDF5(prefix=str(tmp_path / 'out3d'), interval=1, slice=np.s_[16:, :, :], components=['ex'])
    cb._call(sim)

    with h5py.File(tmp_path / 'out3d' / '000000.h5', 'r') as f:
        assert f['ex'].shape == (16, 32, 32)
        ref = np.zeros((32, 32, 32))
        for p in sim.patches:
            ix = p.ipatch_x * sim.nx_per_patch
            iy = p.ipatch_y * sim.ny_per_patch
            iz = p.ipatch_z * sim.nz_per_patch
            ref[ix:ix + sim.nx_per_patch, iy:iy + sim.ny_per_patch, iz:iz + sim.nz_per_patch] = p.fields.ex[:sim.nx_per_patch, :sim.ny_per_patch, :sim.nz_per_patch]
        np.testing.assert_array_equal(f['ex'][:], ref[16:, :, :])


# =============================================================================
# Invalid slice edge cases for SaveFieldsToHDF5
# =============================================================================

def test_hdf5_field_callback_invalid_slice_type(tmp_path):
    """Test SaveFieldsToHDF5 raises ValueError for invalid slice element type (list)."""
    sim = Simulation(nx=32, ny=32, dx=0.1, dy=0.1, npatch_x=2, npatch_y=2, dt_cfl=0.95)
    sim.initialize()
    cb = SaveFieldsToHDF5(prefix=str(tmp_path / 'out'), interval=1, slice=([0, 1], slice(None)), components=['ex'])
    with pytest.raises(ValueError):
        cb._call(sim)


def test_hdf5_field_callback_invalid_slice_ellipsis(tmp_path):
    """Test SaveFieldsToHDF5 raises ValueError for Ellipsis in slice."""
    sim = Simulation3D(nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, npatch_x=2, npatch_y=2, npatch_z=2)
    sim.initialize()
    cb = SaveFieldsToHDF5(prefix=str(tmp_path / 'out'), interval=1, slice=np.s_[..., 10], components=['ex'])
    with pytest.raises(ValueError):
        cb._call(sim)


def test_hdf5_field_callback_invalid_slice_neg_step(tmp_path):
    """Test SaveFieldsToHDF5 raises ValueError for negative step in slice."""
    sim = Simulation3D(nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, npatch_x=2, npatch_y=2, npatch_z=2)
    sim.initialize()
    cb = SaveFieldsToHDF5(prefix=str(tmp_path / 'out'), interval=1, slice=np.s_[::-1, :, :], components=['ex'])
    with pytest.raises(ValueError):
        cb._call(sim)


def test_hdf5_field_callback_invalid_slice_axis_mismatch(tmp_path):
    """Test SaveFieldsToHDF5 raises ValueError for axis count mismatch."""
    sim = Simulation3D(nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, npatch_x=2, npatch_y=2, npatch_z=2)
    sim.initialize()
    cb = SaveFieldsToHDF5(prefix=str(tmp_path / 'out'), interval=1, slice=np.s_[:, :], components=['ex'])
    with pytest.raises(ValueError):
        cb._call(sim)


def test_hdf5_field_callback_invalid_slice_empty(tmp_path):
    """Test SaveFieldsToHDF5 raises ValueError for empty slice (start >= stop)."""
    sim = Simulation3D(nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, npatch_x=2, npatch_y=2, npatch_z=2)
    sim.initialize()
    cb = SaveFieldsToHDF5(prefix=str(tmp_path / 'out'), interval=1, slice=np.s_[0:0, :, :], components=['ex'])
    with pytest.raises(ValueError):
        cb._call(sim)


def test_hdf5_field_callback_invalid_slice_newaxis(tmp_path):
    """Test SaveFieldsToHDF5 raises ValueError for None/newaxis in slice."""
    sim = Simulation3D(nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, npatch_x=2, npatch_y=2, npatch_z=2)
    sim.initialize()
    cb = SaveFieldsToHDF5(prefix=str(tmp_path / 'out'), interval=1, slice=np.s_[None, :, :], components=['ex'])
    with pytest.raises(ValueError):
        cb._call(sim)


def test_hdf5_field_callback_invalid_slice_zero_step(tmp_path):
    """Test SaveFieldsToHDF5 raises ValueError for zero step in slice."""
    sim = Simulation3D(nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, npatch_x=2, npatch_y=2, npatch_z=2)
    sim.initialize()
    cb = SaveFieldsToHDF5(prefix=str(tmp_path / 'out'), interval=1, slice=np.s_[:, ::0, :], components=['ex'])
    with pytest.raises(ValueError):
        cb._call(sim)


def test_hdf5_field_callback_invalid_slice_out_of_range(tmp_path):
    """Test SaveFieldsToHDF5 raises ValueError for out-of-range slice start."""
    sim = Simulation3D(nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, npatch_x=2, npatch_y=2, npatch_z=2)
    sim.initialize()
    cb = SaveFieldsToHDF5(prefix=str(tmp_path / 'out'), interval=1, slice=np.s_[999:, :, :], components=['ex'])
    with pytest.raises(ValueError):
        cb._call(sim)


def test_extract_density_2d_slice_none(tmp_path):
    """Test ExtractSpeciesDensity with slice=None (full domain) in 2D."""
    sim = Simulation(nx=32, ny=32, dx=0.1, dy=0.1, npatch_x=2, npatch_y=2, dt_cfl=0.95)
    electrons = Electron(name="electrons", density=lambda x, y: 1.0, ppc=4)
    sim.add_species([electrons])
    cb = ExtractSpeciesDensity(sim, electrons, interval=1, slice=None)
    sim.run(1, callbacks=[cb])
    assert cb.density.shape == (32, 32)
    assert np.isfinite(cb.density).all()
    assert cb.density.max() > 0


def test_extract_density_2d_slice_int(tmp_path):
    """Test ExtractSpeciesDensity with int slice in 2D."""
    sim = Simulation(nx=32, ny=32, dx=0.1, dy=0.1, npatch_x=2, npatch_y=2, dt_cfl=0.95)
    electrons = Electron(name="electrons", density=lambda x, y: 1.0, ppc=4)
    sim.add_species([electrons])
    cb_full = ExtractSpeciesDensity(sim, electrons, interval=1, slice=None)
    cb_slice = ExtractSpeciesDensity(sim, electrons, interval=1, slice=np.s_[:, 5])
    sim.run(1, callbacks=[cb_full, cb_slice])
    assert cb_slice.density.shape == (32, 1)
    assert np.allclose(cb_slice.density, cb_full.density[:, 5:6])


def test_extract_density_2d_slice_stepped(tmp_path):
    """Test ExtractSpeciesDensity with stepped slice in 2D."""
    sim = Simulation(nx=32, ny=32, dx=0.1, dy=0.1, npatch_x=2, npatch_y=2, dt_cfl=0.95)
    electrons = Electron(name="electrons", density=lambda x, y: 1.0, ppc=4)
    sim.add_species([electrons])
    cb_full = ExtractSpeciesDensity(sim, electrons, interval=1, slice=None)
    cb_slice = ExtractSpeciesDensity(sim, electrons, interval=1, slice=np.s_[::2, ::3])
    sim.run(1, callbacks=[cb_full, cb_slice])
    assert cb_slice.density.shape == (16, 11)
    assert np.allclose(cb_slice.density, cb_full.density[::2, ::3])


def test_extract_density_3d_slice_none(tmp_path):
    """Test ExtractSpeciesDensity with slice=None (full domain) in 3D."""
    sim = Simulation3D(nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, npatch_x=2, npatch_y=2, npatch_z=2, dt_cfl=0.95)
    electrons = Electron(name="electrons", density=lambda x, y, z: 1.0, ppc=4)
    sim.add_species([electrons])
    cb = ExtractSpeciesDensity(sim, electrons, interval=1, slice=None)
    sim.run(1, callbacks=[cb])
    assert cb.density.shape == (32, 32, 32)
    assert np.isfinite(cb.density).all()
    assert cb.density.max() > 0


def test_extract_density_3d_slice_plane(tmp_path):
    """Test ExtractSpeciesDensity with plane slice in 3D."""
    sim = Simulation3D(nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, npatch_x=2, npatch_y=2, npatch_z=2, dt_cfl=0.95)
    electrons = Electron(name="electrons", density=lambda x, y, z: 1.0, ppc=4)
    sim.add_species([electrons])
    cb_full = ExtractSpeciesDensity(sim, electrons, interval=1, slice=None)
    cb_slice = ExtractSpeciesDensity(sim, electrons, interval=1, slice=np.s_[:, :, 10])
    sim.run(1, callbacks=[cb_full, cb_slice])
    assert cb_slice.density.shape == (32, 32, 1)
    assert np.allclose(cb_slice.density, cb_full.density[:, :, 10:11])


def test_extract_density_3d_slice_stepped(tmp_path):
    """Test ExtractSpeciesDensity with stepped slice in 3D."""
    sim = Simulation3D(nx=32, ny=32, nz=32, dx=0.1, dy=0.1, dz=0.1, npatch_x=2, npatch_y=2, npatch_z=2, dt_cfl=0.95)
    electrons = Electron(name="electrons", density=lambda x, y, z: 1.0, ppc=4)
    sim.add_species([electrons])
    cb_full = ExtractSpeciesDensity(sim, electrons, interval=1, slice=None)
    cb_slice = ExtractSpeciesDensity(sim, electrons, interval=1, slice=np.s_[::2, ::2, ::5])
    sim.run(1, callbacks=[cb_full, cb_slice])
    assert cb_slice.density.shape == (16, 16, 7)
    assert np.allclose(cb_slice.density, cb_full.density[::2, ::2, ::5])
