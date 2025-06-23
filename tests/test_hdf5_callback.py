import pytest
import h5py
import os
import numpy as np
from lambdapic.simulation import Simulation
from lambdapic.core.species import Electron
from lambdapic.callback.hdf5 import (
    SaveFieldsToHDF5,
    SaveSpeciesDensityToHDF5,
    SaveParticlesToHDF5
)

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

def test_hdf5_particles_callback_2d(tmp_path):
    """Test saving particle data to HDF5 in 2D simulation."""
    sim = Simulation(
        nx=32,
        ny=32,
        dx=0.1,
        dy=0.1,
        npatch_x=2,
        npatch_y=2,
        dt_cfl=0.95
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
