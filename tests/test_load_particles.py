import pytest
import h5py
import os
import numpy as np
from pathlib import Path
from lambdapic.simulation import Simulation, Simulation3D
from lambdapic.core.species import Electron
from lambdapic.callback.utils import LoadParticles
from lambdapic.callback.hdf5 import SaveParticlesToHDF5

def test_load_particles_callback_2d(tmp_path):
    """Test loading particle data from HDF5 in 2D simulation."""
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
    
    # Add electron species
    electrons = Electron(
        name="electrons",
        density=lambda x, y: 1.0,
        ppc=4,
    )
    sim.add_species([electrons])
    
    # Initialize simulation to create particles
    sim.initialize()
    
    # First, save particles to HDF5 to create test data
    output_dir = tmp_path / "particles_output"
    save_callback = SaveParticlesToHDF5(
        species=electrons,
        prefix=str(output_dir),
        interval=1,
        attrs=['x', 'y', 'w']
    )
    
    # Save particles at time 0
    save_callback(sim)
    
    # Verify the HDF5 file was created
    h5_file = output_dir / "electrons_particles_000000.h5"
    assert os.path.exists(h5_file)
    
    # Verify file contents
    with h5py.File(h5_file, 'r') as f:
        assert 'x' in f
        assert 'y' in f
        assert 'w' in f
        
        # Get original particle data for comparison
        original_x = f['x'][:]
        original_y = f['y'][:]
        original_w = f['w'][:]
    
    # Create a new simulation to test loading
    sim2 = Simulation(
        nx=32,
        ny=32,
        dx=0.1,
        dy=0.1,
        npatch_x=2,
        npatch_y=2,
        dt_cfl=0.95
    )
    
    # Add the same electron species
    electrons2 = Electron(
        name="electrons",
        density=lambda x, y: 0.0,  # Zero density to start with empty particles
        ppc=0,
    )
    sim2.add_species([electrons2])
    
    # Initialize the second simulation
    sim2.initialize()
    
    # Verify initial particles are empty (due to zero density)
    total_particles_before = 0
    for p in sim2.patches:
        total_particles_before += p.particles[electrons2.ispec].is_alive.sum()
    assert total_particles_before == 0
    
    # Create and run with load particles callback
    load_callback = LoadParticles(
        species=electrons2,
        file=str(h5_file),
        interval=lambda sim: sim.itime == 0  # Load only at first timestep
    )
    
    # Execute the load callback
    load_callback(sim2)
    
    # Verify particles were loaded correctly
    total_particles_after = 0
    loaded_x = []
    loaded_y = []
    loaded_w = []
    loaded_id = []
    
    for p in sim2.patches:
        particles = p.particles[electrons2.ispec]
        alive = particles.is_alive
        total_particles_after += alive.sum()
        
        # Collect loaded particle data
        loaded_x.extend(particles.x[alive])
        loaded_y.extend(particles.y[alive])
        loaded_w.extend(particles.w[alive])
        loaded_id.extend(particles.id[alive])
    
    # Verify total number of particles matches
    assert total_particles_after == len(original_x)
    
    # Verify particle data matches (within floating point precision)
    np.testing.assert_allclose(np.sort(loaded_x), np.sort(original_x), rtol=1e-10)
    np.testing.assert_allclose(np.sort(loaded_y), np.sort(original_y), rtol=1e-10)
    np.testing.assert_allclose(np.sort(loaded_w), np.sort(original_w), rtol=1e-10)
    
def test_load_particles_missing_attributes(tmp_path):
    """Test LoadParticles behavior with missing attributes in HDF5 file."""
    # Create test HDF5 file with only some attributes
    h5_file = tmp_path / "incomplete_particles.h5"
    
    num_particles = 100
    with h5py.File(h5_file, 'w') as f:
        f.create_dataset('x', data=np.random.uniform(0, 3.2, num_particles))
        f.create_dataset('y', data=np.random.uniform(0, 3.2, num_particles))
        # Intentionally omit 'w' and 'id' to test error handling
    
    # Setup simulation
    sim = Simulation(
        nx=32,
        ny=32,
        dx=0.1,
        dy=0.1,
        npatch_x=2,
        npatch_y=2,
        dt_cfl=0.95
    )
    
    electrons = Electron(
        name="electrons",
        density=lambda x, y: 0.0,
        ppc=0,
    )
    sim.add_species([electrons])
    sim.initialize()
    
    load_callback = LoadParticles(
        species=electrons,
        file=str(h5_file)
    )
    
    # Should raise an error when trying to access missing attributes
    with pytest.raises(ValueError):
        load_callback(sim)

def test_load_particles_empty_file(tmp_path):
    """Test LoadParticles with empty HDF5 file."""
    # Create empty HDF5 file
    h5_file = tmp_path / "empty_particles.h5"
    
    with h5py.File(h5_file, 'w') as f:
        # Create empty datasets
        f.create_dataset('x', data=np.array([]))
        f.create_dataset('y', data=np.array([]))
        f.create_dataset('w', data=np.array([]))
        f.create_dataset('id', data=np.array([], dtype=np.uint64))
        f.attrs['time'] = 0.0
        f.attrs['itime'] = 0
    
    # Setup simulation
    sim = Simulation(
        nx=32,
        ny=32,
        dx=0.1,
        dy=0.1,
        npatch_x=2,
        npatch_y=2,
        dt_cfl=0.95
    )
    
    electrons = Electron(
        name="electrons",
        density=lambda x, y: 0.0,
        ppc=0,
    )
    sim.add_species([electrons])
    sim.initialize()
    
    load_callback = LoadParticles(
        species=electrons,
        file=str(h5_file)
    )
    
    # Should not raise an error with empty file
    load_callback(sim)
    
    # Verify no particles were loaded
    total_particles = 0
    for p in sim.patches:
        total_particles += p.particles[electrons.ispec].is_alive.sum()
    
    assert total_particles == 0

def test_load_particles_file_not_found():
    """Test LoadParticles with non-existent file."""
    # Setup simulation
    sim = Simulation(
        nx=32,
        ny=32,
        dx=0.1,
        dy=0.1,
        npatch_x=2,
        npatch_y=2,
        dt_cfl=0.95
    )
    
    electrons = Electron(
        name="electrons",
        density=lambda x, y: 1.0,
        ppc=4,
    )
    sim.add_species([electrons])
    sim.initialize()
    
    load_callback = LoadParticles(
        species=electrons,
        file="non_existent_file.h5"
    )
    
    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_callback(sim)

def test_load_particles_callback_3d(tmp_path):
    """Test loading particle data from HDF5 in 3D simulation."""
    # Setup minimal 3D simulation
    sim = Simulation3D(
        nx=16,
        ny=16,
        nz=16,
        dx=0.1,
        dy=0.1,
        dz=0.1,
        npatch_x=2,
        npatch_y=2,
        npatch_z=2,
        dt_cfl=0.95
    )
    
    # Add electron species
    electrons = Electron(
        name="electrons",
        density=lambda x, y, z: 1.0,
        ppc=4,
    )
    sim.add_species([electrons])
    
    # Initialize simulation to create particles
    sim.initialize()
    
    # First, save particles to HDF5 to create test data
    output_dir = tmp_path / "particles_output_3d"
    save_callback = SaveParticlesToHDF5(
        species=electrons,
        prefix=str(output_dir),
        interval=1,
        attrs=['x', 'y', 'z', 'w']
    )
    
    # Save particles at time 0
    save_callback(sim)
    
    # Verify the HDF5 file was created
    h5_file = output_dir / "electrons_particles_000000.h5"
    assert os.path.exists(h5_file)
    
    # Verify file contents
    with h5py.File(h5_file, 'r') as f:
        assert 'x' in f
        assert 'y' in f
        assert 'z' in f
        assert 'w' in f
        
        # Get original particle data for comparison
        original_x = f['x'][:]
        original_y = f['y'][:]
        original_z = f['z'][:]
        original_w = f['w'][:]
    
    # Create a new simulation to test loading
    sim2 = Simulation3D(
        nx=16,
        ny=16,
        nz=16,
        dx=0.1,
        dy=0.1,
        dz=0.1,
        npatch_x=2,
        npatch_y=2,
        npatch_z=2,
        dt_cfl=0.95
    )
    
    # Add the same electron species
    electrons2 = Electron(
        name="electrons",
        density=lambda x, y, z: 0.0,  # Zero density to start with empty particles
        ppc=0,
    )
    sim2.add_species([electrons2])
    
    # Initialize the second simulation
    sim2.initialize()
    
    # Verify initial particles are empty (due to zero density)
    total_particles_before = 0
    for p in sim2.patches:
        total_particles_before += p.particles[electrons2.ispec].is_alive.sum()
    assert total_particles_before == 0
    
    # Create and run with load particles callback
    load_callback = LoadParticles(
        species=electrons2,
        file=str(h5_file),
        interval=lambda sim: sim.itime == 0  # Load only at first timestep
    )
    
    # Execute the load callback
    load_callback(sim2)
    
    # Verify particles were loaded correctly
    total_particles_after = 0
    loaded_x = []
    loaded_y = []
    loaded_z = []
    loaded_w = []
    loaded_id = []
    
    for p in sim2.patches:
        particles = p.particles[electrons2.ispec]
        alive = particles.is_alive
        total_particles_after += alive.sum()
        
        # Collect loaded particle data
        loaded_x.extend(particles.x[alive])
        loaded_y.extend(particles.y[alive])
        loaded_z.extend(particles.z[alive])
        loaded_w.extend(particles.w[alive])
        loaded_id.extend(particles.id[alive])
    
    # Verify total number of particles matches
    assert total_particles_after == len(original_x)
    
    # Verify particle data matches (within floating point precision)
    np.testing.assert_allclose(np.sort(loaded_x), np.sort(original_x), rtol=1e-10)
    np.testing.assert_allclose(np.sort(loaded_y), np.sort(original_y), rtol=1e-10)
    np.testing.assert_allclose(np.sort(loaded_z), np.sort(original_z), rtol=1e-10)
    np.testing.assert_allclose(np.sort(loaded_w), np.sort(original_w), rtol=1e-10)
