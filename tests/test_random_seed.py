"""
Test random seed reproducibility in LambdaPIC simulations.
"""

import numpy as np
import pytest
from lambdapic import Simulation, Simulation3D, Electron


def test_random_seed_reproducibility_2d():
    """Test that 2D simulations with the same seed produce identical results."""
    # Create two simulations with the same seed
    sim1 = Simulation(
        nx=32, ny=32, dx=1.0, dy=1.0,
        npatch_x=2, npatch_y=2,
        random_seed=42
    )
    
    sim2 = Simulation(
        nx=32, ny=32, dx=1.0, dy=1.0,
        npatch_x=2, npatch_y=2,
        random_seed=42
    )
    
    # Add the same species to both simulations
    electron1 = Electron(density=lambda x, y: 1.0, ppc=4)
    electron2 = Electron(density=lambda x, y: 1.0, ppc=4)
    
    sim1.add_species([electron1])
    sim2.add_species([electron2])
    
    # Initialize both simulations
    sim1.initialize()
    sim2.initialize()
    
    # Check that particle positions are identical
    for ipatch in range(len(sim1.patches)):
        for ispec in range(len(sim1.patches.species)):
            particles1 = sim1.patches[ipatch].particles[ispec]
            particles2 = sim2.patches[ipatch].particles[ispec]
            
            # Check same number of particles
            assert particles1.npart == particles2.npart
            
            # Check identical positions
            np.testing.assert_array_equal(particles1.x, particles2.x)
            np.testing.assert_array_equal(particles1.y, particles2.y)
            np.testing.assert_array_equal(particles1.w, particles2.w)


def test_random_seed_different_results():
    """Test that different seeds produce different results."""
    # Create two simulations with different seeds
    sim1 = Simulation(
        nx=32, ny=32, dx=1.0, dy=1.0,
        npatch_x=2, npatch_y=2,
        random_seed=42
    )
    
    sim2 = Simulation(
        nx=32, ny=32, dx=1.0, dy=1.0,
        npatch_x=2, npatch_y=2,
        random_seed=123
    )
    
    # Add the same species to both simulations
    electron = Electron(density=lambda x, y: 1.0, ppc=4)
    
    sim1.add_species([electron])
    sim2.add_species([electron])
    
    # Initialize both simulations
    sim1.initialize()
    sim2.initialize()
    
    # Check that particle positions are different
    positions_different = False
    for ipatch in range(len(sim1.patches)):
        for ispec in range(len(sim1.patches.species)):
            particles1 = sim1.patches[ipatch].particles[ispec]
            particles2 = sim2.patches[ipatch].particles[ispec]
            
            # Check if any positions are different
            if not np.array_equal(particles1.x, particles2.x):
                positions_different = True
                break
    
    assert positions_different, "Different seeds should produce different particle positions"


def test_random_seed_none():
    """Test that random_seed=None uses non-deterministic initialization."""
    sim = Simulation(
        nx=32, ny=32, dx=1.0, dy=1.0,
        npatch_x=2, npatch_y=2,
        random_seed=None
    )
    
    electron = Electron(density=lambda x, y: 1.0, ppc=4)
    
    sim.add_species([electron])
    
    # Should initialize without error
    sim.initialize()


def test_random_seed_reproducibility_3d():
    """Test that 3D simulations with the same seed produce identical results."""
    # Create two 3D simulations with the same seed
    sim1 = Simulation3D(
        nx=16, ny=16, nz=16,
        dx=1.0, dy=1.0, dz=1.0,
        npatch_x=2, npatch_y=2, npatch_z=2,
        random_seed=42
    )
    
    sim2 = Simulation3D(
        nx=16, ny=16, nz=16,
        dx=1.0, dy=1.0, dz=1.0,
        npatch_x=2, npatch_y=2, npatch_z=2,
        random_seed=42
    )
    
    # Add the same species to both simulations
    electron1 = Electron(density=lambda x, y, z: 1.0, ppc=4)
    electron2 = Electron(density=lambda x, y, z: 1.0, ppc=4)
    
    sim1.add_species([electron1])
    sim2.add_species([electron2])
    
    # Initialize both simulations
    sim1.initialize()
    sim2.initialize()
    
    # Check that particle positions are identical
    for ipatch in range(len(sim1.patches)):
        for ispec in range(len(sim1.patches.species)):
            particles1 = sim1.patches[ipatch].particles[ispec]
            particles2 = sim2.patches[ipatch].particles[ispec]
            
            # Check same number of particles
            assert particles1.npart == particles2.npart
            
            # Check identical positions
            np.testing.assert_array_equal(particles1.x, particles2.x)
            np.testing.assert_array_equal(particles1.y, particles2.y)
            np.testing.assert_array_equal(particles1.z, particles2.z)
            np.testing.assert_array_equal(particles1.w, particles2.w)


if __name__ == "__main__":
    test_random_seed_reproducibility_2d()
    test_random_seed_different_results()
    test_random_seed_none()
    test_random_seed_reproducibility_3d()
    print("All random seed tests passed!")