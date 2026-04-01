"""Test rebalancing functionality for 2D and 3D simulations."""
import pytest
import numpy as np
from mpi4py import MPI

from lambdapic import Simulation2D, Simulation3D, Electron


def _store_particles_2d(sim):
    """Store particle data for 2D simulation."""
    particles_local = {}
    for p in sim.patches:
        for ispec in range(len(sim.patches.species)):
            particles = p.particles[ispec]
            alive_mask = ~particles.is_dead
            for i in range(len(particles._id)):
                if alive_mask[i]:
                    particles_local[particles._id[i]] = {
                        'x': particles.x[i],
                        'y': particles.y[i],
                        'ux': particles.ux[i],
                        'uy': particles.uy[i],
                        'w': particles.w[i]
                    }
    return particles_local


def _store_particles_3d(sim):
    """Store particle data for 3D simulation."""
    particles_local = {}
    for p in sim.patches:
        for ispec in range(len(sim.patches.species)):
            particles = p.particles[ispec]
            alive_mask = ~particles.is_dead
            for i in range(len(particles._id)):
                if alive_mask[i]:
                    particles_local[particles._id[i]] = {
                        'x': particles.x[i],
                        'y': particles.y[i],
                        'z': particles.z[i],
                        'ux': particles.ux[i],
                        'uy': particles.uy[i],
                        'uz': particles.uz[i],
                        'w': particles.w[i]
                    }
    return particles_local


def _verify_particles_2d(before_particles, after_particles):
    """Verify particles preserved correctly in 2D simulation."""
    assert len(after_particles) == len(before_particles), \
        f"Particle count mismatch: {len(before_particles)} -> {len(after_particles)}"

    sample_size = min(100, len(before_particles))
    sample_ids = list(before_particles.keys())[:sample_size]

    for pid in sample_ids:
        assert pid in after_particles, f"Particle {pid} missing after rebalance"
        before = before_particles[pid]
        after = after_particles[pid]

        assert np.isclose(after['x'], before['x']), f"Particle {pid} x changed"
        assert np.isclose(after['y'], before['y']), f"Particle {pid} y changed"
        assert np.isclose(after['ux'], before['ux']), f"Particle {pid} ux changed"
        assert np.isclose(after['uy'], before['uy']), f"Particle {pid} uy changed"
        assert np.isclose(after['w'], before['w']), f"Particle {pid} w changed"


def _verify_particles_3d(before_particles, after_particles):
    """Verify particles preserved correctly in 3D simulation."""
    assert len(after_particles) == len(before_particles), \
        f"Particle count mismatch: {len(before_particles)} -> {len(after_particles)}"

    sample_size = min(100, len(before_particles))
    sample_ids = list(before_particles.keys())[:sample_size]

    for pid in sample_ids:
        assert pid in after_particles, f"Particle {pid} missing after rebalance"
        before = before_particles[pid]
        after = after_particles[pid]

        assert np.isclose(after['x'], before['x']), f"Particle {pid} x changed"
        assert np.isclose(after['y'], before['y']), f"Particle {pid} y changed"
        assert np.isclose(after['z'], before['z']), f"Particle {pid} z changed"
        assert np.isclose(after['ux'], before['ux']), f"Particle {pid} ux changed"
        assert np.isclose(after['uy'], before['uy']), f"Particle {pid} uy changed"
        assert np.isclose(after['uz'], before['uz']), f"Particle {pid} uz changed"
        assert np.isclose(after['w'], before['w']), f"Particle {pid} w changed"


def _get_particle_count(sim, comm):
    """Get global particle count."""
    npart_local = sum((~p.particles[0].is_dead).sum() for p in sim.patches)
    return comm.allreduce(npart_local, op=MPI.SUM)


def _run_rebalance_test(sim, comm, rank, store_func, verify_func, npatches):
    """Common rebalance test logic."""
    sim.initialize()

    initial_npart_global = _get_particle_count(sim, comm)
    if rank == 0:
        print(f"Initial global particles: {initial_npart_global}")

    sim.run(nsteps=5)

    before_npart_global = _get_particle_count(sim, comm)
    before_particles_local = store_func(sim)

    if rank == 0:
        print("Starting rebalance...")
    sim.rebalance()
    if rank == 0:
        print("Rebalance completed")

    after_npart_global = _get_particle_count(sim, comm)
    assert after_npart_global == before_npart_global, \
        f"Particle count changed: {before_npart_global} -> {after_npart_global}"

    after_particles_local = store_func(sim)

    all_before_particles = comm.gather(before_particles_local, root=0)
    all_after_particles = comm.gather(after_particles_local, root=0)

    if rank == 0:
        before_particles = {}
        for p_dict in all_before_particles:
            before_particles.update(p_dict)

        after_particles = {}
        for p_dict in all_after_particles:
            after_particles.update(p_dict)

        verify_func(before_particles, after_particles)

    sim.run(nsteps=5)

    final_npart_global = _get_particle_count(sim, comm)
    if rank == 0:
        print(f"Final global particles: {final_npart_global}")
        print(f"Test passed: particles conserved through rebalance")


@pytest.mark.mpi
@pytest.mark.slow
def test_rebalance_2d():
    """Test 2D rebalancing with multiple ranks."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        pytest.skip("Test requires at least 2 MPI ranks")

    def density_func(x, y):
        return 1.0 + 9.0 * np.exp(-((x - 30e-6)**2 + (y - 30e-6)**2) / (10e-6)**2)

    electrons = Electron(density=density_func, ppc=1, pusher="boris")

    sim = Simulation2D(
        nx=60, ny=60, dx=1e-6, dy=1e-6,
        npatch_x=3, npatch_y=3, nsteps=10, n_guard=3, dt_cfl=0.95,
        boundary_conditions={
            'xmin': 'periodic', 'xmax': 'periodic',
            'ymin': 'periodic', 'ymax': 'periodic'
        },
        comm=comm
    )
    sim.add_species([electrons])

    _run_rebalance_test(sim, comm, rank, _store_particles_2d, _verify_particles_2d, 9)


@pytest.mark.mpi
@pytest.mark.slow
def test_rebalance_3d():
    """Test 3D rebalancing with multiple ranks."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        pytest.skip("Test requires at least 2 MPI ranks")

    def density_func(x, y, z):
        return 1.0 + 9.0 * np.exp(
            -((x - 20e-6)**2 + (y - 20e-6)**2 + (z - 20e-6)**2) / (10e-6)**2
        )

    electrons = Electron(density=density_func, ppc=10, pusher="boris")

    sim = Simulation3D(
        nx=40, ny=40, nz=40, dx=1e-6, dy=1e-6, dz=1e-6,
        npatch_x=2, npatch_y=2, npatch_z=2, nsteps=10, n_guard=3, dt_cfl=0.95,
        boundary_conditions={
            'xmin': 'periodic', 'xmax': 'periodic',
            'ymin': 'periodic', 'ymax': 'periodic',
            'zmin': 'periodic', 'zmax': 'periodic'
        },
        comm=comm
    )
    sim.add_species([electrons])

    _run_rebalance_test(sim, comm, rank, _store_particles_3d, _verify_particles_3d, 8)


if __name__ == "__main__":
    test_rebalance_2d()
    test_rebalance_3d()
