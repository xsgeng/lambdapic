import numpy as np
import pytest

from lambdapic import Simulation
from lambdapic.core.species import Electron
from lambdapic.core.collision.collision import Collision
from lambdapic.callback.utils import SetTemperature


def _assert_no_nans_particles_on_patches(sim, ispec: int):
    for p in sim.patches:
        part = p.particles[ispec]
        assert len(part.ux) > 0
        for arr in (part.ux, part.uy, part.uz, part.inv_gamma):
            assert np.all(np.isfinite(arr))
        assert np.all(part.inv_gamma > 0)
        assert np.all(part.inv_gamma <= 1.0)


def _total_energy(sim, ispecs):
    E = 0.0
    for ispec in ispecs:
        m = sim.patches.species[ispec].m
        for p in sim.patches:
            part = p.particles[ispec]
            alive = ~part.is_dead
            if alive.any():
                E += np.sum(part.w[alive] * (1.0 / part.inv_gamma[alive]) * m)
    return E


def _setup_sim(nx=16, ny=16, dx=1e-6, dy=1e-6, npatch_x=2, npatch_y=2, ppc=100):
    sim = Simulation(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        npatch_x=npatch_x,
        npatch_y=npatch_y,
        random_seed=1234,
    )

    # Two electron species; names will be uniquified internally
    e1 = Electron(density=lambda x, y: 1.0e29, ppc=ppc)
    e2 = Electron(density=lambda x, y: 1.0e29, ppc=ppc)
    sim.add_species([e1, e2])

    # Register collision groups before initialize so _init_sorter creates
    # 2D-bucket sorters (ny_buckets=patches.ny). 2D buckets force
    # reverse_x=False in _decide_reverse_x, keeping both species' bucket
    # numbering consistent for inter-species collisions. Without this,
    # sorters would take the 1D-bucket path and _decide_reverse_x could
    # mirror one species but not the other, silently breaking collisions.
    sim.add_collision([[e1, e2]])

    sim.initialize()

    # Heat particles: sample relativistic Maxwell-Jüttner to ensure collisions
    def _heat_species(ispec: int, theta: float = 0.1):
        for p in sim.patches:
            part = p.particles[ispec]
            alive = part.is_alive
            n = int(alive.sum())
            if n == 0:
                continue
            ux, uy, uz = SetTemperature.sample_maxwell_juttner(n, theta)
            part.ux[alive] = ux
            part.uy[alive] = uy
            part.uz[alive] = uz
            part.inv_gamma[alive] = 1.0 / np.sqrt(1.0 + part.ux[alive]**2 + part.uy[alive]**2 + part.uz[alive]**2)

    _heat_species(0, theta=0.1)
    _heat_species(1, theta=0.1)

    # Ensure bucket bounds are computed for collisions after heating
    for sorter in sim.sorter:
        sorter()

    return sim


def _setup_inter_collision(sim: Simulation, lnLambda: float = 2.0) -> Collision:
    # sim.collision was created by add_collision() + initialize(); its
    # bucket_bound_*_lists hold references that were refreshed by the
    # post-heating sort in _setup_sim. Only debye length and lnLambda
    # remain to be set.
    coll = sim.collision
    coll.lnLambda = lnLambda
    coll.calculate_debye_length()
    return coll


def _setup_intra_collision(sim: Simulation, ispec: int = 0, lnLambda: float = 2.0) -> Collision:
    """Setup Collision to perform ONLY intra-species collisions for the chosen species."""
    species = sim.patches.species
    groups = [[species[ispec], species[ispec]]]  # only intra for the selected species
    coll = Collision(groups, sim.patches, sim.sorter, sim.rand_gen)

    coll.lnLambda = lnLambda
    coll.generate_particle_lists()
    coll.generate_field_lists()
    coll.calculate_debye_length()
    return coll


def test_inter_collision_runs_no_nans_and_conserves_energy():
    sim = _setup_sim()
    coll = _setup_inter_collision(sim, lnLambda=2.0)

    # Baseline checks
    _assert_no_nans_particles_on_patches(sim, 0)
    _assert_no_nans_particles_on_patches(sim, 1)

    E_before = _total_energy(sim, [0, 1])

    # Take a small collision step
    dt = 1e-15
    coll(dt)

    # Post-collision validity
    _assert_no_nans_particles_on_patches(sim, 0)
    _assert_no_nans_particles_on_patches(sim, 1)

    E_after = _total_energy(sim, [0, 1])
    assert np.isclose(E_before, E_after, rtol=1e-3, atol=0.0)


def test_inter_collision_alters_momenta_for_inter_species():
    sim = _setup_sim()
    coll = _setup_inter_collision(sim, lnLambda=2.0)

    # Snapshot momenta
    ux0 = [p.particles[0].ux.copy() for p in sim.patches]
    uy0 = [p.particles[0].uy.copy() for p in sim.patches]
    uz0 = [p.particles[0].uz.copy() for p in sim.patches]

    ux1 = [p.particles[1].ux.copy() for p in sim.patches]
    uy1 = [p.particles[1].uy.copy() for p in sim.patches]
    uz1 = [p.particles[1].uz.copy() for p in sim.patches]

    # Advance collisions
    coll(1e-15)

    # Check that at least one component changed for either species
    changed = False
    for ip, p in enumerate(sim.patches):
        if not np.allclose(p.particles[0].ux, ux0[ip]) or \
           not np.allclose(p.particles[0].uy, uy0[ip]) or \
           not np.allclose(p.particles[0].uz, uz0[ip]):
            changed = True
            break
        if not np.allclose(p.particles[1].ux, ux1[ip]) or \
           not np.allclose(p.particles[1].uy, uy1[ip]) or \
           not np.allclose(p.particles[1].uz, uz1[ip]):
            changed = True
            break

    assert changed, "Expected collisions to alter at least one momentum component"


def test_intra_collision_runs_no_nans_and_conserves_energy_single_species():
    sim = _setup_sim()
    # Only collide species 0 with itself
    coll = _setup_intra_collision(sim, ispec=0, lnLambda=2.0)

    _assert_no_nans_particles_on_patches(sim, 0)
    _assert_no_nans_particles_on_patches(sim, 1)

    # Only species 0 should be affected, and its total energy should be conserved
    E0_before = _total_energy(sim, [0])

    dt = 1e-12
    coll(dt)

    _assert_no_nans_particles_on_patches(sim, 0)
    _assert_no_nans_particles_on_patches(sim, 1)

    E0_after = _total_energy(sim, [0])
    assert np.isclose(E0_before, E0_after, rtol=1e-10, atol=0.0)


def test_intra_collision_alters_only_target_species():
    sim = _setup_sim()
    # Only collide species 0 with itself
    coll = _setup_intra_collision(sim, ispec=0, lnLambda=2.0)

    # Snapshot both species' momenta
    ux0 = [p.particles[0].ux.copy() for p in sim.patches]
    uy0 = [p.particles[0].uy.copy() for p in sim.patches]
    uz0 = [p.particles[0].uz.copy() for p in sim.patches]

    ux1 = [p.particles[1].ux.copy() for p in sim.patches]
    uy1 = [p.particles[1].uy.copy() for p in sim.patches]
    uz1 = [p.particles[1].uz.copy() for p in sim.patches]

    # Advance intra collisions for species 0 only
    coll(1e-12)

    # Species 0 should change
    changed0 = False
    for ip, p in enumerate(sim.patches):
        if not np.allclose(p.particles[0].ux, ux0[ip]) or \
           not np.allclose(p.particles[0].uy, uy0[ip]) or \
           not np.allclose(p.particles[0].uz, uz0[ip]):
            changed0 = True
            break
    assert changed0, "Expected intra collisions to alter species 0 momentum"

    # Species 1 should remain unchanged (not part of collision group)
    for ip, p in enumerate(sim.patches):
        assert np.allclose(p.particles[1].ux, ux1[ip])
        assert np.allclose(p.particles[1].uy, uy1[ip])
        assert np.allclose(p.particles[1].uz, uz1[ip])


def test_inter_collision_errors_on_mismatched_reverse_x():
    """Regression for the flaky inter-species collision test introduced by
    the perf/sort-drift-symmetry merge.

    The merge added ``_decide_reverse_x``, which mirrors the x bucket
    numbering per-species based on its w-weighted mean ux. Inter-species
    collisions iterate both species' bucket bounds in lockstep, so a
    mismatched ``reverse_x`` between the two species pairs every bucket
    against an empty one and the collision step silently becomes a no-op.

    The Collision module must reject this misconfiguration with a clear
    error rather than silently doing nothing.
    """
    # Build a sim WITHOUT add_collision so sorters take the 1D-bucket path
    # (the misconfigured path that _decide_reverse_x can break).
    sim = Simulation(
        nx=16, ny=16, dx=1e-6, dy=1e-6,
        npatch_x=2, npatch_y=2, random_seed=1234,
    )
    e1 = Electron(density=lambda x, y: 1.0e29, ppc=100)
    e2 = Electron(density=lambda x, y: 1.0e29, ppc=100)
    sim.add_species([e1, e2])
    sim.initialize()
    for sorter in sim.sorter:
        sorter()

    # Force the exact mismatched state that _decide_reverse_x can produce
    # for 1D-bucket sorters with opposite mean-ux signs.
    sim.sorter[0].reverse_x = True
    sim.sorter[0]._reverse_decided = True
    sim.sorter[1].reverse_x = False
    sim.sorter[1]._reverse_decided = True

    species = sim.patches.species
    groups = [[species[0], species[1]]]
    coll = Collision(groups, sim.patches, sim.sorter, sim.rand_gen)
    coll.generate_particle_lists()
    with pytest.raises(ValueError, match="reverse_x"):
        coll.generate_field_lists()
