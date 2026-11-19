"""Automatic reconcile of particle-array mutations across run() (issue #58).

Particle-mutating callbacks at late stages (``qed_create_particles`` and
later) must not need to call :meth:`Simulation.update_lists` manually:
the next step's sorter, pusher and QED modules must consume the live
arrays. Likewise a Simulation restored from a dill pickle must rebind its
module particle lists before the first step consumes them.
"""

import tempfile
from pathlib import Path

import dill
import numpy as np
import pytest
from scipy.constants import c, e, epsilon_0, m_e, pi

from lambdapic import Electron, Photon, Simulation
from lambdapic.callback.callback import callback
from lambdapic.callback.injector import ParticleInjector
from lambdapic.callback.restart import RestartDump

l0 = 0.8e-6
omega0 = 2 * pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2

EZ = 1e12  # uniform out-of-plane E: one Boris push kicks uz by q*EZ*dt/(m*c)


def _make_sim() -> Simulation:
    """2D periodic sim with a radiating electron + photon species.

    The radiation module holds typed particle lists (and disables the
    unified pusher), making the pusher/QED paths the typed-list-sensitive
    consumers exercised by the reconcile tests.
    """
    sim = Simulation(
        nx=32,
        ny=32,
        dx=l0 / 20,
        dy=l0 / 20,
        npatch_x=2,
        npatch_y=2,
        boundary_conditions={
            "xmin": "periodic",
            "xmax": "periodic",
            "ymin": "periodic",
            "ymax": "periodic",
        },
        random_seed=42,
    )
    ele = Electron(density=lambda x, y: 0.01 * nc, ppc=2, radiation="photons")
    pho = Photon()
    ele.set_photon(pho)
    sim.add_species([ele, pho])
    sim.initialize()
    return sim


def _freshness_errors(sim: Simulation) -> list[str]:
    """Report module particle lists that no longer reference live arrays.

    Covers the run loop's consumers: the pusher and QED typed lists
    (detached by extend()/prune() and by deserialization) and the
    sorter's plain lists (detached by deserialization).
    """
    errors: list[str] = []
    for ispec in range(len(sim.patches.species)):
        pusher = sim.pusher[ispec]
        sorter = sim.sorter[ispec]
        rad = sim.radiation[ispec]
        for ipatch, p in enumerate(sim.patches):
            part = p.particles[ispec]
            if pusher.ux_list[ipatch] is not part.ux:
                errors.append(
                    f"itime={sim.itime} species={ispec} patch={ipatch}: "
                    "pusher.ux_list stale"
                )
            if sorter.x_list[ipatch] is not part.x:
                errors.append(
                    f"itime={sim.itime} species={ispec} patch={ipatch}: "
                    "sorter.x_list stale"
                )
            if rad is not None:
                if rad.ux_list[ipatch] is not part.ux:
                    errors.append(
                        f"itime={sim.itime} species={ispec} patch={ipatch}: "
                        "radiation.ux_list stale"
                    )
                photon = p.particles[rad.photon_ispec]
                if rad.ux_pho_list[ipatch] is not photon.ux:
                    errors.append(
                        f"itime={sim.itime} species={ispec} patch={ipatch}: "
                        "radiation.ux_pho_list stale"
                    )
    return errors


def _freshness_probe():
    """Callback asserting list freshness right before the consumers.

    Raises inside the run loop (single-rank re-raise is clean) so the
    regression fails at the invariant itself, before the stale lists
    poison any numba kernel.
    """

    @callback(stage="maxwell_1", interval=1)
    def probe(sim: Simulation) -> None:
        errors = _freshness_errors(sim)
        assert not errors, "stale module particle lists: " + "; ".join(errors[:6])

    return probe


def _fill_extension(part, n: int) -> None:
    """Make the n freshly extended slots of ``part`` live, at rest."""
    part.x[-n:] = part.x[0]
    part.y[-n:] = part.y[0]
    part.z[-n:] = 0.0
    part.w[-n:] = part.w[0]
    part.ux[-n:] = 0.0
    part.uy[-n:] = 0.0
    part.uz[-n:] = 0.0
    part.inv_gamma[-n:] = 1.0
    part.chi[-n:] = 0.0
    part.tau[-n:] = part.tau[0]
    part.delta[-n:] = part.delta[0]
    part.is_dead[-n:] = False


@pytest.mark.unit
class TestLateStageExtension:
    """Late-stage particle extension must reconcile before the next step."""

    def test_next_step_consumers_use_the_new_arrays(self) -> None:
        sim = _make_sim()
        npart0 = sim.patches[0].particles[0].npart

        @callback(stage="qed_create_particles", interval=lambda sim: sim.itime == 0)
        def extend(sim: Simulation) -> None:
            part = sim.patches[0].particles[0]
            part.extend(4)
            _fill_extension(part, 4)

        sim.run(nsteps=3, callbacks=[extend, _freshness_probe()])

        assert sim.patches[0].particles[0].npart == npart0 + 4

    def test_new_particles_are_pushed_by_the_next_step(self) -> None:
        sim = _make_sim()
        for p in sim.patches:
            p.fields.ez[:] = EZ

        tail_ids: list[np.ndarray] = []

        @callback(stage="qed_create_particles", interval=lambda sim: sim.itime == 0)
        def extend(sim: Simulation) -> None:
            part = sim.patches[0].particles[0]
            part.extend(4)
            _fill_extension(part, 4)
            tail_ids.append(part._id[-4:].copy())

        sim.run(nsteps=2, callbacks=[extend])

        ele = sim.species[0]
        du = ele.q * EZ * sim.dt / (ele.m * c)

        part = sim.patches[0].particles[0]
        tail = np.isin(part._id, tail_ids[0])
        assert tail.sum() == 4
        orig = ~tail & ~part.is_dead
        tail_uz = part.uz[tail]
        orig_uz = part.uz[orig]
        # Injected slots receive one kick; original electrons receive two.
        assert np.all(np.abs(tail_uz) > 0.5 * abs(du))
        assert np.allclose(orig_uz, 2 * tail_uz[0], rtol=1e-3)


@pytest.mark.unit
class TestDeserializedSimulation:
    """A dill-restored Simulation must rebind its module lists before running."""

    def _assert_pushed_once(self, sim: Simulation) -> None:
        """All alive electrons carry exactly one Boris kick from EZ."""
        ele = sim.species[0]
        du = ele.q * EZ * sim.dt / (ele.m * c)
        for p in sim.patches:
            part = p.particles[0]
            assert np.allclose(part.uz[~part.is_dead], du, rtol=1e-4)

    def test_dill_roundtrip_first_step_reconciles(self) -> None:
        sim = _make_sim()
        sim.run(nsteps=1)  # move the state off initialization

        sim2 = dill.loads(dill.dumps(sim))
        for p in sim2.patches:
            p.fields.ez[:] = EZ

        sim2.run(nsteps=1, callbacks=[_freshness_probe()])

        self._assert_pushed_once(sim2)

    def test_restart_dump_load_leaves_reconciled_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sim = _make_sim()
            ckpt = RestartDump(tmp, interval=1)
            sim.run(nsteps=1, callbacks=[ckpt])

            # the 'end'-stage dump happens before the loop's itime
            # increment, so the checkpoint of the last run step is
            # itime-1 (here: the only directory written)
            ckpt_dir = sorted(Path(tmp).glob("ckpt_*"))[-1]

            loaded = RestartDump.load(ckpt_dir)

        assert _freshness_errors(loaded) == []

        for p in loaded.patches:
            p.fields.ez[:] = EZ

        loaded.run(nsteps=1, callbacks=[_freshness_probe()])

        self._assert_pushed_once(loaded)


@pytest.mark.unit
class TestExplicitRefreshIdempotence:
    """Existing explicit update_lists() refreshes keep working (idempotent)."""

    def test_particle_injector_stays_reconciled(self) -> None:
        sim = _make_sim()
        n_alive0 = sum(
            int(p.particles[0].is_alive.sum()) for p in sim.patches
        )

        injector = ParticleInjector(
            sim.species[0], side="xmin", density=0.1 * nc, ux=3.0, ppc=4
        )

        sim.run(nsteps=3, callbacks=[injector, _freshness_probe()])

        n_alive = sum(int(p.particles[0].is_alive.sum()) for p in sim.patches)
        assert n_alive > n_alive0
