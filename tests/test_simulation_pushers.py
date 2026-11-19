"""Simulation pusher wiring: unsupported pusher configs must fail at init.

A constructed species instance can carry any ``pusher`` string if the
attribute is assigned after construction. The ``Simulation`` only
implements ``'boris'``, ``'photon'`` and ``'boris+tbmt'``; an unsupported
value used to be silently skipped by ``_init_pushers``, leaving a pusher
list shorter than the species list: every later species' pusher shifted
onto a lower index (wrong ``q``/``m`` and particle arrays bound in
``PusherBase.__init__``) and ``run()`` failed deep inside the timestep
loop with an ``IndexError`` -- or worse, pushed the wrong species. The
misaligned list must never be constructed: ``initialize()`` fails fast
naming the species and the invalid pusher, and a failed init leaves no
pushers and no modified species behind.
"""
from __future__ import annotations

import pytest

from lambdapic import Electron, Photon, Simulation
from lambdapic.core.pusher.pusher import (
    BorisPusher,
    BorisTBMTPusher,
    PhotonPusher,
)


@pytest.fixture(autouse=True)
def _isolate_species_registry(isolated_registry):
    """Keep this module's species out of the global ``_ALL_SPECIES`` registry.

    The rejection tests register species the simulation refuses to wire;
    leaking them into the shared registry breaks unrelated species-less
    simulations on the same xdist worker (they auto-register every
    polluted species at init).
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_sim(species_list, **sim_kwargs) -> Simulation:
    """Small 2D simulation with the given species, not yet initialized."""
    kwargs: dict = dict(nx=16, ny=16, dx=0.1, dy=0.1,
                        npatch_x=2, npatch_y=2, dt_cfl=0.95)
    kwargs.update(sim_kwargs)
    sim = Simulation(**kwargs)
    sim.add_species(species_list)
    return sim


# ---------------------------------------------------------------------------
# 1. Unsupported pushers are rejected at initialize()
# ---------------------------------------------------------------------------
class TestUnsupportedPusherRejected:
    """The Simulation rejects pushers it cannot dispatch."""

    def test_unknown_pusher_rejected_at_init(self):
        """A pusher mutated after species construction has no validator left.

        SpeciesConfig Literal validation only runs in ``__post_init__``;
        assigning ``ele.pusher = 'verlet'`` afterwards leaves
        ``_init_pushers`` as the last line of defense.
        """
        ele = Electron(name='drift_electron',
                       density=lambda x, y: 1.0, ppc=4)
        ele.pusher = 'verlet'
        sim = _make_sim([ele])
        with pytest.raises(ValueError, match=(
                r"Species 'drift_electron'.*pusher='verlet'"
        )):
            sim.initialize()

    def test_tbmt_without_polarization_rejected_at_init(self):
        """The spin pusher requires a polarization vector."""
        ele = Electron(name='spinless_electron',
                       density=lambda x, y: 1.0, ppc=4,
                       pusher='boris+tbmt')  # no polarization
        sim = _make_sim([ele])
        with pytest.raises(ValueError, match="polarization"):
            sim.initialize()


# ---------------------------------------------------------------------------
# 2. Failure atomicity: no pushers, no species changes left behind
# ---------------------------------------------------------------------------
class TestInitFailureAtomicity:
    """A rejected config must not leave pushers or species changes behind."""

    def test_failed_init_leaves_no_pusher_list(self):
        """No partially constructed pusher list survives a failed init.

        The pusher list must either be complete or absent: a list missing
        entries is exactly the species-index misalignment bug.
        """
        ok = Electron(name='boris_electron',
                      density=lambda x, y: 1.0, ppc=4)
        bad = Electron(name='unwired_electron',
                       density=lambda x, y: 1.0, ppc=4)
        bad.pusher = 'verlet'
        sim = _make_sim([ok, bad])
        with pytest.raises(ValueError, match="pusher='verlet'"):
            sim.initialize()
        assert not hasattr(sim, 'pusher')

    def test_failed_init_leaves_species_unmodified(self):
        """The boris->TBMT auto-upgrade must not run before validation.

        The first species would be upgraded (and its config mutated) if
        construction had started; validation runs first, so a later
        species' failure leaves it untouched.
        """
        upgradable = Electron(name='spin_electron',
                              density=lambda x, y: 1.0, ppc=4,
                              polarization=(1.0, 0.0, 0.0))
        bad = Electron(name='unwired_electron',
                       density=lambda x, y: 1.0, ppc=4)
        bad.pusher = 'verlet'
        sim = _make_sim([upgradable, bad])
        with pytest.raises(ValueError, match="pusher='verlet'"):
            sim.initialize()
        assert upgradable.pusher == 'boris'

    def test_tbmt_without_polarization_rejected_atomically(self):
        """The TBMT check keeps its message and gains atomicity."""
        ok = Electron(name='spin_electron',
                      density=lambda x, y: 1.0, ppc=4,
                      polarization=(1.0, 0.0, 0.0))
        bad = Electron(name='spinless_electron',
                       density=lambda x, y: 1.0, ppc=4,
                       pusher='boris+tbmt')  # no polarization
        sim = _make_sim([ok, bad])
        with pytest.raises(ValueError, match="polarization"):
            sim.initialize()
        assert not hasattr(sim, 'pusher')
        assert ok.pusher == 'boris'


# ---------------------------------------------------------------------------
# 3. Legal configs: exact pusher-to-species alignment, no regression
# ---------------------------------------------------------------------------
class TestLegalPusherWiring:
    """Supported configs keep their exact pusher-to-species alignment."""

    def test_pusher_list_aligns_with_species(self):
        """Boris, photon and TBMT species each get their own pusher.

        The pusher index must equal the species index: PusherBase binds
        q/m and the particle arrays of ``patches.species[ispec]``.
        """
        boris_ele = Electron(name='boris_electron',
                             density=lambda x, y: 1.0, ppc=4)
        pho = Photon()
        spin_ele = Electron(name='spin_electron',
                            density=lambda x, y: 1.0, ppc=4,
                            polarization=(1.0, 0.0, 0.0))
        sim = _make_sim([boris_ele, pho, spin_ele])
        sim.initialize()

        assert len(sim.pusher) == len(sim.patches.species) == 3
        assert isinstance(sim.pusher[0], BorisPusher)
        assert isinstance(sim.pusher[1], PhotonPusher)
        assert isinstance(sim.pusher[2], BorisTBMTPusher)
        for ispec, pusher in enumerate(sim.pusher):
            assert pusher.ispec == ispec
            assert pusher.q == sim.patches.species[ispec].q
            assert pusher.m == sim.patches.species[ispec].m
        assert sim.patches.species[0].pusher == 'boris'
        assert sim.patches.species[2].pusher == 'boris+tbmt'
