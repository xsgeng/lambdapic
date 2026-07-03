import pytest

from lambdapic import Simulation
from lambdapic.core.species import Electron, Proton, _ALL_SPECIES


@pytest.fixture
def isolated_registry():
    """Snapshot and isolate the global _ALL_SPECIES registry for the test."""
    saved = list(_ALL_SPECIES)
    _ALL_SPECIES.clear()
    yield _ALL_SPECIES
    _ALL_SPECIES.clear()
    _ALL_SPECIES.extend(saved)


def _make_sim():
    return Simulation(
        nx=32,
        ny=32,
        dx=1e-7,
        dy=1e-7,
        npatch_x=2,
        npatch_y=2,
        dt_cfl=0.95,
    )


def test_run_auto_registers_species(isolated_registry):
    ele = Electron(density=lambda x, y: 1e25, ppc=4)

    assert _ALL_SPECIES == [ele]
    assert not ele._ispec == 0 or ele._ispec is None

    sim = _make_sim()
    assert sim.species == []

    sim.run(nsteps=1)

    assert len(sim.species) == 1
    assert sim.species[0] is ele
    assert ele.ispec == 0
    assert sim.patches.species == [ele]


def test_initialize_auto_registers_species(isolated_registry):
    ele = Electron(density=lambda x, y: 1e25, ppc=4)
    proton = Proton(density=lambda x, y: 1e25, ppc=4)

    sim = _make_sim()
    sim.initialize()

    assert len(sim.species) == 2
    assert sim.species[0] is ele
    assert sim.species[1] is proton
    assert ele.ispec == 0
    assert proton.ispec == 1


def test_explicit_add_species_takes_precedence(isolated_registry):
    leaked = Electron(density=lambda x, y: 1e25, ppc=4)

    explicit = Proton(density=lambda x, y: 1e25, ppc=4)

    sim = _make_sim()
    sim.add_species([explicit])

    sim.run(nsteps=1)

    assert sim.species == [explicit]
    assert leaked not in sim.species


def test_no_species_no_registry_is_noop(isolated_registry):
    sim = _make_sim()
    sim.initialize()

    assert sim.species == []


def test_dimension_mismatch_filtered(isolated_registry):
    ele3d = Electron(density=lambda x, y, z: 1e25, ppc=4)

    sim = _make_sim()
    sim.initialize()

    assert ele3d not in sim.species
    assert sim.species == []
