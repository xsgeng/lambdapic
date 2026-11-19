"""Tests for third-party typed.List registration and lifecycle management."""
import importlib.util
import sys
from pathlib import Path

import dill as pickle
import numpy as np
import pytest
from numba import typed
from scipy.constants import c, e, epsilon_0, m_e, pi

from lambdapic import (
    Electron,
    Proton,
    Simulation,
    TypedListIssueKind,
    TypedListRegistration,
    TypedListRegistry,
)

l0 = 0.8e-6
omega0 = 2 * pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2

PERIODIC_2D = {
    "xmin": "periodic",
    "xmax": "periodic",
    "ymin": "periodic",
    "ymax": "periodic",
}


def _make_sim(nspecies: int = 2):
    """Real 2D simulation with two kinetic species, 4x4 periodic patches."""
    sim = Simulation(
        nx=64,
        ny=64,
        dx=l0 / 20,
        dy=l0 / 20,
        npatch_x=4,
        npatch_y=4,
        boundary_conditions=PERIODIC_2D,
        random_seed=42,
    )
    ele = Electron(density=lambda x, y: 0.1 * nc, ppc=4)
    ion = Proton(density=lambda x, y: 0.1 * nc, ppc=2)
    species = [ele, ion][:nspecies]
    sim.add_species(species)
    sim.initialize()
    return sim, species


class _ParticleHolder:
    """Minimal third-party holder mirroring particle arrays."""

    def __init__(self):
        self.observations = []

    def observe(self, ispec):
        """Record npart as seen through the mirrors (first patch)."""
        self.observations.append(self.x_list[ispec][0].size)


class _FieldHolder:
    """Minimal third-party holder mirroring field arrays."""

    pass


class _BypassedHolder:
    """Holder that mutates patch arrays outside the registry contract."""

    pass


@pytest.mark.unit
class TestRegistration:
    """Acceptance 1: public registration interface and its validation."""

    @pytest.fixture(autouse=True)
    def _isolated(self, isolated_registry):
        """Keep the global Species registry untouched across these tests."""
        yield

    def test_particle_holder_references_live_arrays(self):
        sim, (ele, ion) = _make_sim()
        holder = _ParticleHolder()

        reg = sim.register_typed_list_holder(
            holder,
            particle_lists={"x_list": "x", "ux_list": "ux"},
            species=(ele,),
        )

        assert isinstance(reg, TypedListRegistration)
        assert reg.holder is holder
        assert reg.refreshes_particles is True
        assert reg.refreshes_fields is False
        assert reg.name

        ispec = ele.ispec
        for ipatch, p in enumerate(sim.patches):
            assert holder.x_list[ispec][ipatch] is p.particles[ispec].x
            assert holder.ux_list[ispec][ipatch] is p.particles[ispec].ux

    def test_field_holder_references_live_arrays(self):
        sim, (ele, ion) = _make_sim()
        holder = _FieldHolder()

        reg = sim.register_typed_list_holder(
            holder,
            field_lists={"ex_list": "ex", "jz_list": "jz"},
        )

        assert reg.refreshes_particles is False
        assert reg.refreshes_fields is True
        assert isinstance(holder.ex_list, typed.List)
        for ipatch, p in enumerate(sim.patches):
            assert holder.ex_list[ipatch] is p.fields.ex
            assert holder.jz_list[ipatch] is p.fields.jz

    def test_species_watch_accepts_objects_names_and_indices(self):
        sim, (ele, ion) = _make_sim()
        holder = _ParticleHolder()

        reg = sim.register_typed_list_holder(
            holder,
            particle_lists={"x_list": "x"},
            species=(ele, ion.name, ion.ispec),
        )

        assert reg.species == (ele.name, ion.name)
        for s in (ele, ion):
            assert s.ispec in holder.x_list

    def test_same_name_replaces_previous_holder(self):
        """Registry lifecycle: re-registration drops the old holder instance."""
        sim, (ele, ion) = _make_sim()
        holder_a = _ParticleHolder()
        holder_b = _ParticleHolder()

        sim.register_typed_list_holder(
            holder_a, particle_lists={"x_list": "x"}, species=(ele,)
        )
        sim.register_typed_list_holder(
            holder_b,
            particle_lists={"x_list": "x"},
            species=(ele,),
            name=_ParticleHolder.__name__,
        )

        registrations = sim.typed_list_registry.registrations
        assert len(registrations) == 1
        assert registrations[0].holder is holder_b

    def test_unregister_releases_holder(self):
        sim, (ele, ion) = _make_sim()
        holder = _ParticleHolder()
        reg = sim.register_typed_list_holder(
            holder, particle_lists={"x_list": "x"}, species=(ele,)
        )

        removed = sim.unregister_typed_list_holder(reg.name)
        assert removed.holder is holder
        assert sim.typed_list_registry.registrations == ()
        assert sim.unregister_typed_list_holder(reg.name) is None

    def test_register_requires_initialized_simulation(self):
        sim = Simulation(
            nx=64, ny=64, dx=l0 / 20, dy=l0 / 20,
            npatch_x=4, npatch_y=4, boundary_conditions=PERIODIC_2D,
        )
        with pytest.raises(RuntimeError, match="initiali"):
            sim.register_typed_list_holder(
                _ParticleHolder(), particle_lists={"x_list": "x"}, species=("electron",)
            )

    def test_unknown_species_is_rejected(self):
        sim, (ele, ion) = _make_sim()
        with pytest.raises(ValueError, match="muon"):
            sim.register_typed_list_holder(
                _ParticleHolder(),
                particle_lists={"x_list": "x"},
                species=("muon",),
            )

    def test_unknown_particle_attribute_is_rejected(self):
        sim, (ele, ion) = _make_sim()
        with pytest.raises(ValueError, match="not_a_particle_attr"):
            sim.register_typed_list_holder(
                _ParticleHolder(),
                particle_lists={"x_list": "not_a_particle_attr"},
                species=(ele,),
            )

    def test_unknown_field_attribute_is_rejected(self):
        sim, (ele, ion) = _make_sim()
        with pytest.raises(ValueError, match="not_a_field_attr"):
            sim.register_typed_list_holder(
                _ParticleHolder(), field_lists={"ex_list": "not_a_field_attr"}
            )

    def test_particle_lists_require_species_watch_set(self):
        sim, (ele, ion) = _make_sim()
        with pytest.raises(ValueError, match="species"):
            sim.register_typed_list_holder(
                _ParticleHolder(), particle_lists={"x_list": "x"}
            )

    def test_species_without_particle_lists_is_rejected(self):
        sim, (ele, ion) = _make_sim()
        with pytest.raises(ValueError, match="particle"):
            sim.register_typed_list_holder(_FieldHolder(), species=(ele,))

    def test_list_name_collision_between_kinds_is_rejected(self):
        sim, (ele, ion) = _make_sim()
        with pytest.raises(ValueError, match="x_list"):
            sim.register_typed_list_holder(
                _ParticleHolder(),
                particle_lists={"x_list": "x"},
                field_lists={"x_list": "ex"},
                species=(ele,),
            )

    def test_shadowing_existing_method_is_rejected(self):
        class _ShadowingHolder:
            def x_list(self):
                pass

        sim, (ele, ion) = _make_sim()
        with pytest.raises(ValueError, match="x_list"):
            sim.register_typed_list_holder(
                _ShadowingHolder(),
                particle_lists={"x_list": "x"},
                species=(ele,),
            )

    def test_non_identifier_list_name_is_rejected(self):
        sim, (ele, ion) = _make_sim()
        with pytest.raises(ValueError, match="identifier"):
            sim.register_typed_list_holder(
                _FieldHolder(), field_lists={"not an identifier": "ex"}
            )

    def test_empty_registration_is_rejected(self):
        sim, (ele, ion) = _make_sim()
        with pytest.raises(ValueError, match="at least one"):
            sim.register_typed_list_holder(_FieldHolder())


@pytest.mark.unit
class TestMutatorRefresh:
    """Acceptance 3 (mutator half): refresh after particle array replacement."""

    @pytest.fixture(autouse=True)
    def _isolated(self, isolated_registry):
        yield

    @pytest.fixture
    def setup(self):
        sim, (ele, ion) = _make_sim()
        holder = _ParticleHolder()
        sim.register_typed_list_holder(
            holder,
            particle_lists={"x_list": "x", "ux_list": "ux"},
            field_lists={"ex_list": "ex"},
            species=(ele,),
        )
        return sim, ele, ion, holder

    def test_extend_refreshes_registered_mirrors(self, setup):
        sim, ele, ion, holder = setup

        for p in sim.patches:
            p.particles[ele.ispec].extend(4)

        # mirrors are stale before update_lists (resize moves the buffer)
        assert holder.x_list[ele.ispec][0] is not sim.patches[0].particles[ele.ispec].x

        sim.update_lists()

        for ipatch, p in enumerate(sim.patches):
            assert holder.x_list[ele.ispec][ipatch] is p.particles[ele.ispec].x
            assert holder.ux_list[ele.ispec][ipatch] is p.particles[ele.ispec].ux
            assert holder.x_list[ele.ispec][ipatch].size == p.particles[ele.ispec].npart
        assert sim.pusher[ele.ispec].x_list[0] is sim.patches[0].particles[ele.ispec].x

    def test_prune_refreshes_registered_mirrors(self, setup):
        sim, ele, ion, holder = setup

        for p in sim.patches:
            particles = p.particles[ele.ispec]
            particles.is_dead[::2] = True
            particles.prune()

        sim.update_lists()

        for ipatch, p in enumerate(sim.patches):
            assert holder.x_list[ele.ispec][ipatch] is p.particles[ele.ispec].x
            assert holder.x_list[ele.ispec][ipatch].size == p.particles[ele.ispec].npart

    def test_field_mirrors_survive_particle_mutators(self, setup):
        sim, ele, ion, holder = setup

        for p in sim.patches:
            p.particles[ele.ispec].extend(2)
        sim.update_lists()

        for ipatch, p in enumerate(sim.patches):
            assert holder.ex_list[ipatch] is p.fields.ex

    def test_extension_without_refresh_is_diagnosed(self, setup):
        sim, ele, ion, holder = setup

        for p in sim.patches:
            p.particles[ele.ispec].extend(4)
        # deliberately bypass update_lists: the mirrors are now stale

        issues = sim.check_typed_list_freshness()
        kinds = {i.kind for i in issues}
        assert TypedListIssueKind.STALE_PARTICLE in kinds
        stale = [i for i in issues if i.kind is TypedListIssueKind.STALE_PARTICLE]
        assert all(i.name == holder.__class__.__name__ for i in stale)
        assert any(i.holder_attr == "x_list" for i in stale)
        assert any(i.species == ele.name for i in stale)
        assert any(i.ipatch == 0 for i in stale)

        sim.update_lists()
        assert sim.check_typed_list_freshness() == []


@pytest.mark.unit
class TestPatchRebuild:
    """Acceptance 3 (rebuild half): recover after patch rebuild."""

    @pytest.fixture(autouse=True)
    def _isolated(self, isolated_registry):
        yield

    def test_rebuild_regenerates_particle_and_field_mirrors(self):
        sim, (ele, ion) = _make_sim()
        holder = _ParticleHolder()
        sim.register_typed_list_holder(
            holder,
            particle_lists={"x_list": "x"},
            field_lists={"ex_list": "ex"},
            species=(ele,),
        )

        # emulate what MovingWindow._fill_particles and a load-balance
        # migration do: replace particle and field array objects outright
        for p in sim.patches:
            p.particles[ele.ispec].initialize(p.particles[ele.ispec].npart)
            p.fields.ex = np.zeros_like(p.fields.ex)

        assert holder.x_list[ele.ispec][0] is not sim.patches[0].particles[ele.ispec].x
        assert holder.ex_list[0] is not sim.patches[0].fields.ex

        sim.update_patches()

        for ipatch, p in enumerate(sim.patches):
            assert holder.x_list[ele.ispec][ipatch] is p.particles[ele.ispec].x
            assert holder.ex_list[ipatch] is p.fields.ex
        assert sim.check_typed_list_freshness() == []

    def test_rebuild_recovers_after_moving_window_style_fill(self):
        sim, (ele, ion) = _make_sim()
        holder = _ParticleHolder()
        sim.register_typed_list_holder(
            holder, particle_lists={"x_list": "x", "ux_list": "ux"}, species=(ele,)
        )

        # same replacement MovingWindow._fill_particles performs on the
        # wrapped patches, followed by the sim.update_patches() it calls
        for p in sim.patches:
            p.particles[ele.ispec].initialize(p.particles[ele.ispec].npart)
        sim.update_patches()

        for ipatch, p in enumerate(sim.patches):
            assert holder.x_list[ele.ispec][ipatch] is p.particles[ele.ispec].x
            assert holder.ux_list[ele.ispec][ipatch] is p.particles[ele.ispec].ux


@pytest.mark.unit
class TestDiagnostics:
    """Acceptance 4: unregistered or wrongly-declared holders get diagnosed."""

    @pytest.fixture(autouse=True)
    def _isolated(self, isolated_registry):
        yield

    def test_check_empty_when_all_fresh(self):
        sim, (ele, ion) = _make_sim()
        holder = _ParticleHolder()
        sim.register_typed_list_holder(
            holder,
            particle_lists={"x_list": "x"},
            field_lists={"ex_list": "ex"},
            species=(ele,),
        )
        assert sim.check_typed_list_freshness() == []

    def test_unregistered_holder_reported_from_callbacks(self):
        sim, (ele, ion) = _make_sim()
        unregistered = _ParticleHolder()
        unregistered.x_list = {
            ele.ispec: typed.List([p.particles[ele.ispec].x for p in sim.patches])
        }

        issues = sim.check_typed_list_freshness(callbacks=[unregistered])
        matches = [
            i for i in issues
            if i.kind is TypedListIssueKind.UNREGISTERED_HOLDER
            and i.holder_attr == "x_list"
        ]
        assert matches, f"expected unregistered-holder issue, got {issues}"
        assert _ParticleHolder.__name__ in matches[0].message

    def test_registered_holder_not_flagged_by_callback_scan(self):
        sim, (ele, ion) = _make_sim()
        holder = _ParticleHolder()
        sim.register_typed_list_holder(
            holder, particle_lists={"x_list": "x"}, species=(ele,)
        )

        issues = sim.check_typed_list_freshness(callbacks=[holder])
        assert issues == []

    def test_holder_owned_scratch_lists_not_flagged(self):
        sim, (ele, ion) = _make_sim()
        holder = _FieldHolder()
        holder.scratch = typed.List(
            [np.zeros((sim.nx_per_patch, sim.ny_per_patch)) for _ in sim.patches]
        )

        issues = sim.check_typed_list_freshness(callbacks=[holder])
        assert [
            i for i in issues if i.kind is TypedListIssueKind.UNREGISTERED_HOLDER
        ] == []

    def test_raw_array_replacement_is_diagnosed(self):
        sim, (ele, ion) = _make_sim()
        holder = _ParticleHolder()
        sim.register_typed_list_holder(
            holder, particle_lists={"x_list": "x"}, species=(ele,)
        )

        # bypass the mutator contract: replace the array object outright,
        # without the extended flag and without update_lists
        p0 = sim.patches[0]
        p0.particles[ele.ispec].x = np.array(p0.particles[ele.ispec].x)

        issues = sim.check_typed_list_freshness()
        stale = [i for i in issues if i.kind is TypedListIssueKind.STALE_PARTICLE]
        assert any(i.ipatch == 0 and i.holder_attr == "x_list" for i in stale)


@pytest.mark.unit
class TestCheckpointContract:
    """Acceptance 4 (checkpoint half): declarations survive, holders do not."""

    @pytest.fixture(autouse=True)
    def _isolated(self, isolated_registry):
        yield

    def test_declarations_survive_dump_and_holders_are_dropped(self):
        sim, (ele, ion) = _make_sim()
        holder = _ParticleHolder()
        reg = sim.register_typed_list_holder(
            holder,
            particle_lists={"x_list": "x", "ux_list": "ux"},
            species=(ele.name,),
        )

        blob = pickle.dumps(sim, byref=True, recurse=True)
        restored = pickle.loads(blob)

        registrations = restored.typed_list_registry.registrations
        assert len(registrations) == 1
        assert registrations[0].name == reg.name
        assert registrations[0].holder is None
        assert registrations[0].species == (ele.name,)
        assert registrations[0].particle_lists == reg.particle_lists

        issues = restored.check_typed_list_freshness()
        kinds = {i.kind for i in issues}
        assert TypedListIssueKind.UNBOUND_REGISTRATION in kinds

    def test_reregistration_after_restore_rebuilds_mirrors(self):
        sim, (ele, ion) = _make_sim()
        holder = _ParticleHolder()
        name = sim.register_typed_list_holder(
            holder, particle_lists={"x_list": "x"}, species=(ele.name,)
        ).name

        blob = pickle.dumps(sim, byref=True, recurse=True)
        restored = pickle.loads(blob)

        fresh_holder = _ParticleHolder()
        restored.register_typed_list_holder(
            fresh_holder,
            particle_lists={"x_list": "x"},
            species=(ele.name,),
            name=name,
        )

        ispec = ele.ispec
        for ipatch, p in enumerate(restored.patches):
            assert fresh_holder.x_list[ispec][ipatch] is p.particles[ispec].x

        for p in restored.patches:
            p.particles[ispec].extend(3)
        restored.update_lists()
        for ipatch, p in enumerate(restored.patches):
            assert fresh_holder.x_list[ispec][ipatch] is p.particles[ispec].x
        assert restored.check_typed_list_freshness() == []

    def test_old_checkpoint_without_registry_attribute_recovers(self):
        """A pre-registry checkpoint lacks ``_typed_list_registry``: lazily recreate."""
        sim, (ele, ion) = _make_sim()
        holder = _ParticleHolder()
        sim.register_typed_list_holder(
            holder, particle_lists={"x_list": "x"}, species=(ele,)
        )

        del sim.__dict__["_typed_list_registry"]

        registry = sim.typed_list_registry
        assert isinstance(registry, TypedListRegistry)
        assert registry.registrations == ()
        sim.update_lists()
        sim.update_patches()

    def test_restartdump_roundtrip_with_registered_holder(self, tmp_path):
        from lambdapic.callback.restart import RestartDump

        sim, (ele, ion) = _make_sim()
        holder = _ParticleHolder()
        reg = sim.register_typed_list_holder(
            holder, particle_lists={"x_list": "x"}, species=(ele,)
        )

        dumper = RestartDump(tmp_path, interval=1)
        sim.run(nsteps=1, callbacks=[])
        dumper._call(sim)

        restored = RestartDump.load(dumper._ckpt_dir(sim.itime))

        registrations = restored.typed_list_registry.registrations
        assert len(registrations) == 1
        assert registrations[0].holder is None
        assert registrations[0].name == reg.name

        resumed_holder = _ParticleHolder()
        restored.register_typed_list_holder(
            resumed_holder,
            particle_lists={"x_list": "x"},
            species=(ele,),
            name=reg.name,
        )
        restored.run(nsteps=1, callbacks=[])

        issues = restored.check_typed_list_freshness(callbacks=[resumed_holder])
        assert issues == []
        ispec = ele.ispec
        for ipatch, p in enumerate(restored.patches):
            assert resumed_holder.x_list[ispec][ipatch] is p.particles[ispec].x


@pytest.mark.unit
class TestRunLifecycle:
    """Integration: refresh and diagnostics inside the real run loop."""

    @pytest.fixture(autouse=True)
    def _isolated(self, isolated_registry):
        yield

    def test_run_refreshes_registered_holder_after_late_stage_extend(self):
        from lambdapic.callback.callback import Callback

        sim, (ele,) = _make_sim(nspecies=1)
        ispec = ele.ispec

        class _Extender(Callback):
            stage = "end"

            def __init__(self):
                self.interval = 1

            def _call(self, sim):
                if sim.itime == 0:
                    for p in sim.patches:
                        p.particles[ispec].extend(2)

        class _WatchingHolder(Callback):
            stage = "current_deposition"

            def __init__(self):
                self.interval = 1
                self.seen_npart = []

            def _call(self, sim):
                self.seen_npart.append(self.x_list[ispec][0].size)

        holder = _WatchingHolder()
        sim.register_typed_list_holder(
            holder,
            particle_lists={"x_list": "x", "ux_list": "ux"},
            field_lists={"ex_list": "ex"},
            species=(ele,),
        )
        extender = _Extender()

        npart0 = sim.patches[0].particles[ispec].npart
        sim.run(nsteps=3, callbacks=[extender, holder])

        assert holder.seen_npart == [npart0, npart0 + 2, npart0 + 2]
        assert sim.check_typed_list_freshness(callbacks=[extender, holder]) == []
        for ipatch, p in enumerate(sim.patches):
            assert holder.x_list[ispec][ipatch] is p.particles[ispec].x
            assert holder.ex_list[ipatch] is p.fields.ex

    def test_run_explicit_update_lists_makes_growth_visible_next_step(self):
        """The ParticleInjector pattern: refresh immediately after extending."""
        from lambdapic.callback.callback import Callback

        sim, (ele,) = _make_sim(nspecies=1)
        ispec = ele.ispec

        class _ExtenderWithRefresh(Callback):
            stage = "end"

            def __init__(self):
                self.interval = 1

            def _call(self, sim):
                if sim.itime == 0:
                    for p in sim.patches:
                        p.particles[ispec].extend(2)
                    sim.update_lists()

        class _WatchingHolder(Callback):
            stage = "current_deposition"

            def __init__(self):
                self.interval = 1
                self.seen_npart = []

            def _call(self, sim):
                self.seen_npart.append(self.x_list[ispec][0].size)

        holder = _WatchingHolder()
        sim.register_typed_list_holder(
            holder, particle_lists={"x_list": "x"}, species=(ele,)
        )
        extender = _ExtenderWithRefresh()

        npart0 = sim.patches[0].particles[ispec].npart
        sim.run(nsteps=2, callbacks=[extender, holder])

        assert holder.seen_npart == [npart0, npart0 + 2]
        assert sim.check_typed_list_freshness(callbacks=[extender, holder]) == []


@pytest.fixture(scope="module")
def fluid_hybrid_example():
    """The migrated fluid-hybrid example, imported once per worker process.

    The module is registered in ``sys.modules`` before execution: its
    numba kernels use ``cache=True``, whose cache-load path re-imports the
    defining module by name and fails for unregistered synthetic modules.
    """
    name = "fluid_hybrid_example"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).parents[1] / "example" / "fluid-hybrid.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
class TestExampleMigration:
    """Acceptance 2: the fluid-hybrid example uses the public API."""

    @pytest.fixture(autouse=True)
    def _isolated(self, isolated_registry):
        yield

    def test_fluid_background_registers_via_public_api(self, fluid_hybrid_example):
        example = fluid_hybrid_example

        sim, (ele, ion) = _make_sim()
        fluid_cb = example.FluidBackground()

        assert not hasattr(fluid_cb, "ex_list")

        sim.run(nsteps=2, callbacks=[fluid_cb])

        registrations = sim.typed_list_registry.registrations
        assert len(registrations) == 1
        assert registrations[0].holder is fluid_cb
        assert registrations[0].refreshes_fields is True

        for ipatch, p in enumerate(sim.patches):
            assert fluid_cb.ex_list[ipatch] is p.fields.ex
            assert fluid_cb.jz_list[ipatch] is p.fields.jz
        assert sim.check_typed_list_freshness(callbacks=[fluid_cb]) == []

    def test_fluid_background_recovers_after_patch_rebuild(self, fluid_hybrid_example):
        example = fluid_hybrid_example

        sim, (ele, ion) = _make_sim()
        fluid_cb = example.FluidBackground()
        sim.run(nsteps=1, callbacks=[fluid_cb])

        # emulate a load-balance migration: field arrays replaced wholesale
        for p in sim.patches:
            for attr in p.fields.attrs:
                setattr(p.fields, attr, np.zeros_like(getattr(p.fields, attr)))
        sim.update_patches()

        for ipatch, p in enumerate(sim.patches):
            assert fluid_cb.ex_list[ipatch] is p.fields.ex
            assert fluid_cb.jz_list[ipatch] is p.fields.jz
        assert sim.check_typed_list_freshness(callbacks=[fluid_cb]) == []
