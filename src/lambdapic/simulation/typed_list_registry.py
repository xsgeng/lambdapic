"""Manage third-party ``numba.typed.List`` lifecycles.

Built-in modules keep per-patch particle and field arrays in typed lists.
Those lists must be refreshed after particle mutators replace arrays and
rebuilt after patch migration. :class:`TypedListRegistry` provides the same
lifecycle management to third-party callbacks through
:meth:`lambdapic.simulation.Simulation.register_typed_list_holder`.

Each registration declares particle and/or field list specifications and a
species watch set. The declaration drives generation, mutator refresh, and
freshness diagnostics.

Checkpoint contract
-------------------
Checkpoints preserve registrations but drop holder objects. After restoring,
re-register each holder under the same name to bind it to the restored arrays.
Unbound registrations are reported by :meth:`TypedListRegistry.check`.
Checkpoints written before the registry existed restore as an empty registry.

Examples
--------
>>> sim.register_typed_list_holder(
...     kicker,
...     particle_lists={"x_list": "x", "ux_list": "ux"},
...     species=(ele,),
... )
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numba import typed

from ..core.patch import Patches
from ..core.species import Species
from ..core.utils.logger import logger


class TypedListIssueKind(str, Enum):
    """Kinds of diagnostics reported by :meth:`TypedListRegistry.check`."""

    STALE_PARTICLE = "stale_particle"
    STALE_FIELD = "stale_field"
    UNBOUND_REGISTRATION = "unbound_registration"
    UNREGISTERED_HOLDER = "unregistered_holder"
    WATCHED_SPECIES_MISSING = "watched_species_missing"


@dataclass(frozen=True)
class ParticleListSpec:
    """Declare one per-patch particle-array typed list.

    Attributes:
        holder_attr: Holder attribute receiving ``dict[int, typed.List]``,
            keyed by species index.
        particle_attr: Particle attribute mirrored per patch, e.g. ``"x"``.
    """

    holder_attr: str
    particle_attr: str


@dataclass(frozen=True)
class FieldListSpec:
    """Declare one flat per-patch field-array typed list.

    Attributes:
        holder_attr: Holder attribute receiving the ``typed.List``.
        field_attr: Field attribute mirrored per patch, e.g. ``"ex"``.
    """

    holder_attr: str
    field_attr: str


@dataclass(frozen=True, eq=False)
class TypedListFreshnessIssue:
    """One diagnostic produced by :meth:`TypedListRegistry.check`.

    Attributes:
        kind: Diagnostic category.
        message: Human-readable description with an actionable hint.
        name: Registration name, when applicable.
        holder_attr: Holder attribute involved, if any.
        species: Species name, for particle-list issues.
        ipatch: Patch index, for per-patch issues.
    """

    kind: TypedListIssueKind
    message: str
    name: str | None = None
    holder_attr: str | None = None
    species: str | None = None
    ipatch: int | None = None

    def __str__(self) -> str:
        return f"[{self.kind.value}] {self.message}"


@dataclass(frozen=True, eq=False)
class TypedListRegistration:
    """One third-party typed-list holder declaration.

    Attributes:
        name: Registration key. Re-registration under the same key replaces
            the previous holder.
        holder: Bound holder, or ``None`` after checkpoint restoration.
        particle_lists: Particle list declarations refreshed after mutators
            and patch rebuilds.
        field_lists: Field list declarations rebuilt after patch rebuilds.
        species: Watched species names, resolved against the live simulation.
    """

    name: str
    holder: Any | None = None
    particle_lists: tuple[ParticleListSpec, ...] = ()
    field_lists: tuple[FieldListSpec, ...] = ()
    species: tuple[str, ...] = ()

    @property
    def refreshes_particles(self) -> bool:
        """Whether this registration watches particle arrays."""
        return bool(self.particle_lists)

    @property
    def refreshes_fields(self) -> bool:
        """Whether this registration watches field arrays."""
        return bool(self.field_lists)

    def without_holder(self) -> TypedListRegistration:
        """Return the serializable declaration without its holder."""
        return replace(self, holder=None)


class TypedListRegistry:
    """Own and refresh third-party per-patch typed lists.

    The simulation owns the registry. Users normally call
    :meth:`lambdapic.simulation.Simulation.register_typed_list_holder`
    instead of constructing a registry directly. The registry stores no
    patch references; every operation receives the current patch collection.
    """

    def __init__(self) -> None:
        self._registrations: dict[str, TypedListRegistration] = {}

    def __len__(self) -> int:
        return len(self._registrations)

    def __contains__(self, key: str | Any) -> bool:
        return self.get(key) is not None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_registrations"] = {
            name: reg.without_holder() for name, reg in self._registrations.items()
        }
        return state

    @property
    def registrations(self) -> tuple[TypedListRegistration, ...]:
        """Current registrations in registration order."""
        return tuple(self._registrations.values())

    def get(self, key: str | Any) -> TypedListRegistration | None:
        """Look up a registration by name or holder identity."""
        if isinstance(key, str):
            return self._registrations.get(key)
        for reg in self._registrations.values():
            if reg.holder is not None and reg.holder is key:
                return reg
        return None

    def register(
        self,
        patches: Patches,
        holder: Any,
        particle_lists: Mapping[str, str] | None,
        field_lists: Mapping[str, str] | None,
        species: Sequence[Species | str | int] | Species | str | int | None,
        name: str | None,
    ) -> TypedListRegistration:
        """Validate a declaration, register its holder, and build its lists.

        Args:
            patches: Current patch collection of an initialized simulation.
            holder: Object receiving the typed-list attributes.
            particle_lists: Mapping of holder attribute names to particle
                attribute names, e.g. ``{"x_list": "x"}``.
            field_lists: Mapping of holder attribute names to field attribute
                names, e.g. ``{"ex_list": "ex"}``.
            species: Species objects, names, or indices to watch. Required
                when particle lists are declared.
            name: Registration key; defaults to the holder class name.

        Returns:
            The stored registration.

        Raises:
            TypeError: If a declaration entry has the wrong type.
            ValueError: If a species, source attribute, or holder attribute
                declaration is invalid.
        """
        particle_lists = dict(particle_lists or {})
        field_lists = dict(field_lists or {})
        name = name if name else type(holder).__name__

        if not isinstance(name, str) or not name:
            raise TypeError(f"registration name must be a non-empty str, got {name!r}")
        if not particle_lists and not field_lists:
            raise ValueError(
                f"typed-list holder {name!r} declares no lists: declare at "
                "least one particle list or field list"
            )

        self._normalize_list_specs(name, holder, particle_lists, "particle")
        self._normalize_list_specs(name, holder, field_lists, "field")

        for holder_attr in field_lists:
            if holder_attr in particle_lists:
                raise ValueError(
                    f"typed-list holder {name!r}: attribute {holder_attr!r} "
                    "is declared as both a particle list and a field list"
                )

        if particle_lists and not species:
            raise ValueError(
                f"typed-list holder {name!r} declares particle lists but no "
                "species watch set: pass species=(...)"
            )
        if species and not particle_lists:
            raise ValueError(
                f"typed-list holder {name!r} declares a species watch set "
                "without particle lists"
            )

        watch = self._normalize_species(patches, species)
        self._validate_list_sources(name, patches, particle_lists, field_lists, watch)

        registration = TypedListRegistration(
            name=name,
            holder=holder,
            particle_lists=tuple(
                ParticleListSpec(holder_attr, particle_attr)
                for holder_attr, particle_attr in sorted(particle_lists.items())
            ),
            field_lists=tuple(
                FieldListSpec(holder_attr, field_attr)
                for holder_attr, field_attr in sorted(field_lists.items())
            ),
            species=watch,
        )

        previous = self._registrations.get(name)
        self._registrations[name] = registration
        if previous is not None:
            logger.debug(f"typed-list registry: replaced registration {name!r}")
        self._generate_one(registration, patches)
        return registration

    def unregister(self, key: str | Any) -> TypedListRegistration | None:
        """Remove a registration by name or holder identity."""
        if isinstance(key, str):
            return self._registrations.pop(key, None)
        reg = self.get(key)
        if reg is not None:
            del self._registrations[reg.name]
        return reg

    def refresh_particle_lists(self, patches: Patches) -> None:
        """Refresh watched lists for particles flagged ``extended``.

        :meth:`lambdapic.simulation.Simulation.update_lists` calls this before
        clearing the mutator flags.
        """
        if not self._registrations:
            return
        species_names = [s.name for s in patches.species]
        for reg in self._registrations.values():
            holder = reg.holder
            if holder is None or not reg.particle_lists:
                continue
            for ispec in self._resolve_species(reg, species_names):
                for ipatch, patch in enumerate(patches):
                    particles = patch.particles[ispec]
                    if not particles.extended:
                        continue
                    for spec in reg.particle_lists:
                        getattr(holder, spec.holder_attr)[ispec][ipatch] = getattr(
                            particles, spec.particle_attr
                        )

    def generate(self, patches: Patches) -> None:
        """Rebuild every bound holder's lists from current patch arrays."""
        for reg in self._registrations.values():
            if reg.holder is not None:
                self._generate_one(reg, patches)

    def check(
        self,
        patches: Patches,
        callbacks: Sequence[Any] | None = None,
    ) -> list[TypedListFreshnessIssue]:
        """Diagnose stale, unbound, and unregistered typed lists.

        Args:
            patches: Current patch collection.
            callbacks: Optional callbacks to scan for unregistered lists.
                Self-owned scratch lists that do not reference live arrays
                are ignored.

        Returns:
            Structured freshness issues; empty when all checks pass.
        """
        issues: list[TypedListFreshnessIssue] = []
        species_names = [s.name for s in patches.species]

        for reg in self._registrations.values():
            if reg.holder is None:
                issues.append(
                    TypedListFreshnessIssue(
                        kind=TypedListIssueKind.UNBOUND_REGISTRATION,
                        message=(
                            f"typed-list registration {reg.name!r} has no holder "
                            "bound; re-register the holder under the same name"
                        ),
                        name=reg.name,
                    )
                )
                continue
            issues += self._check_registration(reg, patches, species_names)

        if callbacks is not None:
            issues += self.find_unregistered_holders(patches, callbacks)
        return issues

    def find_unregistered_holders(
        self,
        patches: Patches,
        callbacks: Sequence[Any],
    ) -> list[TypedListFreshnessIssue]:
        """Report callbacks holding unregistered references to live arrays.

        Detection is exact for lists built before their first array mutation;
        run-start diagnostics therefore provide the most reliable signal.
        """
        registered = {
            id(reg.holder) for reg in self._registrations.values() if reg.holder is not None
        }
        live_ptrs = self._live_array_pointers(patches)
        issues: list[TypedListFreshnessIssue] = []
        seen: set[int] = set()

        for callback in callbacks:
            if id(callback) in registered or id(callback) in seen:
                continue
            seen.add(id(callback))
            namespace = getattr(callback, "__dict__", None)
            if not isinstance(namespace, Mapping):
                continue
            for attr_name, value in namespace.items():
                if self._references_live_arrays(value, live_ptrs):
                    issues.append(
                        TypedListFreshnessIssue(
                            kind=TypedListIssueKind.UNREGISTERED_HOLDER,
                            message=(
                                f"{type(callback).__name__} holds unregistered "
                                f"typed.List attribute {attr_name!r}; register it "
                                "with sim.register_typed_list_holder(...) or "
                                "rebuild it on every call"
                            ),
                            holder_attr=attr_name,
                        )
                    )
        return issues

    def _generate_one(
        self, registration: TypedListRegistration, patches: Patches
    ) -> None:
        holder = registration.holder
        assert holder is not None

        for spec in registration.field_lists:
            setattr(
                holder,
                spec.holder_attr,
                typed.List([getattr(p.fields, spec.field_attr) for p in patches]),
            )

        if registration.particle_lists:
            species_names = [s.name for s in patches.species]
            ispecs = self._resolve_species(registration, species_names)
            for spec in registration.particle_lists:
                mirrors: dict[int, Any] = {}
                for ispec in ispecs:
                    mirrors[ispec] = typed.List(
                        [
                            getattr(p.particles[ispec], spec.particle_attr)
                            for p in patches
                        ]
                    )
                setattr(holder, spec.holder_attr, mirrors)

    def _check_registration(
        self,
        registration: TypedListRegistration,
        patches: Patches,
        species_names: list[str],
    ) -> list[TypedListFreshnessIssue]:
        holder = registration.holder
        assert holder is not None
        issues: list[TypedListFreshnessIssue] = []

        def report_missing_species(name: str) -> None:
            issues.append(
                TypedListFreshnessIssue(
                    kind=TypedListIssueKind.WATCHED_SPECIES_MISSING,
                    message=(
                        f"typed-list registration {registration.name!r} watches "
                        f"missing species {name!r}; available: {species_names}"
                    ),
                    name=registration.name,
                    species=name,
                )
            )

        ispecs = self._resolve_species(
            registration, species_names, on_missing=report_missing_species
        )

        for spec in registration.particle_lists:
            mirrors = getattr(holder, spec.holder_attr, None)
            if not isinstance(mirrors, Mapping):
                issues.append(
                    TypedListFreshnessIssue(
                        kind=TypedListIssueKind.STALE_PARTICLE,
                        message=(
                            f"registration {registration.name!r} lost particle "
                            f"list attribute {spec.holder_attr!r}"
                        ),
                        name=registration.name,
                        holder_attr=spec.holder_attr,
                    )
                )
                continue
            for ispec in ispecs:
                per_patch = mirrors.get(ispec)
                if per_patch is None:
                    issues.append(
                        TypedListFreshnessIssue(
                            kind=TypedListIssueKind.STALE_PARTICLE,
                            message=(
                                f"list {spec.holder_attr!r} of {registration.name!r} "
                                f"has no entry for species {species_names[ispec]!r}"
                            ),
                            name=registration.name,
                            holder_attr=spec.holder_attr,
                            species=species_names[ispec],
                        )
                    )
                    continue
                for ipatch, patch in enumerate(patches):
                    live = getattr(patch.particles[ispec], spec.particle_attr)
                    if per_patch[ipatch] is not live:
                        issues.append(
                            TypedListFreshnessIssue(
                                kind=TypedListIssueKind.STALE_PARTICLE,
                                message=(
                                    f"particle list {spec.holder_attr!r} of "
                                    f"{registration.name!r} is stale at patch "
                                    f"{ipatch} (species {species_names[ispec]!r}); "
                                    "refresh it with sim.update_lists() or rebuild "
                                    "it with sim.update_patches()"
                                ),
                                name=registration.name,
                                holder_attr=spec.holder_attr,
                                species=species_names[ispec],
                                ipatch=ipatch,
                            )
                        )

        for spec in registration.field_lists:
            mirror = getattr(holder, spec.holder_attr, None)
            if not isinstance(mirror, typed.List):
                issues.append(
                    TypedListFreshnessIssue(
                        kind=TypedListIssueKind.STALE_FIELD,
                        message=(
                            f"registration {registration.name!r} lost field list "
                            f"attribute {spec.holder_attr!r}"
                        ),
                        name=registration.name,
                        holder_attr=spec.holder_attr,
                    )
                )
                continue
            for ipatch, patch in enumerate(patches):
                if mirror[ipatch] is not getattr(patch.fields, spec.field_attr):
                    issues.append(
                        TypedListFreshnessIssue(
                            kind=TypedListIssueKind.STALE_FIELD,
                            message=(
                                f"field list {spec.holder_attr!r} of "
                                f"{registration.name!r} is stale at patch {ipatch}; "
                                "rebuild it with sim.update_patches()"
                            ),
                            name=registration.name,
                            holder_attr=spec.holder_attr,
                            ipatch=ipatch,
                        )
                    )
        return issues

    @staticmethod
    def _normalize_list_specs(
        registration_name: str,
        holder: Any,
        specs: Mapping[str, str],
        kind: str,
    ) -> None:
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in specs.items()):
            raise TypeError(
                f"typed-list holder {registration_name!r}: {kind} lists must "
                "map str holder attribute names to str source attribute names"
            )
        for holder_attr in specs:
            if not holder_attr.isidentifier():
                raise ValueError(
                    f"typed-list holder {registration_name!r}: attribute "
                    f"{holder_attr!r} is not a valid Python identifier"
                )
            TypedListRegistry._ensure_holder_attr_is_free(
                registration_name, holder, holder_attr
            )

    @staticmethod
    def _validate_list_sources(
        registration_name: str,
        patches: Patches,
        particle_lists: Mapping[str, str],
        field_lists: Mapping[str, str],
        watch: tuple[str, ...],
    ) -> None:
        species_names = [s.name for s in patches.species]
        for name in watch:
            particles = patches[0].particles[species_names.index(name)]
            for source in particle_lists.values():
                if not hasattr(particles, source):
                    raise ValueError(
                        f"typed-list holder {registration_name!r}: particles of "
                        f"species {name!r} have no attribute {source!r}; "
                        f"available: {sorted(particles.attrs)} (plus is_dead)"
                    )
        fields = patches[0].fields
        for source in field_lists.values():
            if not hasattr(fields, source):
                raise ValueError(
                    f"typed-list holder {registration_name!r}: fields have no "
                    f"attribute {source!r}; available: {fields.attrs}"
                )

    @staticmethod
    def _ensure_holder_attr_is_free(
        registration_name: str,
        holder: Any,
        holder_attr: str,
    ) -> None:
        for klass in type(holder).__mro__:
            if holder_attr in klass.__dict__:
                found = klass.__dict__[holder_attr]
                if found is None or isinstance(found, (dict, typed.typedlist.List)):
                    return
                raise ValueError(
                    f"typed-list holder {registration_name!r}: attribute "
                    f"{holder_attr!r} would shadow an existing attribute"
                )
        current = getattr(holder, "__dict__", {}).get(holder_attr)
        if current is not None and not isinstance(current, (dict, typed.typedlist.List)):
            raise ValueError(
                f"typed-list holder {registration_name!r}: attribute "
                f"{holder_attr!r} would shadow an existing attribute"
            )

    @staticmethod
    def _normalize_species(
        patches: Patches,
        species: Sequence[Species | str | int] | Species | str | int | None,
    ) -> tuple[str, ...]:
        if species is None:
            return ()
        if isinstance(species, (Species, str, int, np.integer)) and not isinstance(species, bool):
            species = (species,)
        available = [s.name for s in patches.species]
        names: list[str] = []
        for entry in species:
            if isinstance(entry, Species):
                name = entry.name
            elif isinstance(entry, str):
                name = entry
            elif isinstance(entry, (int, np.integer)) and not isinstance(entry, bool):
                index = int(entry)
                if not 0 <= index < len(available):
                    raise ValueError(
                        f"species index {index} is out of range; available: {available}"
                    )
                name = available[index]
            else:
                raise TypeError(
                    "species watch entries must be Species objects, names, or "
                    f"indices, got {entry!r}"
                )
            if name not in available:
                raise ValueError(
                    f"species {name!r} is not present; available: {available}"
                )
            if name not in names:
                names.append(name)
        return tuple(names)

    @staticmethod
    def _resolve_species(
        registration: TypedListRegistration,
        species_names: list[str],
        on_missing: Callable[[str], None] | None = None,
    ) -> list[int]:
        index_by_name = {name: index for index, name in enumerate(species_names)}
        indices: list[int] = []
        for name in registration.species:
            if name in index_by_name:
                indices.append(index_by_name[name])
            elif on_missing is not None:
                on_missing(name)
            else:
                raise RuntimeError(
                    f"typed-list registration {registration.name!r} watches "
                    f"missing species {name!r}; available: {species_names}"
                )
        return indices

    @staticmethod
    def _live_array_pointers(patches: Patches) -> dict[int, str]:
        live_ptrs: dict[int, str] = {}
        for patch in patches:
            for ispec, particles in enumerate(patch.particles):
                for attr in (*particles.attrs, "is_dead"):
                    array = getattr(particles, attr, None)
                    if isinstance(array, np.ndarray):
                        live_ptrs[array.ctypes.data] = f"particles[{ispec}].{attr}"
            for attr in patch.fields.attrs:
                array = getattr(patch.fields, attr, None)
                if isinstance(array, np.ndarray):
                    live_ptrs[array.ctypes.data] = f"fields.{attr}"
        return live_ptrs

    @staticmethod
    def _references_live_arrays(value: Any, live_ptrs: dict[int, str]) -> bool:
        candidate_lists: list[Any] = []
        if isinstance(value, typed.typedlist.List):
            candidate_lists.append(value)
        elif isinstance(value, Mapping):
            candidate_lists.extend(
                v for v in value.values() if isinstance(v, typed.typedlist.List)
            )
        elif isinstance(value, (list, tuple, set, frozenset)):
            candidate_lists.extend(
                v for v in value if isinstance(v, typed.typedlist.List)
            )

        for typed_list in candidate_lists:
            for entry in typed_list:
                if isinstance(entry, np.ndarray) and entry.ctypes.data in live_ptrs:
                    return True
        return False
