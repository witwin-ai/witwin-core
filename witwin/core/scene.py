from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields, is_dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

import torch

from .antenna import AntennaState
from .identity import MaterialId, StructureId
from .material import MaterialAssignment, MaterialSpec, PhaseScreen, Structure


def _version(value: int, *, name: str) -> int:
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name} must be non-negative.")
    return resolved


def _tensor_states(
    value,
    *,
    seen: set[int] | None = None,
    include_identity: bool = True,
) -> tuple[tuple, ...]:
    if seen is None:
        seen = set()
    object_id = id(value)
    if object_id in seen:
        return ()
    seen.add(object_id)
    if isinstance(value, torch.Tensor):
        identity = (object_id, value.data_ptr()) if include_identity else ()
        return (
            (
                *identity,
                str(value.device),
                str(value.dtype),
                tuple(value.shape),
                tuple(value.stride()),
                bool(value.requires_grad),
                int(getattr(value, "_version", 0)),
            ),
        )
    if isinstance(value, Mapping):
        return tuple(
            state
            for key in sorted(value, key=lambda item: repr(item))
            for state in _tensor_states(
                value[key], seen=seen, include_identity=include_identity
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(
            state
            for item in value
            for state in _tensor_states(
                item, seen=seen, include_identity=include_identity
            )
        )
    if is_dataclass(value):
        return tuple(
            state
            for field in fields(value)
            for state in _tensor_states(
                getattr(value, field.name),
                seen=seen,
                include_identity=include_identity,
            )
        )
    if hasattr(value, "__dict__"):
        return tuple(
            state
            for name, item in sorted(vars(value).items())
            if not name.endswith("_cache")
            for state in _tensor_states(
                item, seen=seen, include_identity=include_identity
            )
        )
    return ()


def _state_version(
    authored_version: int,
    *,
    identity: tuple,
    tensor_states: tuple[tuple, ...],
) -> int:
    mutation_sum = sum(
        (index + 1) * int(state[-1])
        for index, state in enumerate(tensor_states)
    )
    digest = hashlib.blake2b(
        repr((identity, tensor_states)).encode("utf-8"), digest_size=16
    ).digest()
    return (
        (int(authored_version) << 256)
        | (mutation_sum << 128)
        | int.from_bytes(digest, "big")
    )


@dataclass(frozen=True, slots=True)
class SceneVersions:
    topology: int = 0
    geometry: int = 0
    material: int = 0
    assignment: int = 0

    def __post_init__(self) -> None:
        for name in ("topology", "geometry", "material", "assignment"):
            object.__setattr__(
                self, name, _version(getattr(self, name), name=f"{name}_version")
            )

    def advanced(self, **increments: int) -> SceneVersions:
        unknown = set(increments) - {
            "topology",
            "geometry",
            "material",
            "assignment",
        }
        if unknown:
            raise TypeError(f"Unknown version fields: {sorted(unknown)}")
        if any(int(value) < 0 for value in increments.values()):
            raise ValueError("Version increments must be non-negative.")
        return SceneVersions(
            topology=self.topology + int(increments.get("topology", 0)),
            geometry=self.geometry + int(increments.get("geometry", 0)),
            material=self.material + int(increments.get("material", 0)),
            assignment=self.assignment + int(increments.get("assignment", 0)),
        )


@dataclass(frozen=True, slots=True)
class StructureState:
    """A logical structure plus optional continuous motion/deformation state."""

    structure: Structure
    rigid_motion: object | None = None
    deformation: object | None = None

    @property
    def structure_id(self) -> StructureId:
        if self.structure.structure_id is None:
            raise RuntimeError("StructureState requires a normalized structure ID.")
        return self.structure.structure_id


@dataclass(frozen=True, slots=True)
class EndpointState:
    antenna: AntennaState
    rigid_motion: object | None = None


@dataclass(frozen=True, init=False)
class SceneSnapshot:
    time_s: Any
    _topology_version: int
    _geometry_version: int
    _material_version: int
    _assignment_version: int
    structures: tuple[StructureState, ...]
    endpoints: tuple[EndpointState, ...]
    metadata: Mapping[str, Any]
    _source_scene: Scene | None
    _dynamic_revision: int | None
    _time_revision_at_creation: int | None
    _dynamic_sources: tuple
    _dynamic_source_state_at_creation: tuple[tuple, ...]

    def __init__(
        self,
        *,
        time_s: Any,
        topology_version: int,
        geometry_version: int,
        material_version: int,
        assignment_version: int,
        structures,
        endpoints=(),
        metadata: Mapping[str, Any] | None = None,
        source_scene: Scene | None = None,
        dynamic_revision: int | None = None,
        dynamic_sources=(),
    ) -> None:
        if isinstance(time_s, torch.Tensor) and time_s.ndim != 0:
            raise ValueError("time_s tensor must be scalar.")
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(
            self,
            "_topology_version",
            _version(topology_version, name="topology_version"),
        )
        object.__setattr__(
            self,
            "_geometry_version",
            _version(geometry_version, name="geometry_version"),
        )
        object.__setattr__(
            self,
            "_material_version",
            _version(material_version, name="material_version"),
        )
        object.__setattr__(
            self,
            "_assignment_version",
            _version(assignment_version, name="assignment_version"),
        )
        object.__setattr__(self, "structures", tuple(structures))
        object.__setattr__(self, "endpoints", tuple(endpoints))
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(metadata or {}))
        )
        object.__setattr__(self, "_source_scene", source_scene)
        object.__setattr__(
            self,
            "_dynamic_revision",
            None if dynamic_revision is None else int(dynamic_revision),
        )
        object.__setattr__(
            self,
            "_time_revision_at_creation",
            (
                int(getattr(time_s, "_version", 0))
                if isinstance(time_s, torch.Tensor)
                else None
            ),
        )
        resolved_dynamic_sources = tuple(dynamic_sources)
        object.__setattr__(
            self, "_dynamic_sources", resolved_dynamic_sources
        )
        object.__setattr__(
            self,
            "_dynamic_source_state_at_creation",
            _tensor_states(
                resolved_dynamic_sources, include_identity=True
            ),
        )

    @property
    def topology_version(self) -> int:
        if self._source_scene is not None:
            return self._source_scene.topology_version
        structures = tuple(state.structure for state in self.structures)
        return _state_version(
            self._topology_version,
            identity=tuple(
                (
                    int(structure.structure_id),
                    int(structure.surface_id),
                    None
                    if structure.primitive_ids is None
                    else tuple(int(value) for value in structure.primitive_ids),
                )
                for structure in structures
            ),
            tensor_states=tuple(
                state
                for structure in structures
                for state in _tensor_states(
                    (
                        getattr(structure.geometry, "_faces_tensor", ()),
                        structure.face_uv,
                    ),
                    include_identity=False,
                )
            ),
        )

    @property
    def geometry_version(self) -> int:
        if self._source_scene is not None and self._dynamic_revision is None:
            return self._source_scene.geometry_version
        geometry_state = (
            tuple(
                (
                    state.structure.geometry,
                    state.structure.uv,
                    state.rigid_motion,
                    state.deformation,
                )
                for state in self.structures
            ),
            tuple(
                (state.antenna, state.rigid_motion) for state in self.endpoints
            ),
            self.time_s,
        )
        base_version = (
            self._geometry_version
            if self._source_scene is None
            else self._source_scene.geometry_version
        )
        scalar_time = (
            None
            if isinstance(self.time_s, torch.Tensor)
            else float(self.time_s).hex()
        )
        payload = (
            base_version,
            self._dynamic_revision,
            self._time_revision_at_creation,
            scalar_time,
            _tensor_states(self.time_s, include_identity=True),
            tuple(int(state.structure_id) for state in self.structures),
            tuple(
                int(state.antenna.antenna_id) for state in self.endpoints
            ),
            _tensor_states(geometry_state, include_identity=False),
            self._dynamic_source_state_at_creation,
            _tensor_states(
                self._dynamic_sources, include_identity=True
            ),
        )
        return int.from_bytes(
            hashlib.blake2b(
                repr(payload).encode("utf-8"), digest_size=16
            ).digest(),
            "big",
        )

    @property
    def material_version(self) -> int:
        if self._source_scene is not None:
            return self._source_scene.material_version
        materials: dict[int, MaterialSpec] = {}
        for state in self.structures:
            structure = state.structure
            materials[int(structure.material_id)] = structure.material
        return _state_version(
            self._material_version,
            identity=tuple(
                (
                    material_id,
                    int(getattr(material, "version", 0)),
                )
                for material_id, material in sorted(materials.items())
            ),
            tensor_states=tuple(
                state
                for _, material in sorted(materials.items())
                for state in _tensor_states(
                    material, include_identity=False
                )
            ),
        )

    @property
    def assignment_version(self) -> int:
        if self._source_scene is not None:
            return self._source_scene.assignment_version
        return _state_version(
            self._assignment_version,
            identity=tuple(
                (
                    int(state.structure.structure_id),
                    int(state.structure.surface_id),
                    int(state.structure.assignment_id),
                    int(state.structure.material_id),
                )
                for state in self.structures
            ),
            tensor_states=tuple(
                tensor_state
                for state in self.structures
                for tensor_state in _tensor_states(
                    state.structure.phase_screen,
                    include_identity=False,
                )
            ),
        )

    @property
    def versions(self) -> SceneVersions:
        return SceneVersions(
            topology=self.topology_version,
            geometry=self.geometry_version,
            material=self.material_version,
            assignment=self.assignment_version,
        )

@dataclass(frozen=True, init=False)
class Scene:
    """Canonical logical world model with granular invalidation versions."""

    structures: tuple[Structure, ...]
    endpoints: tuple[AntennaState, ...]
    _versions: SceneVersions
    metadata: Mapping[str, Any]

    def __init__(
        self,
        *,
        structures=(),
        endpoints=(),
        versions: SceneVersions | None = None,
        topology_version: int = 0,
        geometry_version: int = 0,
        material_version: int = 0,
        assignment_version: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        resolved_versions = (
            SceneVersions(
                topology=topology_version,
                geometry=geometry_version,
                material=material_version,
                assignment=assignment_version,
            )
            if versions is None
            else versions
        )
        if not isinstance(resolved_versions, SceneVersions):
            raise TypeError("versions must be a SceneVersions or None.")
        normalized_structures = self._normalize_structures(tuple(structures))
        normalized_endpoints = tuple(endpoints)
        if any(
            not isinstance(endpoint, AntennaState)
            for endpoint in normalized_endpoints
        ):
            raise TypeError("endpoints must contain only AntennaState values.")
        endpoint_ids = [int(endpoint.antenna_id) for endpoint in normalized_endpoints]
        if len(endpoint_ids) != len(set(endpoint_ids)):
            raise ValueError("Antenna IDs must be unique within a Scene.")

        object.__setattr__(self, "structures", normalized_structures)
        object.__setattr__(self, "endpoints", normalized_endpoints)
        object.__setattr__(self, "_versions", resolved_versions)
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(metadata or {}))
        )

    @staticmethod
    def _normalize_structures(
        structures: tuple[Structure, ...],
    ) -> tuple[Structure, ...]:
        if any(not isinstance(structure, Structure) for structure in structures):
            raise TypeError("structures must contain only Structure values.")

        material_objects_by_id: dict[int, MaterialSpec] = {}
        used_structure_ids: set[int] = set()
        used_assignment_ids: set[int] = set()
        used_surface_ids: set[int] = set()
        used_primitive_ids: set[int] = set()
        normalized: list[Structure] = []
        for structure in structures:
            if (
                structure.structure_id is None
                or structure.assignment_id is None
                or structure.material_id is None
            ):
                raise ValueError(
                    "Structure, material, and assignment IDs must be authored "
                    "before Scene construction."
                )
            structure_id = structure.structure_id
            assignment_id = structure.assignment_id
            material_id = structure.material_id
            surface_id = structure.surface_id

            structure_value = int(structure_id)
            assignment_value = int(assignment_id)
            material_value = int(material_id)
            surface_value = int(surface_id)
            if min(
                structure_value,
                assignment_value,
                material_value,
                surface_value,
            ) < 0:
                raise ValueError("Scene IDs must be non-negative.")
            if structure_value in used_structure_ids:
                raise ValueError("Structure IDs must be unique within a Scene.")
            if assignment_value in used_assignment_ids:
                raise ValueError("Assignment IDs must be unique within a Scene.")
            if surface_value in used_surface_ids:
                raise ValueError("Surface IDs must be unique within a Scene.")
            primitive_values = tuple(
                int(value) for value in (structure.primitive_ids or ())
            )
            if any(value < 0 for value in primitive_values):
                raise ValueError("Primitive IDs must be non-negative.")
            if len(primitive_values) != len(set(primitive_values)):
                raise ValueError(
                    "Primitive IDs must be unique within a Structure."
                )
            if used_primitive_ids.intersection(primitive_values):
                raise ValueError(
                    "Primitive IDs must be unique within a Scene."
                )
            existing_material = material_objects_by_id.get(material_value)
            if (
                existing_material is not None
                and existing_material is not structure.material
            ):
                raise ValueError(
                    "One MaterialId cannot identify multiple material objects."
                )

            used_structure_ids.add(structure_value)
            used_assignment_ids.add(assignment_value)
            used_surface_ids.add(surface_value)
            used_primitive_ids.update(primitive_values)
            material_objects_by_id[material_value] = structure.material
            normalized.append(structure)
        return tuple(normalized)

    @property
    def topology_version(self) -> int:
        tensor_states = tuple(
            state
            for structure in self.structures
            for state in _tensor_states(
                getattr(structure.geometry, "_faces_tensor", ())
            )
            + _tensor_states(structure.face_uv)
        )
        identity = tuple(
            (
                int(structure.structure_id),
                int(structure.surface_id),
                None
                if structure.primitive_ids is None
                else tuple(int(value) for value in structure.primitive_ids),
            )
            for structure in self.structures
        )
        return _state_version(
            self._versions.topology,
            identity=identity,
            tensor_states=tensor_states,
        )

    @property
    def geometry_version(self) -> int:
        geometry_states: list[tuple] = []
        for structure in self.structures:
            geometry = structure.geometry
            if hasattr(geometry, "__dict__"):
                geometry_values = {
                    name: value
                    for name, value in vars(geometry).items()
                    if name != "_faces_tensor" and not name.endswith("_cache")
                }
            else:
                geometry_values = geometry
            geometry_states.extend(_tensor_states(geometry_values))
            geometry_states.extend(_tensor_states(structure.uv))
        geometry_states.extend(_tensor_states(self.endpoints))
        return _state_version(
            self._versions.geometry,
            identity=tuple(
                int(structure.structure_id) for structure in self.structures
            )
            + tuple(int(endpoint.antenna_id) for endpoint in self.endpoints),
            tensor_states=tuple(geometry_states),
        )

    @property
    def material_version(self) -> int:
        materials: dict[int, MaterialSpec] = {}
        for structure in self.structures:
            materials[int(structure.material_id)] = structure.material
        return _state_version(
            self._versions.material,
            identity=tuple(
                (
                    material_id,
                    int(getattr(material, "version", 0)),
                )
                for material_id, material in sorted(materials.items())
            ),
            tensor_states=tuple(
                state
                for _, material in sorted(materials.items())
                for state in _tensor_states(material)
            ),
        )

    @property
    def assignment_version(self) -> int:
        return _state_version(
            self._versions.assignment,
            identity=tuple(
                (
                    int(structure.structure_id),
                    int(structure.surface_id),
                    int(structure.assignment_id),
                    int(structure.material_id),
                )
                for structure in self.structures
            ),
            tensor_states=tuple(
                state
                for structure in self.structures
                for state in _tensor_states(structure.phase_screen)
            ),
        )

    @property
    def versions(self) -> SceneVersions:
        return SceneVersions(
            topology=self.topology_version,
            geometry=self.geometry_version,
            material=self.material_version,
            assignment=self.assignment_version,
        )

    @property
    def assignments(self) -> tuple[MaterialAssignment, ...]:
        return tuple(
            MaterialAssignment(
                assignment_id=structure.assignment_id,
                surface_id=structure.surface_id,
                material_id=structure.material_id,
                material=structure.material,
                phase_screen=structure.phase_screen,
            )
            for structure in self.structures
        )

    def structure(self, structure_id: StructureId | int) -> Structure:
        target = int(structure_id)
        for structure in self.structures:
            if int(structure.structure_id) == target:
                return structure
        raise KeyError(f"Unknown StructureId {target}.")

    def with_structures(self, structures) -> Scene:
        return Scene(
            structures=structures,
            endpoints=self.endpoints,
            versions=self._versions.advanced(topology=1, assignment=1),
            metadata=self.metadata,
        )

    def with_structure_geometry(
        self,
        structure_id: StructureId | int,
        geometry,
        *,
        topology_changed: bool | None = None,
    ) -> Scene:
        target = int(structure_id)
        original = self.structure(target).geometry
        updated = tuple(
            replace(structure, geometry=geometry)
            if int(structure.structure_id) == target
            else structure
            for structure in self.structures
        )
        if all(
            replacement is original
            for replacement, original in zip(updated, self.structures)
        ):
            raise KeyError(f"Unknown StructureId {target}.")
        inferred_topology_change = type(original) is not type(geometry)
        original_faces = getattr(original, "_faces_tensor", None)
        replacement_faces = getattr(geometry, "_faces_tensor", None)
        if original_faces is not None or replacement_faces is not None:
            inferred_topology_change = (
                original_faces is None
                or replacement_faces is None
                or original_faces is not replacement_faces
                or tuple(original_faces.shape)
                != tuple(replacement_faces.shape)
                or int(getattr(original_faces, "_version", 0))
                != int(getattr(replacement_faces, "_version", 0))
            )
        if topology_changed is False and inferred_topology_change:
            raise ValueError(
                "topology_changed=False conflicts with a detectable topology change."
            )
        if topology_changed is None:
            topology_changed = inferred_topology_change
        increments = (
            {"topology": 1, "geometry": 1}
            if topology_changed
            else {"geometry": 1}
        )
        return Scene(
            structures=updated,
            endpoints=self.endpoints,
            versions=self._versions.advanced(**increments),
            metadata=self.metadata,
        )

    def with_structure_material(
        self,
        structure_id: StructureId | int,
        material: MaterialSpec,
        *,
        material_id: MaterialId | int | None = None,
        phase_screen: PhaseScreen | None = None,
    ) -> Scene:
        target = int(structure_id)
        if material_id is None:
            inherited_id = getattr(material, "material_id", None)
            if inherited_id is None:
                raise ValueError(
                    "material_id is required for a MaterialSpec without "
                    "stable material identity."
                )
            resolved_material_id = MaterialId(int(inherited_id))
        else:
            resolved_material_id = MaterialId(int(material_id))
        existing_materials = {
            int(structure.material_id): structure.material
            for structure in self.structures
        }
        existing_material = existing_materials.get(int(resolved_material_id))
        if existing_material is not None and existing_material is not material:
            raise ValueError(
                "with_structure_material cannot redefine an existing MaterialId; "
                "use with_material for a material specification update."
            )
        adds_material = existing_material is None
        updated = tuple(
            replace(
                structure,
                material=material,
                material_id=resolved_material_id,
                phase_screen=phase_screen,
            )
            if int(structure.structure_id) == target
            else structure
            for structure in self.structures
        )
        if all(
            replacement is original
            for replacement, original in zip(updated, self.structures)
        ):
            raise KeyError(f"Unknown StructureId {target}.")
        return Scene(
            structures=updated,
            endpoints=self.endpoints,
            versions=self._versions.advanced(
                material=1 if adds_material else 0,
                assignment=1,
            ),
            metadata=self.metadata,
        )

    def with_material(
        self,
        material_id: MaterialId | int,
        material: MaterialSpec,
    ) -> Scene:
        target = int(material_id)
        declared_id = getattr(material, "material_id", target)
        if int(declared_id) != target:
            raise ValueError(
                "Replacement material identity must match material_id."
            )
        found = False
        updated: list[Structure] = []
        for structure in self.structures:
            if int(structure.material_id) == target:
                found = True
                updated.append(replace(structure, material=material))
            else:
                updated.append(structure)
        if not found:
            raise KeyError(f"Unknown MaterialId {target}.")
        return Scene(
            structures=tuple(updated),
            endpoints=self.endpoints,
            versions=self._versions.advanced(material=1),
            metadata=self.metadata,
        )

    def with_endpoints(self, endpoints) -> Scene:
        return Scene(
            structures=self.structures,
            endpoints=endpoints,
            versions=self._versions.advanced(geometry=1),
            metadata=self.metadata,
        )

    def snapshot(self, time_s: Any = 0.0) -> SceneSnapshot:
        return SceneSnapshot(
            time_s=time_s,
            topology_version=self._versions.topology,
            geometry_version=self._versions.geometry,
            material_version=self._versions.material,
            assignment_version=self._versions.assignment,
            structures=tuple(
                StructureState(structure=structure)
                for structure in self.structures
            ),
            endpoints=tuple(
                EndpointState(antenna=endpoint) for endpoint in self.endpoints
            ),
            metadata=self.metadata,
            source_scene=self,
        )


__all__ = [
    "EndpointState",
    "Scene",
    "SceneSnapshot",
    "SceneVersions",
    "StructureState",
]
