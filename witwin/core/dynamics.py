from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

import torch

from .identity import AntennaId, StructureId
from .scalars import ScalarLike
from .scene import EndpointState, Scene, SceneSnapshot, StructureState


def _optional_tensor(value, *, name: str, shape_tail: tuple[int, ...]):
    if value is None:
        return None
    tensor = (
        value
        if isinstance(value, torch.Tensor)
        else torch.tensor(value, dtype=torch.float32)
    )
    if tensor.shape[-len(shape_tail) :] != shape_tail:
        raise ValueError(f"{name} must end with shape {shape_tail}.")
    if not tensor.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating-point dtype.")
    return tensor


@dataclass(frozen=True, init=False)
class RigidMotion:
    """Additional rigid state composed after the authored logical pose.

    ``translation`` is a world-frame displacement. ``rotation`` is a
    local-to-world rotation composed on the left of the authored rotation;
    Euler values use the antenna ``(yaw, pitch, roll)`` intrinsic Z-Y-X
    convention and quaternions use scalar-first ``(w, x, y, z)``.
    """

    translation: torch.Tensor | None
    rotation: torch.Tensor | None
    velocity: torch.Tensor | None
    angular_velocity: torch.Tensor | None

    def __init__(
        self,
        *,
        translation=None,
        rotation=None,
        velocity=None,
        angular_velocity=None,
    ) -> None:
        object.__setattr__(
            self,
            "translation",
            _optional_tensor(translation, name="translation", shape_tail=(3,)),
        )
        resolved_rotation = (
            None
            if rotation is None
            else (
                rotation
                if isinstance(rotation, torch.Tensor)
                else torch.tensor(rotation, dtype=torch.float32)
            )
        )
        if (
            resolved_rotation is not None
            and resolved_rotation.shape not in {(3,), (4,)}
        ):
            raise ValueError(
                "rotation must have shape (3,) for Euler angles or "
                "(4,) for a quaternion."
            )
        if (
            resolved_rotation is not None
            and not resolved_rotation.dtype.is_floating_point
        ):
            raise TypeError("rotation must use a floating-point dtype.")
        object.__setattr__(self, "rotation", resolved_rotation)
        object.__setattr__(
            self,
            "velocity",
            _optional_tensor(velocity, name="velocity", shape_tail=(3,)),
        )
        object.__setattr__(
            self,
            "angular_velocity",
            _optional_tensor(
                angular_velocity,
                name="angular_velocity",
                shape_tail=(3,),
            ),
        )
        tensor_leaves = tuple(
            value
            for value in (
                self.translation,
                self.rotation,
                self.velocity,
                self.angular_velocity,
            )
            if isinstance(value, torch.Tensor)
        )
        if tensor_leaves and any(
            value.device != tensor_leaves[0].device
            for value in tensor_leaves[1:]
        ):
            raise ValueError("RigidMotion tensor leaves must be on the same device.")
        if tensor_leaves and any(
            value.dtype != tensor_leaves[0].dtype
            for value in tensor_leaves[1:]
        ):
            raise ValueError("RigidMotion tensor leaves must use the same dtype.")


@dataclass(frozen=True, init=False)
class DeformationState:
    """Topology-preserving local-space vertices or vertex offsets.

    Absolute ``vertices`` replace authored local mesh vertices; ``offsets`` are
    added to authored local vertices. Deformation is resolved before authored
    geometry transforms and the snapshot's additional rigid motion.
    """

    vertices: torch.Tensor | None
    offsets: torch.Tensor | None

    def __init__(self, *, vertices=None, offsets=None) -> None:
        if (vertices is None) == (offsets is None):
            raise ValueError(
                "DeformationState requires exactly one of vertices or offsets."
            )
        resolved_vertices = _optional_tensor(
            vertices, name="vertices", shape_tail=(3,)
        )
        resolved_offsets = _optional_tensor(
            offsets, name="offsets", shape_tail=(3,)
        )
        if (
            resolved_vertices is not None
            and resolved_vertices.ndim != 2
        ):
            raise ValueError("vertices must have shape (V, 3).")
        if resolved_offsets is not None and resolved_offsets.ndim != 2:
            raise ValueError("offsets must have shape (V, 3).")
        object.__setattr__(self, "vertices", resolved_vertices)
        object.__setattr__(self, "offsets", resolved_offsets)


@runtime_checkable
class Trajectory(Protocol):
    def at(self, time_s: ScalarLike) -> RigidMotion:
        ...


@runtime_checkable
class Deformation(Protocol):
    def at(self, time_s: ScalarLike) -> DeformationState:
        ...


@dataclass(frozen=True)
class LinearTrajectory:
    """Constant-velocity world displacement from ``origin`` at reference time."""

    origin: torch.Tensor
    velocity: torch.Tensor
    rotation: torch.Tensor | None = None
    angular_velocity: torch.Tensor | None = None
    reference_time_s: float = 0.0

    def __post_init__(self) -> None:
        for name in ("origin", "velocity"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32)
                object.__setattr__(self, name, value)
            if value.shape != (3,):
                raise ValueError(f"{name} must have shape (3,).")
            if not value.dtype.is_floating_point:
                raise TypeError(f"{name} must use a floating-point dtype.")
        for name, shapes in (
            ("rotation", {(3,), (4,)}),
            ("angular_velocity", {(3,)}),
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32)
                object.__setattr__(self, name, value)
            if value.shape not in shapes:
                expected = " or ".join(str(shape) for shape in sorted(shapes))
                raise ValueError(f"{name} must have shape {expected}.")
            if not value.dtype.is_floating_point:
                raise TypeError(f"{name} must use a floating-point dtype.")
        tensor_leaves = tuple(
            value
            for value in (
                self.origin,
                self.velocity,
                self.rotation,
                self.angular_velocity,
            )
            if isinstance(value, torch.Tensor)
        )
        if any(value.device != tensor_leaves[0].device for value in tensor_leaves[1:]):
            raise ValueError("Trajectory tensor leaves must be on the same device.")
        if any(value.dtype != tensor_leaves[0].dtype for value in tensor_leaves[1:]):
            raise ValueError("Trajectory tensor leaves must use the same dtype.")

    def at(self, time_s: ScalarLike) -> RigidMotion:
        if (
            isinstance(time_s, torch.Tensor)
            and time_s.device != self.origin.device
        ):
            raise ValueError(
                "Tensor time_s must be on the trajectory tensor device."
            )
        if (
            isinstance(time_s, torch.Tensor)
            and time_s.dtype != self.origin.dtype
        ):
            raise ValueError(
                "Tensor time_s must use the trajectory tensor dtype."
            )
        delta_t = time_s - self.reference_time_s
        return RigidMotion(
            translation=self.origin + self.velocity * delta_t,
            rotation=self.rotation,
            velocity=self.velocity,
            angular_velocity=self.angular_velocity,
        )


@dataclass(frozen=True, init=False)
class DynamicScene:
    """Apply logical motion descriptors without compiling solver resources."""

    scene: Scene
    structure_trajectories: Mapping[StructureId, Trajectory]
    structure_deformations: Mapping[StructureId, Deformation]
    endpoint_trajectories: Mapping[AntennaId, Trajectory]
    dynamic_revision: int

    def __init__(
        self,
        scene: Scene,
        *,
        structure_trajectories: Mapping[StructureId | int, Trajectory]
        | None = None,
        structure_deformations: Mapping[StructureId | int, Deformation]
        | None = None,
        endpoint_trajectories: Mapping[AntennaId | int, Trajectory]
        | None = None,
        dynamic_revision: int = 0,
    ) -> None:
        if not isinstance(scene, Scene):
            raise TypeError("scene must be a Scene.")
        revision = int(dynamic_revision)
        if revision < 0:
            raise ValueError("dynamic_revision must be non-negative.")
        trajectories = {
            StructureId(int(key)): value
            for key, value in (structure_trajectories or {}).items()
        }
        deformations = {
            StructureId(int(key)): value
            for key, value in (structure_deformations or {}).items()
        }
        endpoint_motions = {
            AntennaId(int(key)): value
            for key, value in (endpoint_trajectories or {}).items()
        }
        structure_ids = {structure.structure_id for structure in scene.structures}
        endpoint_ids = {endpoint.antenna_id for endpoint in scene.endpoints}
        unknown_structures = (set(trajectories) | set(deformations)) - structure_ids
        unknown_endpoints = set(endpoint_motions) - endpoint_ids
        if unknown_structures:
            raise KeyError(
                f"Unknown dynamic StructureIds: {sorted(unknown_structures)}"
            )
        if unknown_endpoints:
            raise KeyError(
                f"Unknown dynamic AntennaIds: {sorted(unknown_endpoints)}"
            )
        if any(not isinstance(value, Trajectory) for value in trajectories.values()):
            raise TypeError("structure_trajectories values must implement Trajectory.")
        if any(
            not isinstance(value, Deformation) for value in deformations.values()
        ):
            raise TypeError(
                "structure_deformations values must implement Deformation."
            )
        if any(
            not isinstance(value, Trajectory) for value in endpoint_motions.values()
        ):
            raise TypeError("endpoint_trajectories values must implement Trajectory.")

        object.__setattr__(self, "scene", scene)
        object.__setattr__(
            self,
            "structure_trajectories",
            MappingProxyType(trajectories),
        )
        object.__setattr__(
            self,
            "structure_deformations",
            MappingProxyType(deformations),
        )
        object.__setattr__(
            self, "endpoint_trajectories", MappingProxyType(endpoint_motions)
        )
        object.__setattr__(self, "dynamic_revision", revision)

    def at(self, time_s: ScalarLike) -> SceneSnapshot:
        if isinstance(time_s, torch.Tensor) and time_s.ndim != 0:
            raise ValueError("time_s tensor must be scalar.")
        resolved_time = time_s
        has_dynamic_geometry = bool(
            self.structure_trajectories
            or self.structure_deformations
            or self.endpoint_trajectories
        )
        structures = tuple(
            StructureState(
                structure=structure,
                rigid_motion=(
                    None
                    if structure.structure_id not in self.structure_trajectories
                    else self.structure_trajectories[structure.structure_id].at(
                        resolved_time
                    )
                ),
                deformation=(
                    None
                    if structure.structure_id not in self.structure_deformations
                    else self.structure_deformations[structure.structure_id].at(
                        resolved_time
                    )
                ),
            )
            for structure in self.scene.structures
        )
        endpoints = tuple(
            EndpointState(
                antenna=endpoint,
                rigid_motion=(
                    None
                    if endpoint.antenna_id not in self.endpoint_trajectories
                    else self.endpoint_trajectories[endpoint.antenna_id].at(
                        resolved_time
                    )
                ),
            )
            for endpoint in self.scene.endpoints
        )
        dynamic_sources = (
            tuple(
                ("structure_trajectory", int(key), value)
                for key, value in sorted(
                    self.structure_trajectories.items(), key=lambda item: int(item[0])
                )
            )
            + tuple(
                ("structure_deformation", int(key), value)
                for key, value in sorted(
                    self.structure_deformations.items(), key=lambda item: int(item[0])
                )
            )
            + tuple(
                ("endpoint_trajectory", int(key), value)
                for key, value in sorted(
                    self.endpoint_trajectories.items(), key=lambda item: int(item[0])
                )
            )
        )
        return SceneSnapshot(
            time_s=resolved_time,
            topology_version=self.scene._versions.topology,
            geometry_version=self.scene._versions.geometry,
            material_version=self.scene._versions.material,
            assignment_version=self.scene._versions.assignment,
            structures=structures,
            endpoints=endpoints,
            metadata=self.scene.metadata,
            source_scene=self.scene,
            dynamic_revision=(
                self.dynamic_revision if has_dynamic_geometry else None
            ),
            dynamic_sources=dynamic_sources,
        )


__all__ = [
    "Deformation",
    "DeformationState",
    "DynamicScene",
    "LinearTrajectory",
    "RigidMotion",
    "Trajectory",
]
