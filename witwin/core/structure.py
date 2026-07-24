"""Structure composition and material assignment contracts.

A :class:`Structure` binds one geometry to one material and carries the stable
logical identities that solver-owned compilers map onto their native stores.
Material specifications themselves live in :mod:`witwin.core.material`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch

from .geometry.base import GeometrySpec
from .identity import (
    AssignmentId,
    MaterialId,
    PrimitiveId,
    StructureId,
    SurfaceId,
    new_assignment_id,
    new_material_id,
    new_structure_id,
    new_surface_id,
    reserve_assignment_id,
    reserve_material_id,
    reserve_structure_id,
    reserve_surface_id,
)
from .material import MaterialSpec, PhaseScreen


__all__ = ["MaterialAssignment", "Structure"]


@dataclass(frozen=True)
class MaterialAssignment:
    assignment_id: AssignmentId
    surface_id: SurfaceId
    material_id: MaterialId
    material: MaterialSpec
    phase_screen: PhaseScreen | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.material, MaterialSpec):
            raise TypeError("material must implement MaterialSpec.")
        if self.phase_screen is not None and not isinstance(
            self.phase_screen, PhaseScreen
        ):
            raise TypeError("phase_screen must be a PhaseScreen or None.")


@dataclass(frozen=True, init=False)
class Structure:
    geometry: GeometrySpec
    material: MaterialSpec
    name: str | None
    priority: int
    enabled: bool
    tags: tuple[str, ...]
    metadata: Mapping[str, Any]
    structure_id: StructureId | None
    material_id: MaterialId | None
    assignment_id: AssignmentId | None
    surface_id: SurfaceId
    primitive_ids: tuple[PrimitiveId, ...] | None
    phase_screen: PhaseScreen | None
    uv: torch.Tensor | None
    face_uv: torch.Tensor | None

    def __init__(
        self,
        geometry: GeometrySpec,
        material: MaterialSpec,
        name: str | None = None,
        priority: int = 0,
        enabled: bool = True,
        tags=(),
        metadata: Mapping[str, Any] | None = None,
        *,
        structure_id: StructureId | int | None = None,
        material_id: MaterialId | int | None = None,
        assignment_id: AssignmentId | int | None = None,
        surface_id: SurfaceId | int | None = None,
        primitive_ids=(),
        phase_screen: PhaseScreen | None = None,
        uv: torch.Tensor | None = None,
        face_uv: torch.Tensor | None = None,
    ):
        if not isinstance(geometry, GeometrySpec):
            raise TypeError(
                "Structure geometry must implement GeometrySpec (kind and to_mesh())."
            )
        if not isinstance(material, MaterialSpec):
            raise TypeError(
                "Structure material must implement MaterialSpec "
                "(name, capabilities(), evaluate_static(), and evaluate_at_frequency())."
            )
        normalized_primitive_ids = (
            None
            if primitive_ids is None
            else tuple(PrimitiveId(int(value)) for value in primitive_ids)
        )
        if normalized_primitive_ids == ():
            normalized_primitive_ids = None
        if phase_screen is not None and not isinstance(phase_screen, PhaseScreen):
            raise TypeError("phase_screen must be a PhaseScreen or None.")
        if (uv is None) != (face_uv is None):
            raise ValueError("uv and face_uv must be provided together.")
        if uv is not None:
            if not isinstance(uv, torch.Tensor):
                raise TypeError("uv must be a torch.Tensor.")
            if not isinstance(face_uv, torch.Tensor):
                raise TypeError("face_uv must be a torch.Tensor.")
            if uv.ndim != 2 or uv.shape[1] != 2 or uv.shape[0] == 0:
                raise ValueError("uv must have shape (T, 2) with T > 0.")
            if not uv.dtype.is_floating_point:
                raise TypeError("uv must use a floating-point dtype.")
            if (
                face_uv.ndim != 2
                or face_uv.shape[1] != 3
                or face_uv.shape[0] == 0
            ):
                raise ValueError(
                    "face_uv must have shape (F, 3) with F > 0."
                )
            if face_uv.dtype not in {torch.int32, torch.int64}:
                raise TypeError(
                    "face_uv must use torch.int32 or torch.int64."
                )
            geometry_faces = getattr(geometry, "faces", None)
            geometry_vertices = getattr(geometry, "vertices", None)
            if not isinstance(geometry_faces, torch.Tensor):
                raise TypeError(
                    "uv mappings require geometry with typed mesh faces."
                )
            if face_uv.shape[0] != geometry_faces.shape[0]:
                raise ValueError("face_uv must have one row per mesh face.")
            tensor_leaves = tuple(
                value
                for value in (
                    geometry_vertices,
                    geometry_faces,
                    uv,
                    face_uv,
                )
                if isinstance(value, torch.Tensor)
            )
            if any(
                value.device != tensor_leaves[0].device
                for value in tensor_leaves[1:]
            ):
                raise ValueError(
                    "Mesh and UV tensor leaves must be on the same device."
                )

        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "material", material)
        object.__setattr__(self, "name", None if name is None else str(name))
        object.__setattr__(self, "priority", int(priority))
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "tags", tuple(str(tag) for tag in tags))
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(metadata or {}))
        )
        object.__setattr__(
            self,
            "structure_id",
            new_structure_id()
            if structure_id is None
            else reserve_structure_id(int(structure_id)),
        )
        if material_id is None:
            inherited_material_id = getattr(material, "material_id", None)
            resolved_material_id = (
                new_material_id()
                if inherited_material_id is None
                else reserve_material_id(int(inherited_material_id))
            )
        else:
            resolved_material_id = reserve_material_id(int(material_id))
        object.__setattr__(self, "material_id", resolved_material_id)
        object.__setattr__(
            self,
            "assignment_id",
            new_assignment_id()
            if assignment_id is None
            else reserve_assignment_id(int(assignment_id)),
        )
        object.__setattr__(
            self,
            "surface_id",
            new_surface_id()
            if surface_id is None
            else reserve_surface_id(int(surface_id)),
        )
        object.__setattr__(self, "primitive_ids", normalized_primitive_ids)
        object.__setattr__(self, "phase_screen", phase_screen)
        object.__setattr__(self, "uv", uv)
        object.__setattr__(self, "face_uv", face_uv)

    def primitive_id(self, local_index: int) -> PrimitiveId:
        index = int(local_index)
        if index < 0:
            raise ValueError("local_index must be non-negative.")
        if index >= (1 << 32):
            raise ValueError("local_index must fit in 32 bits.")
        if self.primitive_ids is not None:
            try:
                return self.primitive_ids[index]
            except IndexError as exc:
                raise IndexError("local_index is outside primitive_ids.") from exc
        return PrimitiveId((int(self.structure_id) << 32) | index)
