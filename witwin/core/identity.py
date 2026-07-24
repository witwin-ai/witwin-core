from __future__ import annotations

from threading import Lock
from typing import NewType


StructureId = NewType("StructureId", int)
MaterialId = NewType("MaterialId", int)
AssignmentId = NewType("AssignmentId", int)
PrimitiveId = NewType("PrimitiveId", int)
AntennaId = NewType("AntennaId", int)

_NEXT_STRUCTURE_ID = 0
_NEXT_MATERIAL_ID = 0
_NEXT_ASSIGNMENT_ID = 0
_NEXT_ANTENNA_ID = 0
_ID_LOCK = Lock()


def _validated_id(value: int, *, name: str) -> int:
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name} must be non-negative.")
    return resolved


def new_structure_id() -> StructureId:
    global _NEXT_STRUCTURE_ID
    with _ID_LOCK:
        value = _NEXT_STRUCTURE_ID
        _NEXT_STRUCTURE_ID += 1
    return StructureId(value)


def new_material_id() -> MaterialId:
    global _NEXT_MATERIAL_ID
    with _ID_LOCK:
        value = _NEXT_MATERIAL_ID
        _NEXT_MATERIAL_ID += 1
    return MaterialId(value)


def new_assignment_id() -> AssignmentId:
    global _NEXT_ASSIGNMENT_ID
    with _ID_LOCK:
        value = _NEXT_ASSIGNMENT_ID
        _NEXT_ASSIGNMENT_ID += 1
    return AssignmentId(value)


def new_antenna_id() -> AntennaId:
    global _NEXT_ANTENNA_ID
    with _ID_LOCK:
        value = _NEXT_ANTENNA_ID
        _NEXT_ANTENNA_ID += 1
    return AntennaId(value)


def reserve_structure_id(value: int) -> StructureId:
    global _NEXT_STRUCTURE_ID
    resolved = _validated_id(value, name="structure_id")
    with _ID_LOCK:
        _NEXT_STRUCTURE_ID = max(_NEXT_STRUCTURE_ID, resolved + 1)
    return StructureId(resolved)


def reserve_material_id(value: int) -> MaterialId:
    global _NEXT_MATERIAL_ID
    resolved = _validated_id(value, name="material_id")
    with _ID_LOCK:
        _NEXT_MATERIAL_ID = max(_NEXT_MATERIAL_ID, resolved + 1)
    return MaterialId(resolved)


def reserve_assignment_id(value: int) -> AssignmentId:
    global _NEXT_ASSIGNMENT_ID
    resolved = _validated_id(value, name="assignment_id")
    with _ID_LOCK:
        _NEXT_ASSIGNMENT_ID = max(_NEXT_ASSIGNMENT_ID, resolved + 1)
    return AssignmentId(resolved)


def reserve_antenna_id(value: int) -> AntennaId:
    global _NEXT_ANTENNA_ID
    resolved = _validated_id(value, name="antenna_id")
    with _ID_LOCK:
        _NEXT_ANTENNA_ID = max(_NEXT_ANTENNA_ID, resolved + 1)
    return AntennaId(resolved)


__all__ = [
    "AntennaId",
    "AssignmentId",
    "MaterialId",
    "PrimitiveId",
    "StructureId",
]
