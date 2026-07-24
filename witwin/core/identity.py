"""Stable logical identities for Core world contracts.

Two allocation modes exist, and the difference matters to any caller that
compares results across processes.

``new_*_id()`` draws from a process-global monotonic counter. IDs are unique
and stable for the lifetime of one process, but they depend on how many
objects were constructed earlier in that process. The same scene description
built in a different order, or in a different process, gets a different set of
IDs.

``reserve_*_id(value)`` records a caller-chosen identity verbatim. This is the
mode to use whenever IDs must be reproducible: a scene loaded from a file, a
scene compared against a stored result, or an ``EndpointBatch.stable_ids``
vector that has to mean the same thing in a later run. Solver-owned compilers
key their caches on these identities, so a caller that mixes the two modes for
one logical world can defeat cache reuse.

Core does not derive IDs from content. Choosing a reproducible identity scheme
is the caller's decision, not a Core default.
"""

from __future__ import annotations

from threading import Lock
from typing import NewType


StructureId = NewType("StructureId", int)
MaterialId = NewType("MaterialId", int)
AssignmentId = NewType("AssignmentId", int)
PrimitiveId = NewType("PrimitiveId", int)
AntennaId = NewType("AntennaId", int)
SurfaceId = NewType("SurfaceId", int)

_NEXT_STRUCTURE_ID = 0
_NEXT_MATERIAL_ID = 0
_NEXT_ASSIGNMENT_ID = 0
_NEXT_ANTENNA_ID = 0
_NEXT_SURFACE_ID = 0
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


def new_surface_id() -> SurfaceId:
    global _NEXT_SURFACE_ID
    with _ID_LOCK:
        value = _NEXT_SURFACE_ID
        _NEXT_SURFACE_ID += 1
    return SurfaceId(value)


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


def reserve_surface_id(value: int) -> SurfaceId:
    global _NEXT_SURFACE_ID
    resolved = _validated_id(value, name="surface_id")
    with _ID_LOCK:
        _NEXT_SURFACE_ID = max(_NEXT_SURFACE_ID, resolved + 1)
    return SurfaceId(resolved)


__all__ = [
    "AntennaId",
    "AssignmentId",
    "MaterialId",
    "PrimitiveId",
    "StructureId",
    "SurfaceId",
]
