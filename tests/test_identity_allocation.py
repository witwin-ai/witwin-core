"""Allocation semantics of the two Core identity modes.

These lock in the contract documented in :mod:`witwin.core.identity`: the
``new_*`` counter is process-scoped and order-dependent, while ``reserve_*``
round-trips a caller-chosen value so a scene description can be rebuilt with
identical IDs.
"""

from __future__ import annotations

import torch

from witwin.core import Mesh, PhysicalMaterial, Structure
from witwin.core.identity import (
    new_assignment_id,
    new_material_id,
    new_structure_id,
    new_surface_id,
    reserve_assignment_id,
    reserve_material_id,
    reserve_structure_id,
    reserve_surface_id,
)


def _unit_triangle() -> Mesh:
    return Mesh(
        vertices=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
    )


def test_new_ids_are_monotonic_and_unique_within_a_process():
    for allocator in (
        new_structure_id,
        new_material_id,
        new_assignment_id,
        new_surface_id,
    ):
        first = allocator()
        second = allocator()
        assert second > first


def test_reserved_ids_round_trip_exactly():
    for reserve in (
        reserve_structure_id,
        reserve_material_id,
        reserve_assignment_id,
        reserve_surface_id,
    ):
        assert int(reserve(4321)) == 4321


def test_explicitly_reserved_structure_ids_are_reproducible():
    """The same description built twice yields identical IDs when reserved."""

    def build() -> Structure:
        return Structure(
            geometry=_unit_triangle(),
            material=PhysicalMaterial(name="brick", eps_r=4.4),
            structure_id=11,
            material_id=22,
            assignment_id=33,
            surface_id=44,
        )

    first = build()
    # Allocate unrelated IDs in between so a counter-based scheme would drift.
    for _ in range(5):
        new_structure_id()
        new_material_id()
    second = build()

    assert (
        int(first.structure_id),
        int(first.material_id),
        int(first.assignment_id),
        int(first.surface_id),
    ) == (11, 22, 33, 44)
    assert first.structure_id == second.structure_id
    assert first.material_id == second.material_id
    assert first.assignment_id == second.assignment_id
    assert first.surface_id == second.surface_id


def test_unreserved_structure_ids_drift_with_process_state():
    """Documents why reserved IDs are required for cross-run comparison."""

    def build() -> Structure:
        return Structure(
            geometry=_unit_triangle(),
            material=PhysicalMaterial(name="brick", eps_r=4.4),
        )

    first = build()
    second = build()
    assert first.structure_id != second.structure_id


def test_primitive_ids_derive_from_the_structure_id():
    structure = Structure(
        geometry=_unit_triangle(),
        material=PhysicalMaterial(name="brick", eps_r=4.4),
        structure_id=7,
    )
    assert int(structure.primitive_id(0)) == (7 << 32)
    assert int(structure.primitive_id(3)) == (7 << 32) | 3
