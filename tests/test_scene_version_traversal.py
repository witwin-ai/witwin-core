"""Dispatch contract for the scene version traversal.

The four ``Scene`` version properties walk the whole world model, and
``witwin.channel.scene.compile`` reads all four before it consults its cache,
so this walk sits on the per-solve path. It resolves each node's kind once per
class and skips scalar leaves without a recursive call.

These tests pin what the walk must observe, so a future optimization cannot
quietly stop noticing a mutation. ``test_world_contracts`` and
``test_dynamics_contracts`` cover the version semantics themselves; this file
covers the traversal that feeds them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

import torch

from witwin.core.scene import _tensor_states


def _state_count(value) -> int:
    return len(_tensor_states(value))


def test_walk_reaches_tensors_through_every_container_kind():
    tensor = torch.zeros(3)

    @dataclass
    class Leaf:
        payload: torch.Tensor
        label: str = "ignored"

    class Plain:
        def __init__(self, item):
            self.item = item

    assert _state_count(tensor) == 1
    assert _state_count([tensor]) == 1
    assert _state_count((tensor,)) == 1
    assert _state_count({"k": tensor}) == 1
    assert _state_count(MappingProxyType({"k": tensor})) == 1
    assert _state_count(Leaf(tensor)) == 1
    assert _state_count(Plain(tensor)) == 1
    assert _state_count([{"a": (Leaf(tensor),)}]) == 1


def test_scalar_leaves_contribute_nothing():
    for leaf in (None, True, 7, 1.5, 2j, "text", b"bytes", bytearray(b"x")):
        assert _tensor_states(leaf) == ()
    assert _state_count({"a": 1, "b": None, "c": "x"}) == 0


def test_repeated_objects_are_visited_once():
    tensor = torch.zeros(3)
    assert _state_count([tensor, tensor]) == 1

    shared = {"t": tensor}
    assert _state_count([shared, shared]) == 1


def test_reference_cycles_terminate():
    tensor = torch.zeros(3)
    cyclic: list = [tensor]
    cyclic.append(cyclic)
    assert _state_count(cyclic) == 1


def test_cache_suffixed_attributes_are_excluded():
    class WithCache:
        def __init__(self):
            self.value = torch.zeros(3)
            self.bvh_cache = torch.zeros(3)

    assert _state_count(WithCache()) == 1


def test_every_observed_tensor_attribute_moves_the_state():
    base = torch.zeros((2, 3))
    reference = _tensor_states(base, include_identity=False)

    assert _tensor_states(base, include_identity=False) == reference
    assert _tensor_states(torch.zeros((3, 2)), include_identity=False) != reference
    assert (
        _tensor_states(torch.zeros((2, 3), dtype=torch.float64), include_identity=False)
        != reference
    )
    assert (
        _tensor_states(torch.zeros((2, 3), requires_grad=True), include_identity=False)
        != reference
    )
    assert _tensor_states(base.t(), include_identity=False) != reference

    mutated = torch.zeros((2, 3))
    before = _tensor_states(mutated, include_identity=False)
    mutated[0, 0] = 1.0
    assert _tensor_states(mutated, include_identity=False) != before


def test_identity_is_optional_and_excluded_state_is_position_stable():
    tensor = torch.zeros(3)
    with_identity = _tensor_states(tensor, include_identity=True)[0]
    without_identity = _tensor_states(tensor, include_identity=False)[0]

    assert len(with_identity) == len(without_identity) + 2
    assert with_identity[2:] == without_identity


def test_mapping_traversal_order_follows_sorted_key_repr():
    first = torch.zeros(1)
    second = torch.ones(1)
    forward = _tensor_states({"a": first, "b": second}, include_identity=False)
    reversed_insertion = _tensor_states(
        {"b": second, "a": first}, include_identity=False
    )
    assert forward == reversed_insertion


def test_dataclass_traversal_follows_declared_field_order():
    @dataclass
    class Pair:
        first: torch.Tensor
        second: torch.Tensor = field(default_factory=lambda: torch.ones(1))

    pair = Pair(torch.zeros(1))
    states = _tensor_states(pair, include_identity=False)
    assert states == (
        _tensor_states(pair.first, include_identity=False)[0],
        _tensor_states(pair.second, include_identity=False)[0],
    )
