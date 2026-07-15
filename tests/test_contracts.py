from __future__ import annotations

import pytest
import torch

from witwin.core import (
    FrequencyMaterialSample,
    GeometrySpec,
    MaterialCapabilities,
    MaterialSpec,
    StaticMaterialSample,
    Structure,
)


class DuckGeometry:
    kind = "duck"

    def to_mesh(self, segments: int = 16):
        del segments
        return torch.zeros((3, 3)), torch.tensor([[0, 1, 2]])


class DuckMaterial:
    name = "duck"

    def capabilities(self):
        return MaterialCapabilities()

    def evaluate_static(self):
        return StaticMaterialSample(eps_r=2.0)

    def evaluate_at_frequency(self, frequency: float):
        del frequency
        return FrequencyMaterialSample(eps_r=2.0)


def test_structure_accepts_protocol_conforming_values():
    geometry = DuckGeometry()
    material = DuckMaterial()

    assert isinstance(geometry, GeometrySpec)
    assert isinstance(material, MaterialSpec)
    assert Structure(geometry=geometry, material=material).geometry is geometry


@pytest.mark.parametrize(
    ("geometry", "material", "message"),
    [
        (object(), DuckMaterial(), "GeometrySpec"),
        (DuckGeometry(), object(), "MaterialSpec"),
    ],
)
def test_structure_rejects_values_outside_shared_contracts(geometry, material, message):
    with pytest.raises(TypeError, match=message):
        Structure(geometry=geometry, material=material)
