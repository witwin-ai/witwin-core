"""Scale-free polygon distances: Prism and Cone signs and magnitudes at micrometre and metre scale, float32/64."""

from __future__ import annotations

import math

import pytest
import torch

from witwin.core import Cone, Prism
from witwin.core.geometry.polygon import polygon_loops_signed_distance_2d


def _distance(geometry, point, dtype):
    return geometry.signed_distance(*(torch.tensor(v, dtype=dtype) for v in point)).item()


# Prism builds its polygon angles in float32, so float64 Prism distances carry float32 vertex rounding.
@pytest.mark.parametrize("dtype,rel", [(torch.float64, 1e-6), (torch.float32, 2e-6)])
@pytest.mark.parametrize("scale", [1e-6, 1.0])
def test_prism_distance_is_scale_free(dtype, rel, scale):
    """Hexagon of circumradius R, height R: inside at 0.5 R (nearest edge 0.433 R), outside 2/10/1000 R."""
    prism = Prism(radius=torch.tensor(scale, dtype=dtype), height=torch.tensor(scale, dtype=dtype), num_sides=6)
    for x, expected in ((0.5, -0.25 * math.sqrt(3)), (2.0, 1.0), (10.0, 9.0), (1000.0, 999.0)):
        assert _distance(prism, (x * scale, 0.0, 0.0), dtype) == pytest.approx(expected * scale, rel=rel), x


@pytest.mark.parametrize("dtype,rel", [(torch.float64, 1e-9), (torch.float32, 2e-6)])
@pytest.mark.parametrize("scale", [1e-6, 1.0])
def test_cone_distance_is_scale_free(dtype, rel, scale):
    """Cone radius R, height R (apex at the origin, base at z = R): points beyond the base plane on the axis."""
    cone = Cone(radius=torch.tensor(scale, dtype=dtype), height=torch.tensor(scale, dtype=dtype))
    for z, expected in ((2.0, 1.0), (10.0, 9.0)):
        assert _distance(cone, (0.0, 0.0, z * scale), dtype) == pytest.approx(expected * scale, rel=rel), z
    assert _distance(cone, (0.0, 0.0, 0.75 * scale), dtype) < 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_polygon_loop_edge_distance_below_float32_eps_edges(dtype):
    """A 100 um square: the distance from a point beside an edge is to the edge, not to a vertex."""
    side = 1e-4
    square = torch.tensor([(0, 0), (side, 0), (side, side), (0, side)], dtype=dtype)
    point = torch.tensor([[side / 2, -side / 10]], dtype=dtype)
    value = polygon_loops_signed_distance_2d(point, [square]).item()
    assert value == pytest.approx(side / 10, rel=1e-5)
