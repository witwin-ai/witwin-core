"""ExtrudedPolygon: exact SDF, Prism equivalence, mesh/SDF agreement, bounds, validation, autograd, device."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from witwin.core import Box, ExtrudedPolygon, Prism, geometry_from_dict, geometry_to_dict

L_SHAPE = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]
FRAME = {"position": (0.2, -0.3, 0.4), "rotation": (0.3, -0.2, 0.1)}


def _regular(radius, sides, dtype=torch.float32):
    angles = torch.linspace(0, 2 * np.pi, sides + 1)[:-1]
    return torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1).to(dtype)


def _points(count=4000, scale=1.6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(3, count, generator=generator) * 2 - 1) * scale


def _mesh_volume(geometry):
    vertices, faces = geometry.to_mesh()
    return torch.linalg.det(vertices.double()[faces]).sum().item() / 6


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_regular_polygon_equals_prism(axis):
    """Prism vertices give the same SDF bit for bit, the same volume and the same bounds."""
    prism = Prism(radius=0.7, height=1.3, num_sides=5, axis=axis, **FRAME)
    extruded = ExtrudedPolygon(_regular(0.7, 5), height=1.3, axis=axis, **FRAME)
    points = _points()
    assert torch.equal(prism.signed_distance(*points), extruded.signed_distance(*points))
    assert _mesh_volume(extruded) == pytest.approx(_mesh_volume(prism), rel=1e-6)
    prism_vertices, _ = prism.to_mesh()
    expected = torch.stack((prism_vertices[:10].amin(0), prism_vertices[:10].amax(0)))
    torch.testing.assert_close(extruded.bounds_world, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("reverse", [False, True])
def test_nonconvex_mesh_is_closed_outward_and_on_the_surface(axis, reverse):
    vertices = L_SHAPE[::-1] if reverse else L_SHAPE
    geometry = ExtrudedPolygon(vertices, height=0.5, axis=axis, **FRAME)
    mesh_vertices, faces = geometry.to_mesh()
    assert faces.shape == (2 * 6 + 2 * 4, 3)
    assert _mesh_volume(geometry) == pytest.approx(3.0 * 0.5, rel=1e-6)
    assert geometry.signed_distance(*mesh_vertices.unbind(1)).abs().max().item() <= 1e-6
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    forward = {tuple(e) for e in edges.tolist()}
    assert all((b, a) in forward for a, b in forward)  # every directed edge has its twin: closed, consistent
    bounds = geometry.bounds_world
    torch.testing.assert_close(bounds, torch.stack((mesh_vertices.amin(0), mesh_vertices.amax(0))), rtol=0, atol=1e-6)


def test_nonconvex_signed_distance_is_exact():
    """Reflex-corner region, notch interior, axial caps and the 3D corner use the analytic distances."""
    geometry = ExtrudedPolygon(L_SHAPE, height=1.0, position=(0.0, 0.0, 0.0))
    cases = {
        (1.5, 1.5, 0.0): 0.5,                    # outside the notch, nearest the edge y = 1
        (1.3, 1.4, 0.0): 0.3,                    # outside the notch, nearest the edge x = 1
        (1.2, 1.6, 0.0): 0.2,                    # outside, nearest the edge x = 1
        (0.5, 0.5, 0.0): -0.5,                   # inside the arm: nearest edges, axial cap at 0.5
        (0.5, 1.5, 0.2): -0.3,                   # inside: nearest cap z = 0.5
        (2.5, 0.5, 0.9): math.hypot(0.5, 0.4),   # outside the side and the cap
        (0.9, 0.9, 0.0): -math.hypot(0.1, 0.1),  # inside, nearest the reflex vertex (1, 1)
    }
    for point, expected in cases.items():
        value = geometry.signed_distance(*(torch.tensor(v, dtype=torch.float64) for v in point)).item()
        assert value == pytest.approx(expected, abs=1e-12), point


@pytest.mark.parametrize("dtype,rel", [(torch.float64, 1e-12), (torch.float32, 2e-6)])
@pytest.mark.parametrize("scale", [1e-9, 1e-6, 1.0, 1e3])
def test_signed_distance_is_scale_free(dtype, rel, scale):
    """Exact distances and signs for a hexagon at any length scale, float32 included.

    Inside at 0.5 R: nearest edge at 0.433 R (the cap is at 0.5 R); outside: nearest vertex.
    """
    angles = torch.linspace(0, 2 * np.pi, 7, dtype=torch.float64)[:-1]
    vertices = (torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale).to(dtype)
    geometry = ExtrudedPolygon(vertices, height=torch.tensor(scale, dtype=dtype))
    for x, expected in ((0.5, -0.25 * math.sqrt(3)), (2.0, 1.0), (10.0, 9.0), (1000.0, 999.0)):
        value = geometry.signed_distance(*(torch.tensor(v * scale, dtype=dtype) for v in (x, 0.0, 0.0))).item()
        assert value == pytest.approx(expected * scale, rel=rel), (x, value / scale)


def test_box_cross_section_matches_box_signed_distance():
    geometry = ExtrudedPolygon([(-0.5, -0.4), (0.5, -0.4), (0.5, 0.4), (-0.5, 0.4)], height=0.6, **FRAME)
    box = Box(size=(1.0, 0.8, 0.6), **FRAME)
    points = _points(seed=1)
    torch.testing.assert_close(geometry.signed_distance(*points), box.signed_distance(*points), rtol=0, atol=4e-7)
    torch.testing.assert_close(geometry.to_mask(*points, beta=0.05), box.to_mask(*points, beta=0.05), rtol=0, atol=1e-6)


def test_collinear_vertices_keep_positive_triangles():
    vertices = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0), (0.0, 1.0)]
    geometry = ExtrudedPolygon(vertices, height=1.0)
    mesh_vertices, faces = geometry.to_mesh()
    cap = faces[len(vertices) * 2:]
    a, b, c = (mesh_vertices[cap[:, k]].double() for k in range(3))
    areas = torch.linalg.cross(b - a, c - a).norm(dim=1) / 2
    assert bool((areas > 1e-6).all())
    assert _mesh_volume(geometry) == pytest.approx(3.0, rel=1e-6)


@pytest.mark.parametrize("vertices,message", [
    ([(0, 0), (1, 1), (1, 0), (0, 1)], "nonzero area|simple"),
    ([(0, 0), (2, 0), (2, 2), (1, -1), (0, 2)], "simple"),
    ([(0, 0), (2, 0), (1, 0), (1, 1)], "simple"),
    ([(0, 0), (1, 0), (1, 0), (0, 1)], "repeat"),
    ([(0, 0), (1, 0), (1, 1), (0, 0)], "repeat"),
    ([(0, 0), (1, 0), (2, 0)], "nonzero area"),
    ([(0, 0), (1, 0)], "N >= 3"),
    ([(0, 0), (4, 0), (4, 4), (2, 0.0), (0, 4)], "simple"),
    ([(0, 0), (1, 0), (float("nan"), 1)], "finite"),
])
def test_rejects_non_simple_polygons(vertices, message):
    with pytest.raises(ValueError, match=message):
        ExtrudedPolygon(vertices)


def test_validation_is_scale_free():
    """A 1 um L-shaped via outline in SI behaves like the unit outline."""
    scale = 1e-6
    small = ExtrudedPolygon(torch.tensor(L_SHAPE, dtype=torch.float64) * scale, height=torch.tensor(scale, dtype=torch.float64))
    assert _mesh_volume(small) == pytest.approx(3.0 * scale ** 3, rel=1e-5)


def test_serialization_round_trip_keeps_fields():
    geometry = ExtrudedPolygon(torch.tensor(L_SHAPE, dtype=torch.float64), height=0.4, axis="y", **FRAME)
    rebuilt = geometry_from_dict(geometry_to_dict(geometry))
    assert type(rebuilt) is ExtrudedPolygon and rebuilt.axis == "y"
    assert rebuilt.vertices.dtype == torch.float64 and torch.equal(rebuilt.vertices, geometry.vertices)
    points = _points(seed=2)
    assert torch.equal(rebuilt.signed_distance(*points), geometry.signed_distance(*points))


def test_occupancy_gradients_reach_vertices_height_position_and_rotation():
    vertices = torch.tensor(L_SHAPE, dtype=torch.float64, requires_grad=True)
    height = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    position = torch.tensor((0.1, 0.0, 0.0), dtype=torch.float64, requires_grad=True)
    rotation = torch.tensor((0.1, 0.05, 0.2), dtype=torch.float64, requires_grad=True)
    geometry = ExtrudedPolygon(vertices, height=height, position=position, rotation=rotation)
    axis = torch.linspace(-0.5, 2.5, 25, dtype=torch.float64)
    xx, yy, zz = torch.meshgrid(axis, axis, torch.linspace(-0.6, 0.6, 13, dtype=torch.float64), indexing="ij")
    total = geometry.to_mask(xx, yy, zz).sum()
    grads = torch.autograd.grad(total, (vertices, height, position, rotation))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    assert geometry.vertices.dtype == torch.float64


def test_vertex_gradcheck_through_signed_distance():
    point = tuple(torch.tensor(v, dtype=torch.float64) for v in (1.3, 1.7, 0.1))

    def distance(vertices):
        return ExtrudedPolygon(vertices, height=1.0).signed_distance(*point)

    vertices = torch.tensor(L_SHAPE, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(distance, (vertices,), eps=1e-6, atol=1e-8)


def test_nested_tensor_vertices_match_stacked():
    theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    nested = ExtrudedPolygon([(0.0, 0.0), (1.0 + theta, 0.0), (1.0, 1.0), (0.0, 1.0)])
    stacked = ExtrudedPolygon(torch.stack([
        torch.stack([theta.new_tensor(0.0), theta.new_tensor(0.0)]),
        torch.stack([1.0 + theta, theta.new_tensor(0.0)]),
        torch.stack([theta.new_tensor(1.0), theta.new_tensor(1.0)]),
        torch.stack([theta.new_tensor(0.0), theta.new_tensor(1.0)]),
    ]))
    assert nested.vertices.dtype == torch.float64
    torch.testing.assert_close(nested.vertices, stacked.vertices, rtol=0, atol=0)
    assert torch.autograd.grad(nested.vertices.sum(), theta)[0] == 1.0


def test_python_vertices_are_float32_and_device_defaults_to_none():
    import inspect
    geometry = ExtrudedPolygon(L_SHAPE)
    assert geometry.vertices.dtype == torch.float32
    assert geometry.vertices.device.type == "cpu"
    assert inspect.signature(ExtrudedPolygon).parameters["device"].default is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device check")
def test_device_follows_authored_tensors_and_conflicts_raise():
    vertices = torch.tensor(L_SHAPE, device="cuda")
    geometry = ExtrudedPolygon(vertices, height=0.5)
    assert geometry.position.device.type == "cuda" and geometry.height.device.type == "cuda"
    with pytest.raises(ValueError, match="conflicts"):
        ExtrudedPolygon(vertices, position=torch.zeros(3))
