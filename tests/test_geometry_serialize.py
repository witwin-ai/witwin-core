from __future__ import annotations

import inspect
import json

import pytest
import torch

from witwin.core import (
    Box,
    Cone,
    Cylinder,
    Difference,
    Ellipsoid,
    GeometryBase,
    HollowBox,
    Intersection,
    Mesh,
    Prism,
    Pyramid,
    Sphere,
    Torus,
    Union,
    geometry_from_dict,
    geometry_to_dict,
)
from witwin.core.geometry.base import _rotation_quaternion


def _geometry_cases():
    frame = {"position": (0.2, -0.3, 0.4), "rotation": (0.3, -0.2, 0.1)}
    vertices, faces = Box(position=(2, 3, 4), size=(1.2, 0.8, 1.4)).to_mesh()
    mesh = Mesh(
        vertices, faces, scale=(0.7, 1.1, 0.9), recenter=True,
        fill_mode="surface", surface_thickness=torch.tensor(0.07),
        source_path="/authored/scene.obj", topology_diagnostics=True, **frame,
    )
    a = Box(size=(1.3, 0.9, 1.1), **frame)
    b = Sphere(position=(-0.1, 0.2, 0.3), radius=0.6, rotation=(0.1, 0.2, -0.3))
    return [
        Box(size=(1.3, 0.9, 1.1), position=frame["position"], rotation=(1.2, 0.1, -0.2, 0.3)),
        Sphere(radius=0.7, position=frame["position"], rotation=torch.tensor([0.3, -0.2, 0.1])),
        Cylinder(radius=0.4, height=1.3, axis="x", **frame),
        Ellipsoid(radii=(0.7, 0.9, 1.2), **frame),
        Cone(radius=0.6, height=1.3, axis="y", **frame),
        Pyramid(base_size=0.8, height=1.3, axis="x", **frame),
        Prism(radius=0.6, height=1.3, num_sides=5, axis="y", **frame),
        Torus(major_radius=0.8, minor_radius=0.2, axis="x", **frame),
        HollowBox(outer_size=(1.3, 1.1, 0.9), inner_size=(0.7, 0.5, 0.3), **frame),
        mesh,
        Union(a, b, smoothness=0.12, **frame),
        Intersection(a, b, smoothness=0.08, **frame),
        Difference(a, b, Sphere(radius=0.2), smoothness=0.04, **frame),
        Difference(Union(a, b, smoothness=0.09, **frame), mesh, smoothness=0.03, **frame),
        Mesh(vertices, faces, scale=(0.8, 0.7, 0.6), recenter=False,
             fill_mode="solid", topology_diagnostics=False, **frame),
    ]


def _assert_field_equal(expected, actual):
    if isinstance(expected, GeometryBase):
        _assert_constructor_fields_equal(expected, actual)
    elif isinstance(expected, torch.Tensor):
        assert torch.equal(expected, torch.as_tensor(actual))
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            _assert_field_equal(left, right)
    else:
        assert expected == actual


def _assert_constructor_fields_equal(expected, actual):
    assert type(expected) is type(actual)
    for name in inspect.signature(type(expected)).parameters:
        if name == "device":
            continue
        left, right = getattr(expected, name), getattr(actual, name)
        if name == "rotation":
            left, right = _rotation_quaternion(left), _rotation_quaternion(right)
        _assert_field_equal(left, right)


def test_round_trip_every_geometry():
    points = 4 * torch.rand((1000, 1, 3), generator=torch.Generator().manual_seed(17)) - 2
    coords = points.unbind(dim=-1)
    for geometry in _geometry_cases():
        d = geometry_to_dict(geometry)
        assert d == geometry.to_dict()
        assert d["rotation"] == _rotation_quaternion(geometry.rotation).tolist()
        restored = geometry_from_dict(json.loads(json.dumps(d)), device="cpu")
        _assert_constructor_fields_equal(geometry, restored)
        _assert_constructor_fields_equal(geometry, GeometryBase.from_dict(d))
        expected_sdf = geometry.signed_distance(*coords)
        actual_sdf = restored.signed_distance(*coords)
        assert expected_sdf.numel() == actual_sdf.numel() == 1000
        torch.testing.assert_close(actual_sdf, expected_sdf, atol=1e-6, rtol=0)
        expected_vertices, expected_faces = geometry.to_mesh(segments=4)
        actual_vertices, actual_faces = restored.to_mesh(segments=4)
        assert torch.equal(actual_vertices, expected_vertices)
        assert torch.equal(actual_faces, expected_faces)

    class UnsupportedBox(Box):
        pass

    with pytest.raises(TypeError, match="not in witwin.scene/1: UnsupportedBox"):
        geometry_to_dict(UnsupportedBox())


def test_every_constructor_parameter_is_serialized():
    for geometry in _geometry_cases() + [Box(), Union(Box()), Difference(Box(), Sphere())]:
        expected_keys = set(inspect.signature(type(geometry)).parameters) - {"device"}
        assert set(geometry_to_dict(geometry)) == expected_keys | {"type"}


def test_large_arrays_by_reference():
    vertices, faces = Sphere(radius=0.7).to_mesh(segments=32)
    mesh = Mesh(
        vertices, faces, position=(0.2, -0.1, 0.3), rotation=(0.1, 0.3, -0.2),
        scale=(0.8, 1.2, 0.9), recenter=True, fill_mode="solid", topology_diagnostics=False,
    )
    assert vertices.numel() > 4096
    arrays = {}
    d = mesh.to_dict(arrays=arrays)
    assert arrays[d["vertices"]["array"]] is vertices
    assert arrays[d["faces"]["array"]] is faces
    assert isinstance(d["position"], list)
    restored = Mesh.from_dict(json.loads(json.dumps(d)), arrays=arrays, device="cpu")
    _assert_constructor_fields_equal(mesh, restored)
    for expected, actual in zip(mesh.to_mesh(), restored.to_mesh()):
        assert torch.equal(expected, actual)

    arrays = {}
    tree = Union(mesh, mesh, position=(0.1, 0.2, 0.3), rotation=(0.2, -0.1, 0.3))
    d = geometry_to_dict(tree, arrays=arrays)
    assert len(arrays) == 4
    assert d["geometries"][0]["vertices"] != d["geometries"][1]["vertices"]
    _assert_constructor_fields_equal(tree, geometry_from_dict(d, arrays=arrays))
