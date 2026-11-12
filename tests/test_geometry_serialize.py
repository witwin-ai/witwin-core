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
    ExtrudedPolygon,
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
        ExtrudedPolygon([(0.0, 0.0), (0.8, 0.0), (0.8, 0.3), (0.3, 0.3), (0.3, 0.7), (0.0, 0.7)],
                        height=1.3, axis="x", **frame),
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
        assert isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype
        assert torch.equal(expected, actual)
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
        quaternion = _rotation_quaternion(geometry.rotation)
        assert d["rotation"] == {"type": "Tensor", "dtype": str(quaternion.dtype).removeprefix("torch."),
                                 "values": quaternion.tolist()}
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
    assert d["position"] == {"type": "Tensor", "dtype": "float32", "values": mesh.position.tolist()}
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


def _float64(*values):
    return torch.tensor(values, dtype=torch.float64)


def _dtype_cases():
    vertices, faces = Box(position=(0.2, 0.1, 0.3), size=(1 / 3, 0.7, 0.9)).to_mesh()
    vertices64, faces32 = vertices.to(torch.float64) / 3, faces.to(torch.int32)
    box = Box(position=_float64(0.1, 1 / 3, -0.0007), size=_float64(0.03747405725, 0.2, 1e-3),
              rotation=_float64(0.9, 0.1, -0.2, 0.3))
    sphere = Sphere(position=(0.1, 0.2, 0.3), radius=torch.tensor(0.0013879280462962963, dtype=torch.float64))
    return [
        box,
        sphere,
        Cylinder(radius=torch.tensor(0.1, dtype=torch.float64), height=1.3, axis="x"),
        Ellipsoid(radii=_float64(0.7, 0.9, 1.1), position=_float64(0.0, 0.0, 0.0)),
        Union(box, sphere),
        Union(box, sphere, smoothness=0.0),
        Union(box, sphere, smoothness=torch.tensor(0.0)),
        Difference(box, sphere, smoothness=torch.tensor(0.01, dtype=torch.float64)),
        Mesh(vertices64, faces32, topology_diagnostics=True, recenter=False),
        Mesh(vertices64, faces32, topology_diagnostics=False, fill_mode="solid", recenter=False),
        Mesh(vertices64, faces32, recenter=False),
        Mesh(vertices64.numpy(), faces32.numpy(), recenter=False),
        Mesh(vertices, faces32, topology_diagnostics=False, fill_mode="solid", recenter=False),
    ]


def _assert_same_geometry(expected, actual):
    _assert_constructor_fields_equal(expected, actual)
    if isinstance(expected, Union | Intersection | Difference):
        assert actual.smooth == expected.smooth
    if isinstance(expected, Mesh):
        assert actual.topology_diagnostics == expected.topology_diagnostics
        assert actual.is_watertight == expected.is_watertight


def test_tensor_dtypes_round_trip_exactly():
    for geometry in _dtype_cases():
        d = geometry_to_dict(geometry)
        text = json.dumps(d)
        restored = geometry_from_dict(json.loads(text), device="cpu")
        _assert_same_geometry(geometry, restored)
        assert json.dumps(geometry_to_dict(restored)) == text
    union = geometry_to_dict(Union(Box(), Sphere(), smoothness=0.0))
    assert union["smoothness"] == 0.0
    assert geometry_to_dict(Box(size=_float64(0.1, 0.2, 0.3)))["size"] == {
        "type": "Tensor", "dtype": "float64", "values": [0.1, 0.2, 0.3]}


def _legacy(value):
    """What the writers before dtype tags emitted: bare nested lists for inline tensors."""
    if isinstance(value, dict):
        if value.get("type") == "Tensor":
            return value["values"]
        return {key: _legacy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_legacy(item) for item in value]
    return value


def test_legacy_bare_lists():
    # Python-authored geometry stores float32 values, which the old bare lists carry exactly.
    # Mesh.surface_thickness keeps the authored type, which bare documents do not record.
    def thickness_as_float(geometry):
        for child in getattr(geometry, "geometries", ()):
            thickness_as_float(child)
        if isinstance(geometry, Mesh) and geometry.surface_thickness is not None:
            geometry.surface_thickness = float(geometry.surface_thickness)
        return geometry

    for geometry in _geometry_cases():
        restored = geometry_from_dict(json.loads(json.dumps(_legacy(geometry_to_dict(geometry)))))
        _assert_constructor_fields_equal(thickness_as_float(geometry), thickness_as_float(restored))
    # A float64 field holding any value float32 cannot represent comes back float64 and exact.
    box, sphere = _dtype_cases()[:2]
    for geometry in (box, sphere, _dtype_cases()[7]):
        _assert_same_geometry(geometry, geometry_from_dict(json.loads(json.dumps(_legacy(geometry_to_dict(geometry))))))
    text = ('{"type": "box", "position": [0.0, 0.0, 0.0015], "size": [0.02, 0.5, 0.25], '
            '"rotation": [1.0, 0.0, 0.0, 0.0]}')
    legacy = geometry_from_dict(json.loads(text))
    assert legacy.position.dtype == legacy.size.dtype == torch.float64
    assert legacy.position.tolist() == [0.0, 0.0, 0.0015] and legacy.size.tolist() == [0.02, 0.5, 0.25]
    assert legacy.rotation.dtype == torch.float32
    # Values float32 represents exactly keep the old float32 reading; old integer lists stay int64.
    mesh = geometry_from_dict(_legacy(geometry_to_dict(_dtype_cases()[-1])))
    assert mesh.vertices.dtype == torch.float32 and mesh.faces.dtype == torch.int64
    assert torch.equal(mesh.vertices, _dtype_cases()[-1].vertices)
