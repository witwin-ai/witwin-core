from __future__ import annotations

import pytest
import torch

from witwin.core import Box, Cylinder, Mesh, Sphere


def _grid():
    axis = torch.linspace(-1.0, 1.0, 33, dtype=torch.float32)
    xx, yy, zz = torch.meshgrid(axis, axis, axis, indexing="ij")
    return xx, yy, zz


@pytest.mark.parametrize(
    ("geometry", "segments", "inside_point", "outside_point", "expected_faces"),
    [
        (
            Box(position=(0.1, -0.2, 0.3), size=(0.6, 0.4, 0.8)),
            16,
            (0.1, -0.2, 0.3),
            (0.8, -0.2, 0.3),
            12,
        ),
        (
            Sphere(position=(-0.2, 0.0, 0.1), radius=0.35),
            4,
            (-0.2, 0.0, 0.1),
            (0.3, 0.0, 0.1),
            48,
        ),
        (
            Cylinder(position=(0.0, 0.2, -0.1), radius=0.25, height=0.7, axis="z"),
            16,
            (0.0, 0.2, -0.1),
            (0.4, 0.2, -0.1),
            64,
        ),
    ],
)
def test_geometry_construction_to_mesh_and_to_mask(geometry, segments, inside_point, outside_point, expected_faces):
    vertices, faces = geometry.to_mesh(segments=segments)

    assert isinstance(vertices, torch.Tensor)
    assert isinstance(faces, torch.Tensor)
    assert vertices.shape[1] == 3
    assert faces.shape == (expected_faces, 3)
    assert faces.dtype == torch.int64

    centroid = vertices.mean(dim=0)
    assert torch.allclose(centroid, geometry.position, atol=5e-2)

    xx, yy, zz = _grid()
    occupancy = geometry.to_mask(xx, yy, zz)
    inside_value = geometry.to_mask(
        torch.tensor(inside_point[0]),
        torch.tensor(inside_point[1]),
        torch.tensor(inside_point[2]),
    ).item()
    outside_value = geometry.to_mask(
        torch.tensor(outside_point[0]),
        torch.tensor(outside_point[1]),
        torch.tensor(outside_point[2]),
    ).item()

    assert occupancy.dtype.is_floating_point
    assert torch.all((occupancy >= 0.0) & (occupancy <= 1.0))
    assert inside_value > 0.5
    assert outside_value < 0.5
    assert torch.any(occupancy > 0.5)
    assert torch.any(occupancy < 0.5)


def test_mesh_roundtrip_preserves_world_vertices_and_faces():
    base = Box(position=(0.25, -0.15, 0.4), size=(0.5, 0.3, 0.7))
    vertices, faces = base.to_mesh()
    mesh = Mesh(vertices, faces, position=(0.0, 0.0, 0.0), recenter=False)

    roundtrip_vertices, roundtrip_faces = mesh.to_mesh()

    assert isinstance(roundtrip_vertices, torch.Tensor)
    assert isinstance(roundtrip_faces, torch.Tensor)
    assert torch.allclose(roundtrip_vertices, vertices)
    assert torch.equal(roundtrip_faces, faces)


def test_mesh_dynamic_views_and_bounds_follow_tensor_updates():
    vertices = torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
            [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
        ],
        dtype=torch.int64,
    )
    position = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
    mesh = Mesh(vertices, faces, position=position, recenter=False)

    before_vertices = mesh.world_vertices_tensor().clone()
    before_bounds = mesh.bounds_world
    before_key = mesh.geometry_state_key()

    with torch.no_grad():
        position.copy_(torch.tensor([0.25, -0.10, 0.05], dtype=torch.float32))
        mesh._vertices_tensor[0, 0] = -0.75

    after_vertices = mesh.world_vertices_tensor()
    after_bounds = mesh.bounds_world
    after_key = mesh.geometry_state_key()

    assert not torch.allclose(before_vertices, after_vertices)
    assert before_bounds != after_bounds
    assert before_key != after_key
    np_vertices = mesh.vertices
    assert np_vertices.shape == (8, 3)
    assert np_vertices[0, 0] == pytest.approx(-0.75, abs=1e-6)


def test_mesh_sdf_cache_tracks_geometry_state_and_skips_trainable_geometry():
    vertices = torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
            [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
        ],
        dtype=torch.int64,
    )
    axis = torch.linspace(-0.75, 0.75, 9, dtype=torch.float32)

    static_mesh = Mesh(vertices.clone(), faces, recenter=False)
    before = static_mesh.signed_distance(axis, axis, axis)
    assert len(static_mesh._sdf_cache) == 1
    cached_entry = next(iter(static_mesh._sdf_cache.values()))
    assert cached_entry["geometry_key"] == static_mesh.geometry_state_key()

    with torch.no_grad():
        static_mesh.position.copy_(torch.tensor([0.25, 0.0, 0.0], dtype=torch.float32))

    after = static_mesh.signed_distance(axis, axis, axis)
    assert len(static_mesh._sdf_cache) == 1
    refreshed_entry = next(iter(static_mesh._sdf_cache.values()))
    assert refreshed_entry["geometry_key"] == static_mesh.geometry_state_key()
    assert not torch.allclose(before, after)

    trainable_vertices = torch.nn.Parameter(vertices.clone())
    trainable_mesh = Mesh(trainable_vertices, faces, recenter=False)
    _ = trainable_mesh.signed_distance(axis, axis, axis)
    assert trainable_mesh._sdf_cache == {}
