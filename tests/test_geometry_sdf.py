from __future__ import annotations

import pytest
import torch

from witwin.core import Box, Cone, Cylinder, Ellipsoid, HollowBox, Mesh, Prism, Pyramid, Sphere, Torus


def _scalar_occupancy(geometry, point, *, beta=0.05):
    x, y, z = (torch.tensor(coord, dtype=torch.float32) for coord in point)
    return geometry.to_mask(x, y, z, beta=beta).item()


def _asymmetric_grid():
    x = torch.linspace(-0.45, 0.35, 11, dtype=torch.float32)
    y = torch.linspace(-0.30, 0.50, 9, dtype=torch.float32)
    z = torch.linspace(-0.40, 0.20, 7, dtype=torch.float32)
    return torch.meshgrid(x, y, z, indexing="ij")


def _cube_mesh(*, size=1.0, requires_grad=False):
    half = size / 2.0
    vertices = torch.tensor(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ],
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    faces = torch.tensor(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
            [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
        ],
        dtype=torch.int64,
    )
    return vertices, faces


def _cube_mesh_with_flipped_face(*, size=1.0):
    vertices, faces = _cube_mesh(size=size)
    faces = faces.clone()
    faces[0] = faces[0, [0, 2, 1]]
    return vertices, faces


def _cube_mesh_with_degenerate_face(*, size=1.0):
    vertices, faces = _cube_mesh(size=size)
    faces = faces.clone()
    faces[0] = torch.tensor([0, 0, 1], dtype=faces.dtype)
    return vertices, faces


@pytest.mark.parametrize(
    ("geometry", "inside_point", "surface_point", "outside_point"),
    [
        (
            Box(size=(2.0, 2.0, 2.0)),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.4, 0.0, 0.0),
        ),
        (
            Sphere(radius=1.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.4, 0.0, 0.0),
        ),
        (
            Cylinder(radius=0.5, height=2.0, axis="z"),
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.8, 0.0, 0.0),
        ),
        (
            Torus(major_radius=1.0, minor_radius=0.25, axis="z"),
            (1.0, 0.0, 0.0),
            (1.25, 0.0, 0.0),
            (1.6, 0.0, 0.0),
        ),
        (
            HollowBox(outer_size=(2.0, 2.0, 2.0), inner_size=(1.0, 1.0, 1.0)),
            (0.8, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        (
            Ellipsoid(radii=(1.0, 0.75, 0.5)),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.3, 0.0, 0.0),
        ),
        (
            Cone(radius=0.5, height=1.0, axis="z"),
            (0.0, 0.0, 0.5),
            (0.25, 0.0, 0.5),
            (0.35, 0.0, 0.5),
        ),
        (
            Pyramid(base_size=1.0, height=1.0, axis="z"),
            (0.0, 0.0, 0.5),
            (0.25, 0.0, 0.5),
            (0.35, 0.0, 0.5),
        ),
        (
            Prism(radius=0.5, height=1.0, num_sides=6, axis="z"),
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.7, 0.0, 0.0),
        ),
    ],
)
def test_primitive_signed_distance_has_expected_sign(geometry, inside_point, surface_point, outside_point):
    inside = geometry.signed_distance(*(torch.tensor(coord, dtype=torch.float32) for coord in inside_point)).item()
    surface = geometry.signed_distance(*(torch.tensor(coord, dtype=torch.float32) for coord in surface_point)).item()
    outside = geometry.signed_distance(*(torch.tensor(coord, dtype=torch.float32) for coord in outside_point)).item()

    assert inside < 0.0
    assert abs(surface) < 1.0e-5
    assert outside > 0.0


@pytest.mark.parametrize(
    ("geometry", "inside_point", "boundary_point", "outside_point"),
    [
        (Box(size=(1.0, 1.0, 1.0)), (0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.8, 0.0, 0.0)),
        (Sphere(radius=0.5), (0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.8, 0.0, 0.0)),
        (Cylinder(radius=0.4, height=1.0, axis="z"), (0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (0.7, 0.0, 0.0)),
        (Torus(major_radius=0.8, minor_radius=0.2, axis="z"), (0.8, 0.0, 0.0), (1.0, 0.0, 0.0), (1.3, 0.0, 0.0)),
        (
            HollowBox(outer_size=(1.4, 1.4, 1.4), inner_size=(0.8, 0.8, 0.8)),
            (0.55, 0.0, 0.0),
            (0.7, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        (Ellipsoid(radii=(0.6, 0.4, 0.3)), (0.0, 0.0, 0.0), (0.6, 0.0, 0.0), (0.9, 0.0, 0.0)),
        (Cone(radius=0.4, height=1.0, axis="z"), (0.0, 0.0, 0.5), (0.2, 0.0, 0.5), (0.3, 0.0, 0.5)),
        (Pyramid(base_size=1.0, height=1.0, axis="z"), (0.0, 0.0, 0.5), (0.25, 0.0, 0.5), (0.35, 0.0, 0.5)),
        (Prism(radius=0.4, height=1.0, num_sides=5, axis="z"), (0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (0.6, 0.0, 0.0)),
    ],
)
def test_primitive_occupancy_is_monotone_and_bounded(geometry, inside_point, boundary_point, outside_point):
    inside = _scalar_occupancy(geometry, inside_point, beta=0.05)
    boundary = _scalar_occupancy(geometry, boundary_point, beta=0.05)
    outside = _scalar_occupancy(geometry, outside_point, beta=0.05)

    assert 0.0 <= outside <= boundary <= inside <= 1.0
    assert boundary == pytest.approx(0.5, abs=5.0e-2)


def test_box_occupancy_has_gradients_for_position_size_and_rotation():
    xx, yy, zz = _asymmetric_grid()
    position = torch.nn.Parameter(torch.tensor([0.05, -0.08, 0.02], dtype=torch.float32))
    size = torch.nn.Parameter(torch.tensor([0.55, 0.45, 0.35], dtype=torch.float32))
    rotation = torch.nn.Parameter(torch.tensor([0.20, -0.15, 0.25], dtype=torch.float32))

    geometry = Box(position=position, size=size, rotation=rotation)
    loss = geometry.to_mask(xx, yy, zz, beta=0.06).sum()
    loss.backward()

    for grad in (position.grad, size.grad, rotation.grad):
        assert grad is not None
        assert torch.all(torch.isfinite(grad))
        assert torch.any(grad.abs() > 1.0e-6)


def test_sphere_occupancy_has_gradients_for_position_and_radius():
    xx, yy, zz = _asymmetric_grid()
    position = torch.nn.Parameter(torch.tensor([-0.05, 0.03, -0.04], dtype=torch.float32))
    radius = torch.nn.Parameter(torch.tensor(0.42, dtype=torch.float32))

    geometry = Sphere(position=position, radius=radius)
    loss = geometry.to_mask(xx, yy, zz, beta=0.06).sum()
    loss.backward()

    assert position.grad is not None
    assert radius.grad is not None
    assert torch.all(torch.isfinite(position.grad))
    assert torch.isfinite(radius.grad)
    assert torch.any(position.grad.abs() > 1.0e-6)
    assert abs(radius.grad.item()) > 1.0e-6


def test_cylinder_occupancy_has_gradients_for_position_radius_height_and_rotation():
    xx, yy, zz = _asymmetric_grid()
    position = torch.nn.Parameter(torch.tensor([0.04, -0.02, 0.01], dtype=torch.float32))
    radius = torch.nn.Parameter(torch.tensor(0.28, dtype=torch.float32))
    height = torch.nn.Parameter(torch.tensor(0.70, dtype=torch.float32))
    rotation = torch.nn.Parameter(torch.tensor([0.10, 0.25, -0.20], dtype=torch.float32))

    geometry = Cylinder(position=position, radius=radius, height=height, axis="z", rotation=rotation)
    loss = geometry.to_mask(xx, yy, zz, beta=0.06).sum()
    loss.backward()

    for grad in (position.grad, rotation.grad):
        assert grad is not None
        assert torch.all(torch.isfinite(grad))
        assert torch.any(grad.abs() > 1.0e-6)

    assert radius.grad is not None
    assert height.grad is not None
    assert torch.isfinite(radius.grad)
    assert torch.isfinite(height.grad)
    assert abs(radius.grad.item()) > 1.0e-6
    assert abs(height.grad.item()) > 1.0e-6


def test_ellipsoid_occupancy_has_gradients_for_position_radii_and_rotation():
    xx, yy, zz = _asymmetric_grid()
    position = torch.nn.Parameter(torch.tensor([0.02, -0.03, 0.01], dtype=torch.float32))
    radii = torch.nn.Parameter(torch.tensor([0.35, 0.25, 0.20], dtype=torch.float32))
    rotation = torch.nn.Parameter(torch.tensor([0.10, -0.20, 0.15], dtype=torch.float32))

    geometry = Ellipsoid(position=position, radii=radii, rotation=rotation)
    loss = geometry.to_mask(xx, yy, zz, beta=0.06).sum()
    loss.backward()

    for grad in (position.grad, radii.grad, rotation.grad):
        assert grad is not None
        assert torch.all(torch.isfinite(grad))
        assert torch.any(grad.abs() > 1.0e-6)


def test_cone_occupancy_has_gradients_for_position_radius_height_and_rotation():
    xx, yy, zz = _asymmetric_grid()
    position = torch.nn.Parameter(torch.tensor([0.02, -0.01, 0.00], dtype=torch.float32))
    radius = torch.nn.Parameter(torch.tensor(0.30, dtype=torch.float32))
    height = torch.nn.Parameter(torch.tensor(0.70, dtype=torch.float32))
    rotation = torch.nn.Parameter(torch.tensor([0.05, 0.18, -0.12], dtype=torch.float32))

    geometry = Cone(position=position, radius=radius, height=height, axis="z", rotation=rotation)
    loss = geometry.to_mask(xx, yy, zz, beta=0.06).sum()
    loss.backward()

    for grad in (position.grad, rotation.grad):
        assert grad is not None
        assert torch.all(torch.isfinite(grad))
        assert torch.any(grad.abs() > 1.0e-6)
    for grad in (radius.grad, height.grad):
        assert grad is not None
        assert torch.isfinite(grad)
        assert abs(grad.item()) > 1.0e-6


def test_pyramid_occupancy_has_gradients_for_position_size_height_and_rotation():
    xx, yy, zz = _asymmetric_grid()
    position = torch.nn.Parameter(torch.tensor([-0.03, 0.04, -0.02], dtype=torch.float32))
    base_size = torch.nn.Parameter(torch.tensor(0.60, dtype=torch.float32))
    height = torch.nn.Parameter(torch.tensor(0.75, dtype=torch.float32))
    rotation = torch.nn.Parameter(torch.tensor([0.08, -0.10, 0.16], dtype=torch.float32))

    geometry = Pyramid(position=position, base_size=base_size, height=height, axis="z", rotation=rotation)
    loss = geometry.to_mask(xx, yy, zz, beta=0.06).sum()
    loss.backward()

    for grad in (position.grad, rotation.grad):
        assert grad is not None
        assert torch.all(torch.isfinite(grad))
        assert torch.any(grad.abs() > 1.0e-6)
    for grad in (base_size.grad, height.grad):
        assert grad is not None
        assert torch.isfinite(grad)
        assert abs(grad.item()) > 1.0e-6


def test_prism_occupancy_has_gradients_for_position_radius_height_and_rotation():
    xx, yy, zz = _asymmetric_grid()
    position = torch.nn.Parameter(torch.tensor([0.01, 0.02, -0.03], dtype=torch.float32))
    radius = torch.nn.Parameter(torch.tensor(0.28, dtype=torch.float32))
    height = torch.nn.Parameter(torch.tensor(0.60, dtype=torch.float32))
    rotation = torch.nn.Parameter(torch.tensor([-0.12, 0.07, 0.14], dtype=torch.float32))

    geometry = Prism(position=position, radius=radius, height=height, num_sides=5, axis="z", rotation=rotation)
    loss = geometry.to_mask(xx, yy, zz, beta=0.06).sum()
    loss.backward()

    for grad in (position.grad, rotation.grad):
        assert grad is not None
        assert torch.all(torch.isfinite(grad))
        assert torch.any(grad.abs() > 1.0e-6)
    for grad in (radius.grad, height.grad):
        assert grad is not None
        assert torch.isfinite(grad)
        assert abs(grad.item()) > 1.0e-6


def test_mesh_signed_distance_matches_box_sign_for_closed_cube():
    vertices, faces = _cube_mesh(size=1.0)
    mesh = Mesh(vertices, faces, recenter=False, fill_mode="solid")
    box = Box(size=(1.0, 1.0, 1.0))

    sample_points = (
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.8, 0.0, 0.0),
    )
    for point in sample_points:
        coords = tuple(torch.tensor(coord, dtype=torch.float32) for coord in point)
        mesh_distance = mesh.signed_distance(*coords).item()
        box_distance = box.signed_distance(*coords).item()
        assert mesh_distance == pytest.approx(box_distance, abs=5.0e-3)


def test_mesh_occupancy_has_gradients_for_vertices():
    xx, yy, zz = _asymmetric_grid()
    vertices, faces = _cube_mesh(size=0.7, requires_grad=True)
    mesh = Mesh(vertices, faces, recenter=False, fill_mode="solid")
    loss = mesh.to_mask(xx, yy, zz, beta=0.06).sum()
    loss.backward()

    assert vertices.grad is not None
    assert torch.all(torch.isfinite(vertices.grad))
    assert torch.any(vertices.grad.abs() > 1.0e-6)


def test_mesh_watertight_detection_rejects_inconsistent_orientation():
    vertices, faces = _cube_mesh_with_flipped_face(size=1.0)
    mesh = Mesh(
        vertices.numpy(),
        faces.numpy(),
        recenter=False,
        fill_mode="auto",
        topology_diagnostics=True,
    )

    assert mesh.is_watertight is False
    assert mesh.boundary_edge_count == 0
    assert mesh.non_manifold_edge_count == 0
    assert mesh.inconsistent_edge_orientation_count > 0


def test_mesh_watertight_detection_rejects_degenerate_faces():
    vertices, faces = _cube_mesh_with_degenerate_face(size=1.0)
    mesh = Mesh(
        vertices.numpy(),
        faces.numpy(),
        recenter=False,
        fill_mode="auto",
        topology_diagnostics=True,
    )

    assert mesh.is_watertight is False
    assert mesh.degenerate_face_count > 0


def test_mesh_sdf_optimization_reduces_loss_for_trainable_translation():
    base_vertices, faces = _cube_mesh(size=0.8)
    sample_points = torch.tensor(
        [
            [-0.55, 0.00, 0.00],
            [-0.25, 0.10, -0.05],
            [-0.05, -0.15, 0.12],
            [0.15, 0.00, 0.08],
            [0.35, -0.10, 0.00],
            [0.55, 0.00, 0.00],
            [0.05, 0.30, 0.00],
            [0.00, 0.00, 0.45],
        ],
        dtype=torch.float32,
    )
    target_offset = torch.tensor([-0.18, 0.04, 0.02], dtype=torch.float32)
    initial_offset = torch.tensor([0.22, -0.08, -0.05], dtype=torch.float32)

    target_mesh = Mesh(base_vertices + target_offset.unsqueeze(0), faces, recenter=False, fill_mode="solid")
    target_sdf = target_mesh.signed_distance(
        sample_points[:, 0].view(-1, 1, 1),
        sample_points[:, 1].view(-1, 1, 1),
        sample_points[:, 2].view(-1, 1, 1),
    ).reshape(-1).detach()

    offset = torch.nn.Parameter(initial_offset.clone())
    optimizer = torch.optim.Adam([offset], lr=4.0e-2)

    initial_loss = None
    final_loss = None
    for _ in range(40):
        optimizer.zero_grad()
        mesh = Mesh(base_vertices + offset.unsqueeze(0), faces, recenter=False, fill_mode="solid")
        sdf = mesh.signed_distance(
            sample_points[:, 0].view(-1, 1, 1),
            sample_points[:, 1].view(-1, 1, 1),
            sample_points[:, 2].view(-1, 1, 1),
        ).reshape(-1)
        loss = torch.mean((sdf - target_sdf) ** 2)
        if initial_loss is None:
            initial_loss = float(loss.item())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    assert offset.grad is not None
    assert torch.all(torch.isfinite(offset.grad))
    assert final_loss is not None and initial_loss is not None
    assert final_loss < initial_loss
    assert torch.linalg.norm(offset.detach() - target_offset) < torch.linalg.norm(initial_offset - target_offset)
