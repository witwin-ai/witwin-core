from __future__ import annotations

import inspect
import math

import pytest
import torch

from witwin.core import Box, Difference, Intersection, Mesh, Sphere, Structure, Union
from witwin.core.material import PhysicalMaterial


def _points(*points, dtype=torch.float32):
    coords = torch.tensor(points, dtype=dtype)
    return coords[:, 0], coords[:, 1], coords[:, 2]


def _grid(half_extent, count, dtype=torch.float32):
    axis = torch.linspace(-half_extent, half_extent, count, dtype=dtype)
    xx, yy, zz = torch.meshgrid(axis, axis, axis, indexing="ij")
    return xx, yy, zz, float(axis[1] - axis[0]) ** 3


def _lens_volume(radius, separation):
    return math.pi * (4 * radius + separation) * (2 * radius - separation) ** 2 / 12


def _closed_mesh(vertices, faces):
    return Mesh(vertices.detach().numpy(), faces.numpy(), recenter=False)


def test_union_signed_distance_matches_analytic_values():
    union = Union(Box(size=(2, 2, 2)), Sphere(position=(2, 0, 0), radius=0.5))
    sd = union.signed_distance(*_points((0, 0, 0), (1.5, 0, 0), (2, 0, 0), (3, 0, 0), (0, 0, 2)))
    torch.testing.assert_close(sd, torch.tensor([-1.0, 0.0, -0.5, 0.5, 1.0]))


def test_intersection_of_spheres_is_exact_lens_distance():
    lens = Intersection(Sphere(position=(-0.5, 0, 0), radius=1.0), Sphere(position=(0.5, 0, 0), radius=1.0))
    sd = lens.signed_distance(*_points((0, 0, 0), (0.5, 0, 0), (1, 0, 0), (-1.5, 0, 0)))
    torch.testing.assert_close(sd, torch.tensor([-0.5, 0.0, 0.5, 1.0]))


def test_difference_carves_hole_with_exact_distances():
    shell = Difference(Box(size=(2, 2, 2)), Sphere(radius=0.5))
    sd = shell.signed_distance(*_points((0, 0, 0), (0.5, 0, 0), (0.75, 0, 0), (1, 0, 0), (2, 0, 0)))
    torch.testing.assert_close(sd, torch.tensor([0.5, 0.0, -0.25, 0.0, 1.0]))


def test_difference_subtracts_union_of_several_tools():
    base = Box(size=(4, 1, 1))
    tools = (Sphere(position=(-1, 0, 0), radius=0.3), Sphere(position=(1, 0, 0), radius=0.3))
    carved = Difference(base, *tools)
    assert carved.base is base and carved.tools == tools
    xx, yy, zz, _ = _grid(2.5, 21)
    expected = torch.maximum(
        base.signed_distance(xx, yy, zz),
        -torch.minimum(tools[0].signed_distance(xx, yy, zz), tools[1].signed_distance(xx, yy, zz)),
    )
    torch.testing.assert_close(carved.signed_distance(xx, yy, zz), expected)


def test_node_frame_matches_primitive_rotation_and_translation():
    rotation = (0.3, -0.2, 0.7)
    node = Union(Box(size=(2, 1, 0.5)), position=(1, -0.5, 0.25), rotation=rotation)
    reference = Box(position=(1, -0.5, 0.25), size=(2, 1, 0.5), rotation=rotation)
    xx, yy, zz, _ = _grid(2.0, 15)
    torch.testing.assert_close(node.signed_distance(xx, yy, zz), reference.signed_distance(xx, yy, zz))
    reference_vertices, _ = reference.to_mesh()
    torch.testing.assert_close(
        node.bounds_world,
        torch.stack((reference_vertices.amin(dim=0), reference_vertices.amax(dim=0))),
    )


def test_occupancy_volume_fractions_match_analytic_volumes():
    xx, yy, zz, cell = _grid(1.6, 97)
    a = Sphere(position=(-0.5, 0, 0), radius=1.0)
    b = Sphere(position=(0.5, 0, 0), radius=1.0)
    sphere_volume = 4.0 / 3.0 * math.pi
    lens = _lens_volume(1.0, 1.0)
    cases = (
        (Union(a, b), 2 * sphere_volume - lens),
        (Intersection(a, b), lens),
        (Difference(a, b), sphere_volume - lens),
        (Difference(Box(size=(2, 2, 2)), Sphere(radius=0.6)), 8.0 - 4.0 / 3.0 * math.pi * 0.6**3),
    )
    for geometry, expected in cases:
        volume = float(geometry.to_mask(xx, yy, zz).sum()) * cell
        assert volume == pytest.approx(expected, rel=1.0e-2), type(geometry).__name__


def test_gradient_flows_to_child_parameters_through_occupancy():
    xx, yy, zz, cell = _grid(1.2, 81)
    radius = torch.tensor(0.6, requires_grad=True)
    shift = torch.zeros(3, requires_grad=True)
    shell = Difference(Box(position=shift, size=(2, 2, 2)), Sphere(radius=radius))
    volume = shell.to_mask(xx, yy, zz, beta=0.02).sum() * cell
    volume.backward()
    # d(volume)/d(radius) is minus the hole surface area.
    assert float(radius.grad) == pytest.approx(-4.0 * math.pi * 0.6**2, rel=3.0e-2)
    assert shift.grad is not None and torch.isfinite(shift.grad).all()


def test_smooth_composition_gradcheck_and_sharp_limit():
    point = _points((0.45, 0.1, -0.05), (0.2, 0.6, 0.3), dtype=torch.float64)
    radius = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    center = torch.tensor([0.4, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    smoothness = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)

    def objective(radius, center, smoothness):
        box = Box(size=torch.ones(3, dtype=torch.float64))
        sphere = Sphere(position=center, radius=radius)
        return torch.stack(
            [
                Union(box, sphere, smoothness=smoothness).signed_distance(*point),
                Intersection(box, sphere, smoothness=smoothness).signed_distance(*point),
                Difference(box, sphere, smoothness=smoothness).signed_distance(*point),
            ]
        )

    assert torch.autograd.gradcheck(objective, (radius, center, smoothness))


def test_smooth_union_is_exact_away_from_seam_and_bounded_near_it():
    a = Sphere(position=(-0.6, 0, 0), radius=0.5)
    b = Sphere(position=(0.6, 0, 0), radius=0.5)
    k = 0.2
    sharp = Union(a, b)
    smooth = Union(a, b, smoothness=k)
    xx, yy, zz, _ = _grid(1.5, 31)
    da, db = a.signed_distance(xx, yy, zz), b.signed_distance(xx, yy, zz)
    sharp_sd, smooth_sd = sharp.signed_distance(xx, yy, zz), smooth.signed_distance(xx, yy, zz)
    far = (da - db).abs() >= k
    torch.testing.assert_close(smooth_sd[far], sharp_sd[far])
    assert bool(torch.all(smooth_sd <= sharp_sd + 1.0e-6))
    assert bool(torch.all(smooth_sd >= sharp_sd - k / 4 - 1.0e-6))
    # The blend bridges the gap at the seam: the origin is inside only when smooth.
    origin = _points((0, 0, 0))
    assert float(sharp.signed_distance(*origin)) > 0.0
    smooth_bridge = Union(a, b, smoothness=0.6)
    assert float(smooth_bridge.signed_distance(*origin)) < 0.0


def test_bounds_follow_composition_rules():
    a = Box(size=(2, 2, 2))
    b = Sphere(position=(1.5, 0, 0), radius=1.0)
    torch.testing.assert_close(Union(a, b).bounds_world, torch.tensor([[-1.0, -1.0, -1.0], [2.5, 1.0, 1.0]]))
    torch.testing.assert_close(Intersection(a, b).bounds_world, torch.tensor([[0.5, -1.0, -1.0], [1.0, 1.0, 1.0]]))
    torch.testing.assert_close(Difference(a, b).bounds_world, torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
    grown = Union(a, b, smoothness=0.4).bounds_world
    torch.testing.assert_close(grown, torch.tensor([[-1.1, -1.1, -1.1], [2.6, 1.1, 1.1]]))

    radius = torch.tensor(1.0, requires_grad=True)
    Union(a, Sphere(position=(1.5, 0, 0), radius=radius)).bounds_world[1, 0].backward()
    assert float(radius.grad) == pytest.approx(1.0)


def test_to_mesh_is_closed_oriented_and_matches_volume():
    a = Sphere(position=(-0.5, 0, 0), radius=1.0)
    b = Sphere(position=(0.5, 0, 0), radius=1.0)
    sphere_volume = 4.0 / 3.0 * math.pi
    lens = _lens_volume(1.0, 1.0)
    for geometry, expected in (
        (Union(a, b), 2 * sphere_volume - lens),
        (Intersection(a, b), lens),
        (Difference(Box(size=(2, 2, 2)), Sphere(radius=0.6), rotation=(0.2, 0.4, 0.1)), 8.0 - sphere_volume * 0.216),
    ):
        vertices, faces = geometry.to_mesh(resolution=40)
        assert faces.dtype == torch.int64
        mesh = _closed_mesh(vertices, faces)
        assert mesh.is_watertight, type(geometry).__name__
        assert mesh.signed_volume > 0.0
        assert mesh.signed_volume == pytest.approx(expected, rel=2.0e-2)


def test_to_mesh_vertices_are_differentiable_in_child_parameters():
    radius = torch.tensor(0.8, requires_grad=True)
    union = Union(Box(size=(1, 1, 1)), Sphere(position=(0.6, 0, 0), radius=radius))
    vertices, _ = union.to_mesh(resolution=24)
    vertices[:, 0].max().backward()
    assert float(radius.grad) == pytest.approx(1.0, abs=0.1)


def test_mesh_children_and_nested_booleans():
    cube = Box(size=(1, 1, 1))
    vertices, faces = cube.to_mesh()
    mesh = Mesh(vertices.numpy(), faces.numpy(), recenter=False)
    nested = Difference(Union(mesh, Sphere(position=(0.5, 0, 0), radius=0.4)), Box(position=(0, 0, 0.5), size=(3, 3, 0.4)))
    xx, yy, zz, _ = _grid(1.0, 9)
    expected = torch.maximum(
        torch.minimum(cube.signed_distance(xx, yy, zz), Sphere(position=(0.5, 0, 0), radius=0.4).signed_distance(xx, yy, zz)),
        -Box(position=(0, 0, 0.5), size=(3, 3, 0.4)).signed_distance(xx, yy, zz),
    )
    torch.testing.assert_close(nested.signed_distance(xx, yy, zz), expected, atol=1.0e-4, rtol=0)
    assert _closed_mesh(*nested.to_mesh(resolution=24)).is_watertight


def test_child_tensors_are_exposed_on_the_node():
    radius = torch.tensor(0.4, requires_grad=True)
    inner = Union(Box(size=(1, 1, 1)), Sphere(radius=radius))
    outer = Difference(inner, Box(position=(0, 0, 1), size=(2, 2, 1)))
    tensors = [value for value in vars(outer).values() if isinstance(value, torch.Tensor)]
    assert any(tensor is radius for tensor in tensors)
    assert outer.has_trainable_geometry()
    assert not Union(Box(), Sphere()).has_trainable_geometry()


def test_structure_accepts_boolean_geometry():
    structure = Structure(Difference(Box(), Sphere(radius=0.2)), PhysicalMaterial(eps_r=2.0))
    assert structure.geometry.kind == "difference"


def test_construction_validation_and_device_default():
    for cls in (Union, Intersection, Difference):
        assert inspect.signature(cls).parameters["device"].default is None
    with pytest.raises(ValueError):
        Difference(Box())
    with pytest.raises(ValueError):
        Union()
    with pytest.raises(ValueError):
        Union(Box(), Sphere(), smoothness=-0.1)
    with pytest.raises(TypeError):
        Union(Box(), object())
    assert Union([Box(), Sphere()]).geometries[1].kind == "sphere"
    with pytest.raises(ValueError):
        Intersection(Box(), Box(position=(5, 0, 0))).to_mesh()
    assert not torch.cuda.is_initialized()
