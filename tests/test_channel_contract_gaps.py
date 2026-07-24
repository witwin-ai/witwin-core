from __future__ import annotations

import math

import pytest
import torch

from witwin.core import (
    AntennaState,
    Mesh,
    PhysicalMaterial,
    ReceiverGrid,
    Scene,
    Structure,
)


def _triangle_mesh() -> tuple[torch.Tensor, torch.Tensor]:
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    return vertices, faces


def test_receiver_grid_is_typed_rx_and_preserves_tensor_leaves():
    origin = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    x_axis = torch.nn.Parameter(
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    )
    y_axis = torch.nn.Parameter(
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    )
    spacing = torch.nn.Parameter(
        torch.tensor([0.5, 0.25], dtype=torch.float64)
    )
    grid = ReceiverGrid(
        41,
        origin,
        x_axis,
        y_axis,
        (2, 3),
        spacing,
    )

    assert isinstance(grid, AntennaState)
    assert grid.role == "rx"
    assert grid.antenna_id == 41
    assert grid.origin is origin
    assert grid.position is origin
    assert grid.x_axis is x_axis
    assert grid.y_axis is y_axis
    assert grid.spacing is spacing
    assert grid.shape == (2, 3)

    points = grid.points()
    assert points.shape == (6, 3)
    assert torch.equal(
        points[-1],
        torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64),
    )
    points.sum().backward()
    assert origin.grad is not None
    assert x_axis.grad is not None
    assert y_axis.grad is not None
    assert spacing.grad is not None


def test_antenna_orientation_and_local_polarization_are_differentiable():
    orientation = torch.nn.Parameter(
        torch.tensor([math.pi / 2.0, 0.0, 0.0], dtype=torch.float64)
    )
    polarization = torch.nn.Parameter(
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    )
    antenna = AntennaState(
        42,
        "tx",
        torch.zeros(3, dtype=torch.float64),
        orientation=orientation,
        polarization=polarization,
    )

    world = antenna.polarization_world()
    assert torch.allclose(
        world,
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        atol=1.0e-12,
    )
    world.sum().backward()
    assert orientation.grad is not None
    assert polarization.grad is not None


def test_scalar_first_quaternion_orientation_preserves_autograd():
    orientation = torch.nn.Parameter(
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    )
    antenna = AntennaState(
        43,
        "rx",
        torch.zeros(3, dtype=torch.float64),
        orientation=orientation,
    )

    rotation = antenna.orientation_matrix()
    assert torch.equal(rotation, torch.eye(3, dtype=torch.float64))
    rotation.sum().backward()
    assert orientation.grad is not None


def test_receiver_grid_participates_in_scene_endpoint_identity():
    first = ReceiverGrid(
        50,
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (2, 2),
        (1.0, 1.0),
    )
    Scene(endpoints=[first])

    duplicate = ReceiverGrid(
        50,
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (2, 2),
        (1.0, 1.0),
    )
    with pytest.raises(ValueError, match="Antenna IDs must be unique"):
        Scene(endpoints=[first, duplicate])


def test_structure_uv_topology_preserves_tensor_identity_and_versions():
    vertices, faces = _triangle_mesh()
    uv = torch.nn.Parameter(
        torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float64,
        )
    )
    face_uv = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    structure = Structure(
        Mesh(vertices, faces, recenter=False, fill_mode="surface"),
        PhysicalMaterial(),
        uv=uv,
        face_uv=face_uv,
    )
    scene = Scene(structures=[structure])

    assert structure.uv is uv
    assert structure.face_uv is face_uv
    geometry_version = scene.geometry_version
    topology_version = scene.topology_version

    with torch.no_grad():
        uv.add_(0.25)
    assert scene.geometry_version != geometry_version
    assert scene.topology_version == topology_version

    geometry_version = scene.geometry_version
    face_uv[0, 0] = 0
    assert scene.topology_version != topology_version
    assert scene.geometry_version == geometry_version


def test_structure_uv_contract_rejects_incomplete_or_mismatched_topology():
    vertices, faces = _triangle_mesh()
    mesh = Mesh(vertices, faces, recenter=False, fill_mode="surface")
    material = PhysicalMaterial()
    uv = torch.zeros((3, 2))

    with pytest.raises(ValueError, match="provided together"):
        Structure(mesh, material, uv=uv)
    with pytest.raises(ValueError, match="one row per mesh face"):
        Structure(
            mesh,
            material,
            uv=uv,
            face_uv=torch.zeros((2, 3), dtype=torch.int32),
        )
    with pytest.raises(TypeError, match="int32 or torch.int64"):
        Structure(
            mesh,
            material,
            uv=uv,
            face_uv=torch.zeros((1, 3), dtype=torch.int16),
        )


def test_perfect_conductor_has_explicit_identity_and_finite_parameters():
    pec = PhysicalMaterial.perfect_conductor(material_id=70)
    sample = pec.evaluate_static()

    assert pec.material_id == 70
    assert pec.conductor_model == "perfect"
    assert pec.capabilities().conductive
    assert pec.capabilities().perfect_conductor
    assert math.isfinite(float(sample.eps_r))
    assert math.isfinite(float(sample.mu_r))
    assert math.isfinite(float(sample.sigma_e))
    assert not PhysicalMaterial().capabilities().perfect_conductor


def test_surface_identity_is_stable_in_structure_and_assignment():
    material = PhysicalMaterial()
    first = Structure(
        Mesh(*_triangle_mesh(), recenter=False, fill_mode="surface"),
        material,
        surface_id=81,
    )
    assert first.surface_id == 81
    assert Scene(structures=[first]).assignments[0].surface_id == 81

    duplicate = Structure(
        Mesh(*_triangle_mesh(), recenter=False, fill_mode="surface"),
        material,
        surface_id=81,
    )
    with pytest.raises(ValueError, match="Surface IDs must be unique"):
        Scene(structures=[first, duplicate])

    with pytest.raises(ValueError, match="surface_id must be non-negative"):
        Structure(
            Mesh(*_triangle_mesh(), recenter=False, fill_mode="surface"),
            material,
            surface_id=-1,
        )
