from __future__ import annotations

import torch
import pytest

from witwin.core import (
    AntennaState,
    Box,
    DynamicScene,
    LinearTrajectory,
    PhysicalMaterial,
    Scene,
    Structure,
)


def test_dynamic_scene_snapshot_is_deterministic_and_granular():
    origin = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
    velocity = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))
    structure = Structure(Box(), PhysicalMaterial(), structure_id=4)
    antenna = AntennaState(2, "rx", origin)
    scene = Scene(structures=[structure], endpoints=[antenna])
    trajectory = LinearTrajectory(origin=origin, velocity=velocity)
    dynamic = DynamicScene(
        scene,
        structure_trajectories={4: trajectory},
        endpoint_trajectories={2: trajectory},
        dynamic_revision=3,
    )

    first = dynamic.at(0.25)
    second = dynamic.at(0.25)
    later = dynamic.at(0.5)

    assert first.versions == second.versions
    assert first.geometry_version != later.geometry_version
    assert first.topology_version == later.topology_version
    assert first.material_version == later.material_version
    assert first.assignment_version == later.assignment_version
    assert torch.equal(
        first.structures[0].rigid_motion.translation,
        second.structures[0].rigid_motion.translation,
    )
    assert first.endpoints[0].antenna.position is origin
    assert first.structures[0].rigid_motion.velocity is velocity


def test_dynamic_snapshot_preserves_gradient_flow_to_trajectory_leaves():
    origin = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
    velocity = torch.nn.Parameter(torch.tensor([2.0, 0.0, 0.0]))
    scene = Scene(
        structures=[
            Structure(Box(), PhysicalMaterial(), structure_id=0)
        ]
    )
    dynamic = DynamicScene(
        scene,
        structure_trajectories={
            0: LinearTrajectory(origin=origin, velocity=velocity)
        },
    )

    translation = dynamic.at(0.5).structures[0].rigid_motion.translation
    translation.sum().backward()

    assert origin.grad is not None
    assert velocity.grad is not None
    assert torch.equal(velocity.grad, torch.full_like(velocity, 0.5))


def test_dynamic_source_in_place_mutation_changes_geometry_version():
    origin = torch.nn.Parameter(torch.zeros(3))
    velocity = torch.nn.Parameter(torch.ones(3))
    scene = Scene(
        structures=[Structure(Box(), PhysicalMaterial(), structure_id=8)]
    )
    dynamic = DynamicScene(
        scene,
        structure_trajectories={
            8: LinearTrajectory(origin=origin, velocity=velocity)
        },
    )
    first = dynamic.at(0.5)
    same = dynamic.at(0.5)
    initial_version = first.geometry_version

    assert same.geometry_version == initial_version
    with torch.no_grad():
        origin.add_(1.0)
    assert first.geometry_version != initial_version
    after_mutation = dynamic.at(0.5)
    repeated = dynamic.at(0.5)
    assert after_mutation.geometry_version == repeated.geometry_version
    assert after_mutation.geometry_version != initial_version


def test_tensor_time_identity_and_mutation_participate_in_snapshot_state():
    origin = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
    velocity = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))
    scene = Scene(
        structures=[Structure(Box(), PhysicalMaterial(), structure_id=20)]
    )
    dynamic = DynamicScene(
        scene,
        structure_trajectories={
            20: LinearTrajectory(origin=origin, velocity=velocity)
        },
    )
    time_s = torch.nn.Parameter(torch.tensor(0.25))

    first = dynamic.at(time_s)
    first_version = first.geometry_version
    with torch.no_grad():
        time_s.add_(0.25)
    assert first.geometry_version != first_version
    second = dynamic.at(time_s)

    assert first.time_s is time_s
    assert second.time_s is time_s
    assert first.geometry_version != second.geometry_version
    second.structures[0].rigid_motion.translation.sum().backward()
    assert time_s.grad is not None


def test_distinct_tensor_time_inputs_have_distinct_geometry_versions():
    scene = Scene(
        structures=[Structure(Box(), PhysicalMaterial(), structure_id=21)]
    )
    dynamic = DynamicScene(
        scene,
        structure_trajectories={
            21: LinearTrajectory(
                origin=torch.zeros(3),
                velocity=torch.ones(3),
            )
        },
    )

    first = dynamic.at(torch.tensor(0.25))
    second = dynamic.at(torch.tensor(0.5))

    assert first.geometry_version != second.geometry_version


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)
def test_linear_trajectory_rejects_cross_device_tensor_time():
    trajectory = LinearTrajectory(
        origin=torch.zeros(3, device="cuda"),
        velocity=torch.ones(3, device="cuda"),
    )

    with pytest.raises(ValueError, match="trajectory tensor device"):
        trajectory.at(torch.tensor(0.5))


def test_non_tensor_motion_values_use_float32():
    trajectory = LinearTrajectory(origin=(0, 0, 0), velocity=(1, 0, 0))
    snapshot = trajectory.at(0.5)

    assert trajectory.origin.dtype == torch.float32
    assert trajectory.velocity.dtype == torch.float32
    assert snapshot.translation.dtype == torch.float32
