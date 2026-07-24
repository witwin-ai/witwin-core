from __future__ import annotations

import torch
import pytest

from witwin.core import (
    AntennaPattern,
    AntennaState,
    Box,
    Material,
    Mesh,
    PhysicalMaterial,
    Scene,
    Sphere,
    Structure,
)


def test_material_and_structure_keep_continuous_tensor_identity():
    eps_r = torch.nn.Parameter(torch.tensor(3.0))
    sigma_e = torch.nn.Parameter(torch.tensor(0.02))
    position = torch.nn.Parameter(torch.tensor([0.0, 1.0, 2.0]))
    material = PhysicalMaterial(eps_r=eps_r, sigma_e=sigma_e)
    geometry = Box(position=position)
    structure = Structure(geometry=geometry, material=material, name="wall")
    antenna = AntennaState(0, "tx", position)

    scene = Scene(structures=[structure], endpoints=[antenna])

    assert Material is PhysicalMaterial
    assert material.eps_r is eps_r
    assert material.sigma_e is sigma_e
    assert scene.structures[0].geometry.position is position
    assert scene.structures[0].material is material
    assert scene.endpoints[0].position is position
    assert scene.structures[0].structure_id == structure.structure_id
    assert scene.structures[0].material_id == material.material_id
    assert scene.structures[0].assignment_id == structure.assignment_id


def test_antenna_keeps_custom_pattern_and_array_tensor_identity():
    element_positions = torch.randn(
        (4, 3), dtype=torch.float64, requires_grad=True
    )
    weights = torch.randn(
        4, dtype=torch.complex128, requires_grad=True
    )

    def custom(direction):
        return direction[..., 0]

    pattern = AntennaPattern("custom", custom)
    antenna = AntennaState(
        17,
        "tx",
        torch.zeros(3),
        element_positions=element_positions,
        weights=weights,
        pattern=pattern,
    )

    assert antenna.pattern is pattern
    assert antenna.pattern.custom is custom
    assert antenna.element_positions is element_positions
    assert antenna.weights is weights


def test_antenna_non_tensor_continuous_values_use_float32():
    antenna = AntennaState(
        18,
        "rx",
        (0, 1, 2),
        orientation=(0, 0, 0),
        polarization=(1, 0, 0),
        element_positions=((0, 0, 0),),
        weights=(1,),
    )

    assert antenna.position.dtype == torch.float32
    assert antenna.orientation.dtype == torch.float32
    assert antenna.polarization.dtype == torch.float32
    assert antenna.element_positions.dtype == torch.float32
    assert antenna.weights.dtype == torch.float32


def test_scene_assigns_stable_ids_and_reuses_shared_material_identity():
    material = PhysicalMaterial(eps_r=2.5)
    scene = Scene(
        structures=[
            Structure(Box(), material, name="a"),
            Structure(Box(position=(1.0, 0.0, 0.0)), material, name="b"),
        ]
    )

    first, second = scene.structures
    reordered = Scene(structures=[second, first])

    assert first.structure_id != second.structure_id
    assert first.material_id == second.material_id == material.material_id
    assert first.assignment_id != second.assignment_id
    assert tuple(item.structure_id for item in reordered.structures) == (
        second.structure_id,
        first.structure_id,
    )
    assert tuple(item.assignment_id for item in reordered.structures) == (
        second.assignment_id,
        first.assignment_id,
    )


def test_scene_granular_versions_advance_only_for_the_changed_domain():
    material = PhysicalMaterial(eps_r=2.0)
    scene = Scene(structures=[Structure(Box(), material)])
    structure_id = scene.structures[0].structure_id

    geometry_scene = scene.with_structure_geometry(
        structure_id, Box(position=(1.0, 0.0, 0.0))
    )
    topology_scene = scene.with_structure_geometry(
        structure_id, Box(), topology_changed=True
    )
    material_scene = scene.with_material(
        material.material_id,
        PhysicalMaterial(eps_r=4.0, material_id=material.material_id, version=1),
    )
    assignment_scene = scene.with_structure_material(
        scene.structures[0].structure_id, PhysicalMaterial(eps_r=5.0)
    )

    assert geometry_scene.geometry_version != scene.geometry_version
    assert geometry_scene.topology_version == scene.topology_version
    assert topology_scene.topology_version != scene.topology_version
    assert topology_scene.geometry_version != scene.geometry_version
    assert material_scene.material_version != scene.material_version
    assert material_scene.assignment_version == scene.assignment_version
    assert assignment_scene.assignment_version != scene.assignment_version
    assert assignment_scene.material_version != scene.material_version


def test_geometry_kind_change_invalidates_topology_by_default():
    scene = Scene(structures=[Structure(Box(), PhysicalMaterial())])
    structure_id = scene.structures[0].structure_id

    updated = scene.with_structure_geometry(structure_id, Sphere())

    assert updated.topology_version != scene.topology_version
    assert updated.geometry_version != scene.geometry_version
    with pytest.raises(ValueError, match="detectable topology change"):
        scene.with_structure_geometry(
            structure_id,
            Sphere(),
            topology_changed=False,
        )


def test_scene_rejects_invalid_or_duplicate_primitive_ids():
    material = PhysicalMaterial()
    with pytest.raises(ValueError, match="non-negative"):
        Scene(
            structures=[
                Structure(Box(), material, primitive_ids=(-1,))
            ]
        )

    with pytest.raises(ValueError, match="unique within a Scene"):
        Scene(
            structures=[
                Structure(Box(), material, primitive_ids=(100,)),
                Structure(Box(), material, primitive_ids=(100,)),
            ]
        )


def test_explicit_ids_are_reserved_before_automatic_allocation():
    import witwin.core.identity as identity

    previous_next = identity._NEXT_MATERIAL_ID
    try:
        identity._NEXT_MATERIAL_ID = 0
        explicit_material = PhysicalMaterial(material_id=0)
        automatic_material = PhysicalMaterial()
        assert automatic_material.material_id != explicit_material.material_id
    finally:
        identity._NEXT_MATERIAL_ID = previous_next


def test_static_snapshot_references_the_same_logical_leaves():
    vertices = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
    material_value = torch.nn.Parameter(torch.tensor(2.0))
    material = PhysicalMaterial(eps_r=material_value)
    structure = Structure(Box(position=vertices), material)
    scene = Scene(structures=[structure])

    first = scene.snapshot(time_s=1.5)
    second = scene.snapshot(time_s=1.5)

    assert first.time_s == second.time_s == 1.5
    assert first.versions == second.versions == scene.versions
    assert first.structures[0].structure.geometry.position is vertices
    assert first.structures[0].structure.material.eps_r is material_value


def test_in_place_tensor_mutation_changes_only_owning_version_tokens():
    position = torch.tensor([0.0, 0.0, 0.0])
    eps_r = torch.tensor(2.0)
    material = PhysicalMaterial(eps_r=eps_r)
    scene = Scene(structures=[Structure(Box(position=position), material)])
    initial = scene.versions

    position.add_(1.0)
    after_geometry = scene.versions
    eps_r.add_(1.0)
    after_material = scene.versions

    assert after_geometry.geometry != initial.geometry
    assert after_geometry.material == initial.material
    assert after_material.material != after_geometry.material
    assert after_material.geometry == after_geometry.geometry


def test_snapshot_detects_in_place_mutation_of_referenced_tensor_leaves():
    position = torch.tensor([0.0, 0.0, 0.0])
    material_value = torch.tensor(2.0)
    scene = Scene(
        structures=[
            Structure(
                Box(position=position),
                PhysicalMaterial(eps_r=material_value),
            )
        ]
    )
    snapshot = scene.snapshot()
    geometry_version = snapshot.geometry_version
    material_version = snapshot.material_version

    position.add_(1.0)
    assert snapshot.geometry_version != geometry_version
    assert snapshot.material_version == material_version

    material_value.add_(1.0)
    assert snapshot.material_version != material_version


def test_geometry_constructor_keeps_float64_tensor_leaves():
    position = torch.nn.Parameter(
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    )
    size = torch.nn.Parameter(
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    )
    rotation = torch.nn.Parameter(
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    )

    box = Box(position=position, size=size, rotation=rotation)

    assert box.position is position
    assert box.size is size
    assert box.rotation is rotation


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)
def test_box_resolves_non_tensor_parameters_to_cuda_tensor_device():
    position = torch.zeros(3, device="cuda", requires_grad=True)

    box = Box(position=position, size=(1.0, 2.0, 3.0))

    assert box.position is position
    assert box.size.device == position.device
    assert box.rotation.device == position.device
    with pytest.raises(ValueError, match="conflicts"):
        Box(position=position, size=torch.ones(3))


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_mesh_keeps_authored_vertex_tensor_identity(device):
    base = torch.randn(
        (6, 4), device=device, dtype=torch.float64, requires_grad=True
    )
    vertices = base[1:5, :3]
    faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3]], device=device, dtype=torch.int32
    )

    mesh = Mesh(vertices, faces, recenter=False, fill_mode="surface")

    assert mesh.vertices is vertices
    assert mesh.vertices.dtype == torch.float64
    assert mesh.vertices.device == base.device
    assert mesh.vertices.stride() == vertices.stride()
    assert mesh.vertices.storage_offset() == vertices.storage_offset()
    assert mesh.vertices.requires_grad
    assert mesh.faces is faces


def test_tensor_authored_mesh_auto_allows_logical_use_but_not_sdf():
    vertices = torch.randn((4, 3))
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])

    mesh = Mesh(
        vertices,
        faces,
        fill_mode="auto",
        topology_diagnostics=False,
    )

    assert mesh.vertices is vertices
    assert mesh.faces is faces
    assert mesh.is_watertight is None
    mesh.to_mesh()
    point = torch.tensor(0.0)
    with pytest.raises(RuntimeError, match="topology is unknown"):
        mesh.signed_distance(point, point, point)


def test_cpu_tensor_topology_diagnostics_fail_when_stale():
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = torch.tensor([[0, 1, 2]])
    mesh = Mesh(
        vertices,
        faces,
        fill_mode="auto",
        topology_diagnostics=True,
    )
    with torch.no_grad():
        vertices.add_(0.25)
    point = torch.tensor(0.0)

    with pytest.raises(RuntimeError, match="diagnostics are stale"):
        mesh.signed_distance(point, point, point)


def test_mesh_explicit_device_conflict_fails_loudly():
    vertices = torch.randn((4, 3))
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])

    with pytest.raises(ValueError, match="conflicts"):
        Mesh(vertices, faces, fill_mode="surface", device="meta")


def test_mesh_faces_require_int32_or_int64():
    vertices = torch.randn((4, 3))
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int16)

    with pytest.raises(TypeError, match="int32 or torch.int64"):
        Mesh(vertices, faces, fill_mode="surface")

    with pytest.raises(TypeError, match="int32 or torch.int64"):
        Mesh(
            vertices.tolist(),
            [[0.0, 1.0, 2.0]],
            fill_mode="surface",
        )


def test_mesh_to_mesh_preserves_authored_dtype_device_and_grad():
    vertices = torch.randn(
        (4, 3), dtype=torch.float64, requires_grad=True
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
    mesh = Mesh(vertices, faces, recenter=False, fill_mode="surface")

    world_vertices, world_faces = mesh.to_mesh()

    assert world_vertices.dtype == vertices.dtype
    assert world_vertices.device == vertices.device
    assert world_vertices.requires_grad
    assert world_faces is faces
    world_vertices.sum().backward()
    assert vertices.grad is not None


def test_mesh_keeps_surface_thickness_tensor_identity():
    vertices = torch.randn((4, 3), dtype=torch.float64)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    thickness = torch.nn.Parameter(
        torch.tensor(0.01, dtype=torch.float64)
    )

    mesh = Mesh(
        vertices,
        faces,
        fill_mode="surface",
        surface_thickness=thickness,
    )

    assert mesh.surface_thickness is thickness
    point = torch.tensor(0.0, dtype=torch.float64)
    mesh.signed_distance(point, point, point).backward()
    assert thickness.grad is not None
