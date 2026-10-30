from __future__ import annotations

import importlib.util

import witwin.core as core
from witwin.core import math as core_math


STABLE_PUBLIC_API = {
    "AntennaId",
    "AntennaPattern",
    "AntennaState",
    "AssignmentId",
    "BooleanGeometry",
    "Box",
    "ComplexLike",
    "Cone",
    "Cylinder",
    "Deformation",
    "DeformationState",
    "Difference",
    "DispersionSpec",
    "DynamicScene",
    "Ellipsoid",
    "EndpointState",
    "FrequencyMaterialSample",
    "Geometry",
    "GeometryBase",
    "GeometrySpec",
    "geometry_from_dict",
    "geometry_to_dict",
    "HollowBox",
    "Intersection",
    "LinearTrajectory",
    "MaterialAssignment",
    "MaterialCapabilities",
    "MaterialId",
    "MaterialLayer",
    "MaterialSpec",
    "Mesh",
    "PhaseScreen",
    "PhysicalMaterial",
    "PowerLawDispersion",
    "PrimitiveId",
    "Prism",
    "Pyramid",
    "ReceiverGrid",
    "RigidMotion",
    "ScalarLike",
    "Scene",
    "SceneSnapshot",
    "SceneVersions",
    "Sphere",
    "StaticMaterialSample",
    "Structure",
    "StructureId",
    "StructureState",
    "SurfaceId",
    "SurfaceRoughness",
    "Torus",
    "Trajectory",
    "Union",
    "antenna_orientation_matrix",
    "new_antenna_id",
    "new_assignment_id",
    "new_material_id",
    "new_structure_id",
    "new_surface_id",
    "normalize_vec3",
    "quat_from_euler",
    "quat_identity",
    "quat_multiply",
    "quat_to_rotation_matrix",
    "quat_to_rotation_matrix_np",
    "reserve_antenna_id",
    "reserve_assignment_id",
    "reserve_material_id",
    "reserve_structure_id",
    "reserve_surface_id",
}


def test_top_level_public_api_is_stable():
    assert len(core.__all__) == len(set(core.__all__))
    assert set(core.__all__) == STABLE_PUBLIC_API
    assert all(hasattr(core, name) for name in STABLE_PUBLIC_API)


def test_scalar_types_are_owned_by_math_module():
    assert core.ScalarLike is core_math.ScalarLike
    assert core.ComplexLike is core_math.ComplexLike
    assert importlib.util.find_spec("witwin.core.scalars") is None
