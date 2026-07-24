"""Witwin Core - Shared geometry, material, and structure contracts."""

__version__ = "0.4.0"

from .material import (
    DispersionSpec,
    FrequencyMaterialSample,
    MaterialCapabilities,
    MaterialLayer,
    MaterialSpec,
    PhaseScreen,
    PhysicalMaterial,
    PowerLawDispersion,
    StaticMaterialSample,
    SurfaceRoughness,
)
from .scalars import (
    ComplexLike,
    ScalarLike,
)
from .structure import (
    MaterialAssignment,
    Structure,
)
from .identity import (
    AntennaId,
    AssignmentId,
    MaterialId,
    PrimitiveId,
    StructureId,
    SurfaceId,
)
from .antenna import (
    AntennaPattern,
    AntennaState,
    ReceiverGrid,
    antenna_orientation_matrix,
)
from .scene import (
    EndpointState,
    Scene,
    SceneSnapshot,
    SceneVersions,
    StructureState,
)
from .dynamics import (
    Deformation,
    DeformationState,
    DynamicScene,
    LinearTrajectory,
    RigidMotion,
    Trajectory,
)
from .geometry import (
    Box,
    Cone,
    Cylinder,
    Ellipsoid,
    Geometry,
    GeometryBase,
    GeometrySpec,
    HollowBox,
    Mesh,
    Prism,
    Pyramid,
    Sphere,
    Torus,
)
from .math import (
    quat_from_euler,
    quat_identity,
    quat_multiply,
    quat_to_rotation_matrix,
    quat_to_rotation_matrix_np,
)
__all__ = [
    "GeometryBase",
    "GeometrySpec",
    "Box", "Sphere", "Cylinder", "Cone", "Ellipsoid",
    "Pyramid", "Prism", "Torus", "HollowBox",
    "AntennaId",
    "AntennaPattern",
    "AntennaState",
    "ReceiverGrid",
    "AssignmentId",
    "ComplexLike",
    "Deformation",
    "DeformationState",
    "DispersionSpec",
    "DynamicScene",
    "EndpointState",
    "MaterialAssignment",
    "MaterialCapabilities",
    "MaterialId",
    "MaterialLayer",
    "MaterialSpec",
    "Mesh",
    "FrequencyMaterialSample",
    "Geometry",
    "LinearTrajectory",
    "PhaseScreen",
    "PhysicalMaterial",
    "PowerLawDispersion",
    "PrimitiveId",
    "RigidMotion",
    "ScalarLike",
    "Scene",
    "SceneSnapshot",
    "SceneVersions",
    "StaticMaterialSample",
    "Structure",
    "StructureId",
    "SurfaceId",
    "StructureState",
    "SurfaceRoughness",
    "Trajectory",
    "antenna_orientation_matrix",
    "quat_from_euler",
    "quat_identity",
    "quat_multiply",
    "quat_to_rotation_matrix",
    "quat_to_rotation_matrix_np",
]
