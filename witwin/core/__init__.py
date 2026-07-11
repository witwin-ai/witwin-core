"""Witwin Core - Shared geometry, material, and structure contracts."""

__version__ = "0.3.0"

from .material import (
    FrequencyMaterialSample,
    Material,
    MaterialCapabilities,
    MaterialSpec,
    StaticMaterialSample,
    Structure,
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
    "Material",
    "MaterialCapabilities",
    "MaterialSpec",
    "Mesh",
    "FrequencyMaterialSample",
    "Geometry",
    "StaticMaterialSample",
    "Structure",
    "quat_from_euler",
    "quat_identity",
    "quat_multiply",
    "quat_to_rotation_matrix",
    "quat_to_rotation_matrix_np",
]
