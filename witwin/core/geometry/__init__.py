"""Shared differentiable geometry package."""

from .base import GeometryBase, GeometrySpec
from .mesh import Mesh
from .primitives import Box, Cone, Cylinder, Ellipsoid, HollowBox, Prism, Pyramid, Sphere, Torus

Geometry = Box | Sphere | Cylinder | Cone | Ellipsoid | Pyramid | Prism | Torus | HollowBox | Mesh

__all__ = [
    "GeometryBase",
    "GeometrySpec",
    "Box",
    "Sphere",
    "Cylinder",
    "Cone",
    "Ellipsoid",
    "Pyramid",
    "Prism",
    "Torus",
    "HollowBox",
    "Mesh",
    "Geometry",
]
