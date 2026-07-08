"""Shared differentiable geometry package."""

from .base import GeometryBase
from .mesh import Mesh
from .primitives import Box, ComplexPolySlab, Cone, Cylinder, Ellipsoid, HollowBox, PolySlab, Prism, Pyramid, Sphere, Torus
from .smpl import SMPLBody

Geometry = Box | Sphere | Cylinder | Cone | Ellipsoid | Pyramid | Prism | PolySlab | ComplexPolySlab | Torus | HollowBox | Mesh | SMPLBody

__all__ = [
    "GeometryBase",
    "Box",
    "Sphere",
    "Cylinder",
    "Cone",
    "Ellipsoid",
    "Pyramid",
    "Prism",
    "PolySlab",
    "ComplexPolySlab",
    "Torus",
    "HollowBox",
    "Mesh",
    "SMPLBody",
    "Geometry",
]
