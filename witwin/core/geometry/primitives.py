"""Primitive differentiable geometry definitions."""

from __future__ import annotations

import numpy as np
import torch

from .base import (
    GeometryBase,
    _as_scalar,
    _as_vec3,
    _axial_split,
    _constant_tensor,
)
from .mesh_sdf import triangle_mesh_unsigned_distance


def _faces_tensor(data, *, device) -> torch.Tensor:
    return _constant_tensor(data, device=device, dtype=torch.int64)


def _box_signed_distance(dx, dy, dz, half_size: torch.Tensor) -> torch.Tensor:
    q = torch.stack(
        [
            torch.abs(dx) - half_size[0],
            torch.abs(dy) - half_size[1],
            torch.abs(dz) - half_size[2],
        ],
        dim=-1,
    )
    outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
    inside = torch.clamp(torch.amax(q, dim=-1), max=0.0)
    return outside + inside


def _cross_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _safe_signed_denom(value: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.where(value >= 0.0, value.clamp_min(eps), value.clamp_max(-eps))


def _polygon_edge_distance_sq_2d(
    points: torch.Tensor,
    polygon: torch.Tensor,
    *,
    edges: torch.Tensor | None = None,
    rel: torch.Tensor | None = None,
) -> torch.Tensor:
    """Squared unsigned distance from (P, 2) points to the closed edge loop of an (M, 2) polygon.

    ``edges`` and ``rel`` may be passed in when the caller has already materialized them.
    """
    if edges is None:
        edges = torch.roll(polygon, shifts=-1, dims=0) - polygon
    if rel is None:
        rel = points[:, None, :] - polygon[None, :, :]
    edge_norm_sq = torch.sum(edges ** 2, dim=-1).clamp_min(torch.finfo(points.dtype).eps)
    t = torch.clamp(torch.sum(rel * edges[None, :, :], dim=-1) / edge_norm_sq[None, :], 0.0, 1.0)
    closest = polygon[None, :, :] + t[..., None] * edges[None, :, :]
    return torch.sum((points[:, None, :] - closest) ** 2, dim=-1).min(dim=1).values.clamp_min(0.0)


def _convex_polygon_signed_distance_2d(points: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    next_polygon = torch.roll(polygon, shifts=-1, dims=0)
    edges = next_polygon - polygon
    rel = points[:, None, :] - polygon[None, :, :]
    dist = torch.sqrt(_polygon_edge_distance_sq_2d(points, polygon, edges=edges, rel=rel))

    polygon_area = 0.5 * torch.sum(_cross_2d(polygon, next_polygon))
    crosses = _cross_2d(edges[None, :, :], rel)
    if polygon_area.detach().item() >= 0.0:
        inside = torch.all(crosses >= -1.0e-7, dim=1)
    else:
        inside = torch.all(crosses <= 1.0e-7, dim=1)
    return torch.where(inside, -dist, dist)


def _polygon_crossing_count_2d(points: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    """Count +u horizontal-ray crossings of a closed (M, 2) polygon loop for each (P, 2) point."""
    start = polygon
    end = torch.roll(polygon, shifts=-1, dims=0)
    pu = points[:, 0:1]
    pv = points[:, 1:2]
    straddles = (start[None, :, 1] <= pv) != (end[None, :, 1] <= pv)
    denom = _safe_signed_denom(end[:, 1] - start[:, 1], torch.finfo(points.dtype).eps)
    t = (pv - start[None, :, 1]) / denom[None, :]
    crossing_u = start[None, :, 0] + t * (end[:, 0] - start[:, 0])[None, :]
    return (straddles & (pu < crossing_u)).sum(dim=1)


def _polygon_loops_signed_distance_2d(points: torch.Tensor, loops) -> torch.Tensor:
    """Even-odd signed distance from (P, 2) points to one or more closed 2D polygon loops.

    Negative inside; the interior is defined by the even-odd rule over the combined
    crossing count of all loops, so winding direction is irrelevant and additional
    loops carve holes. Valid for arbitrary (including non-convex and self-intersecting)
    polygons.
    """
    distance_sq = torch.stack([_polygon_edge_distance_sq_2d(points, loop) for loop in loops], dim=0)
    distance = torch.sqrt(distance_sq.min(dim=0).values)
    crossings = sum(_polygon_crossing_count_2d(points, loop) for loop in loops)
    inside = crossings % 2 == 1
    return torch.where(inside, -distance, distance)


def _canonical_coords(dx, dy, dz, axis: str):
    if axis == "z":
        return dx, dy, dz
    if axis == "y":
        return dx, dz, dy
    return dz, dy, dx


def _pyramid_mesh_signed_distance(points: torch.Tensor, base_size: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    half_base = base_size / 2
    zero = height.new_zeros(())
    vertices = torch.stack([
        torch.stack([zero, zero, zero]),
        torch.stack([-half_base, -half_base, height]),
        torch.stack([half_base, -half_base, height]),
        torch.stack([half_base, half_base, height]),
        torch.stack([-half_base, half_base, height]),
    ])
    faces = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [0, 4, 3],
            [0, 1, 4],
            [1, 2, 3],
            [1, 3, 4],
        ],
        device=points.device,
        dtype=torch.int64,
    )
    unsigned_distance = triangle_mesh_unsigned_distance(points, vertices.to(device=points.device, dtype=points.dtype), faces)

    axial = points[:, 2]
    lateral_x = points[:, 0]
    lateral_y = points[:, 1]
    half_extent = torch.clamp(0.5 * base_size * axial / _safe_signed_denom(height, torch.finfo(points.dtype).eps), min=0.0)
    inside = (
        (axial >= 0.0)
        & (axial <= height)
        & (torch.abs(lateral_x) <= half_extent)
        & (torch.abs(lateral_y) <= half_extent)
    )
    return torch.where(inside, -unsigned_distance, unsigned_distance)


class Box(GeometryBase):
    kind = "box"

    def __init__(self, position=(0, 0, 0), size=(1, 1, 1), rotation=None, *, device=None):
        super().__init__(position=position, rotation=rotation, device=device)
        self.size: torch.Tensor = _as_vec3(size, device=device)

    def signed_distance(self, xx, yy, zz):
        dx, dy, dz = self._local_coords(xx, yy, zz)
        half_size = self.size.to(device=dx.device, dtype=dx.dtype) / 2
        return _box_signed_distance(dx, dy, dz, half_size)

    def to_mesh(self, segments=16):
        del segments
        half_size = self.size.to(dtype=torch.float32) / 2
        vertices = torch.stack([
            torch.stack([-half_size[0], -half_size[1], -half_size[2]]),
            torch.stack([half_size[0], -half_size[1], -half_size[2]]),
            torch.stack([half_size[0], half_size[1], -half_size[2]]),
            torch.stack([-half_size[0], half_size[1], -half_size[2]]),
            torch.stack([-half_size[0], -half_size[1], half_size[2]]),
            torch.stack([half_size[0], -half_size[1], half_size[2]]),
            torch.stack([half_size[0], half_size[1], half_size[2]]),
            torch.stack([-half_size[0], half_size[1], half_size[2]]),
        ])
        vertices = self._transform_mesh_verts(vertices)
        faces = _faces_tensor([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
            [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
        ], device=vertices.device)
        return vertices, faces


class Sphere(GeometryBase):
    kind = "sphere"

    def __init__(self, position=(0, 0, 0), radius=1.0, rotation=None, *, device=None):
        super().__init__(position=position, rotation=rotation, device=device)
        self.radius: torch.Tensor = _as_scalar(radius, device=device)

    def signed_distance(self, xx, yy, zz):
        dx = xx - self.position[0]
        dy = yy - self.position[1]
        dz = zz - self.position[2]
        radius = self.radius.to(device=dx.device, dtype=dx.dtype)
        return torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2) - radius

    def to_mesh(self, segments=16):
        rings, sectors = segments, segments * 2
        device = self.device
        radius = self.radius.to(device=device, dtype=torch.float32)
        vertices = []
        for ring in range(rings + 1):
            phi = np.pi * ring / rings
            for sector in range(sectors):
                theta = 2 * np.pi * sector / sectors
                vertices.append([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi),
                ])
        vertices = _constant_tensor(vertices, device=device) * radius
        vertices = self._transform_mesh_verts(vertices)
        faces = []
        for ring in range(rings):
            for sector in range(sectors):
                next_sector = (sector + 1) % sectors
                v0 = ring * sectors + sector
                v1 = ring * sectors + next_sector
                v2 = (ring + 1) * sectors + sector
                v3 = (ring + 1) * sectors + next_sector
                if ring > 0:
                    faces.append([v0, v2, v1])
                if ring < rings - 1:
                    faces.append([v1, v2, v3])
        return vertices, _faces_tensor(faces, device=device)


class Cylinder(GeometryBase):
    kind = "cylinder"

    def __init__(self, position=(0, 0, 0), radius=1.0, height=1.0, axis="z", rotation=None, *, device=None):
        super().__init__(position=position, rotation=rotation, device=device)
        self.radius: torch.Tensor = _as_scalar(radius, device=device)
        self.height: torch.Tensor = _as_scalar(height, device=device)
        self.axis: str = self._validate_axis(axis)

    def signed_distance(self, xx, yy, zz):
        dx, dy, dz = self._local_coords(xx, yy, zz)
        axial, radial_a, radial_b = _axial_split(dx, dy, dz, self.axis)
        radius = self.radius.to(device=axial.device, dtype=axial.dtype)
        half_height = self.height.to(device=axial.device, dtype=axial.dtype) / 2
        radial_distance = torch.sqrt(radial_a ** 2 + radial_b ** 2)
        d = torch.stack([radial_distance - radius, torch.abs(axial) - half_height], dim=0)
        outside = torch.sqrt(torch.clamp(d[0], min=0.0) ** 2 + torch.clamp(d[1], min=0.0) ** 2)
        inside = torch.clamp(torch.amax(d, dim=0), max=0.0)
        return outside + inside

    def to_mesh(self, segments=16):
        device = self.device
        half_height = self.height.to(device=device, dtype=torch.float32) / 2
        radius = self.radius.to(device=device, dtype=torch.float32)
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        cos_angles = _constant_tensor(np.cos(angles), device=device)
        sin_angles = _constant_tensor(np.sin(angles), device=device)
        zeros = torch.zeros(segments, device=device, dtype=torch.float32)
        bottom = torch.stack([radius * cos_angles, radius * sin_angles, (-half_height).expand(segments)], dim=1)
        top = torch.stack([radius * cos_angles, radius * sin_angles, half_height.expand(segments)], dim=1)
        caps = torch.stack([
            torch.stack([zeros.new_zeros(()), zeros.new_zeros(()), -half_height]),
            torch.stack([zeros.new_zeros(()), zeros.new_zeros(()), half_height]),
        ])
        vertices = torch.cat([bottom, top, caps], dim=0)
        vertices = self._apply_axis_transform(vertices, self.axis)
        vertices = self._transform_mesh_verts(vertices)
        bottom_center, top_center = 2 * segments, 2 * segments + 1
        faces = []
        for index in range(segments):
            next_index = (index + 1) % segments
            faces.extend([
                [index, next_index, segments + next_index],
                [index, segments + next_index, segments + index],
                [bottom_center, next_index, index],
                [top_center, segments + index, segments + next_index],
            ])
        return vertices, _faces_tensor(faces, device=device)


class Ellipsoid(GeometryBase):
    kind = "ellipsoid"

    def __init__(self, position=(0, 0, 0), radii=(1, 1, 1), rotation=None, *, device=None):
        super().__init__(position=position, rotation=rotation, device=device)
        self.radii: torch.Tensor = _as_vec3(radii, device=device)

    def signed_distance(self, xx, yy, zz):
        dx, dy, dz = self._local_coords(xx, yy, zz)
        radii = torch.clamp(self.radii.to(device=dx.device, dtype=dx.dtype), min=torch.finfo(dx.dtype).eps)
        points = torch.stack([dx, dy, dz], dim=-1)
        scaled = points / radii
        inv_scaled = points / (radii * radii)
        k0 = torch.linalg.norm(scaled, dim=-1)
        k1 = torch.linalg.norm(inv_scaled, dim=-1).clamp_min(torch.finfo(dx.dtype).eps)
        distance = k0 * (k0 - 1.0) / k1
        return torch.where(k0 < 1.0e-6, -torch.min(radii), distance)

    def to_mesh(self, segments=16):
        rings, sectors = segments, segments * 2
        device = self.device
        rx, ry, rz = self.radii.to(device=device, dtype=torch.float32)
        vertices = []
        for ring in range(rings + 1):
            phi = np.pi * ring / rings
            for sector in range(sectors):
                theta = 2 * np.pi * sector / sectors
                vertices.append([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi),
                ])
        vertices = _constant_tensor(vertices, device=device)
        vertices = torch.stack([vertices[:, 0] * rx, vertices[:, 1] * ry, vertices[:, 2] * rz], dim=1)
        vertices = self._transform_mesh_verts(vertices)
        faces = []
        for ring in range(rings):
            for sector in range(sectors):
                next_sector = (sector + 1) % sectors
                v0 = ring * sectors + sector
                v1 = ring * sectors + next_sector
                v2 = (ring + 1) * sectors + sector
                v3 = (ring + 1) * sectors + next_sector
                if ring > 0:
                    faces.append([v0, v2, v1])
                if ring < rings - 1:
                    faces.append([v1, v2, v3])
        return vertices, _faces_tensor(faces, device=device)


class Cone(GeometryBase):
    kind = "cone"

    def __init__(self, position=(0, 0, 0), radius=1.0, height=1.0, axis="z", rotation=None, *, device=None):
        super().__init__(position=position, rotation=rotation, device=device)
        self.radius: torch.Tensor = _as_scalar(radius, device=device)
        self.height: torch.Tensor = _as_scalar(height, device=device)
        self.axis: str = self._validate_axis(axis)

    def signed_distance(self, xx, yy, zz):
        dx, dy, dz = self._local_coords(xx, yy, zz)
        axial, radial_a, radial_b = _axial_split(dx, dy, dz, self.axis)
        height = self.height.to(device=axial.device, dtype=axial.dtype)
        radius = self.radius.to(device=axial.device, dtype=axial.dtype)
        radial = torch.sqrt(radial_a ** 2 + radial_b ** 2)
        profile_points = torch.stack([axial, radial], dim=-1).reshape(-1, 2)
        triangle = torch.stack([
            torch.stack([height.new_zeros(()), height.new_zeros(())]),
            torch.stack([height, radius]),
            torch.stack([height, -radius]),
        ])
        return _convex_polygon_signed_distance_2d(profile_points, triangle).reshape(axial.shape)

    def to_mesh(self, segments=16):
        device = self.device
        radius = self.radius.to(device=device, dtype=torch.float32)
        height = self.height.to(device=device, dtype=torch.float32)
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        cos_angles = _constant_tensor(np.cos(angles), device=device)
        sin_angles = _constant_tensor(np.sin(angles), device=device)
        zeros = torch.zeros(segments, device=device, dtype=torch.float32)
        apex = torch.stack([torch.stack([zeros.new_zeros(()), zeros.new_zeros(()), zeros.new_zeros(())])])
        base = torch.stack([radius * cos_angles, radius * sin_angles, height.expand(segments)], dim=1)
        base_center = torch.stack([torch.stack([zeros.new_zeros(()), zeros.new_zeros(()), height])])
        vertices = torch.cat([apex, base, base_center], dim=0)
        vertices = self._apply_axis_transform(vertices, self.axis)
        vertices = self._transform_mesh_verts(vertices)
        apex_index, base_center_index = 0, segments + 1
        faces = []
        for index in range(segments):
            next_index = (index + 1) % segments
            v0 = 1 + index
            v1 = 1 + next_index
            faces.extend([[apex_index, v1, v0], [base_center_index, v0, v1]])
        return vertices, _faces_tensor(faces, device=device)


class Pyramid(GeometryBase):
    kind = "pyramid"

    def __init__(self, position=(0, 0, 0), base_size=1.0, height=1.0, axis="z", rotation=None, *, device=None):
        super().__init__(position=position, rotation=rotation, device=device)
        self.base_size: torch.Tensor = _as_scalar(base_size, device=device)
        self.height: torch.Tensor = _as_scalar(height, device=device)
        self.axis: str = self._validate_axis(axis)

    def signed_distance(self, xx, yy, zz):
        dx, dy, dz = self._local_coords(xx, yy, zz)
        height = self.height.to(device=dx.device, dtype=dx.dtype)
        base_size = self.base_size.to(device=dx.device, dtype=dx.dtype)
        cx, cy, cz = _canonical_coords(dx, dy, dz, self.axis)
        points = torch.stack([cx, cy, cz], dim=-1).reshape(-1, 3)
        return _pyramid_mesh_signed_distance(points, base_size, height).reshape(dx.shape)

    def to_mesh(self, segments=16):
        del segments
        device = self.device
        half_size = self.base_size.to(device=device, dtype=torch.float32) / 2
        height = self.height.to(device=device, dtype=torch.float32)
        zero = torch.tensor(0.0, device=device, dtype=torch.float32)
        vertices = torch.stack([
            torch.stack([zero, zero, zero]),
            torch.stack([-half_size, -half_size, height]),
            torch.stack([half_size, -half_size, height]),
            torch.stack([half_size, half_size, height]),
            torch.stack([-half_size, half_size, height]),
        ])
        vertices = self._apply_axis_transform(vertices, self.axis)
        vertices = self._transform_mesh_verts(vertices)
        faces = _faces_tensor([[0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4], [1, 2, 3], [1, 3, 4]], device=device)
        return vertices, faces


class Prism(GeometryBase):
    kind = "prism"

    def __init__(self, position=(0, 0, 0), radius=1.0, height=1.0, num_sides=6, axis="z", rotation=None, *, device=None):
        super().__init__(position=position, rotation=rotation, device=device)
        self.radius: torch.Tensor = _as_scalar(radius, device=device)
        self.height: torch.Tensor = _as_scalar(height, device=device)
        if int(num_sides) < 3:
            raise ValueError("num_sides must be >= 3.")
        self.num_sides: int = int(num_sides)
        self.axis: str = self._validate_axis(axis)

    def signed_distance(self, xx, yy, zz):
        dx, dy, dz = self._local_coords(xx, yy, zz)
        axial, px, py = _axial_split(dx, dy, dz, self.axis)
        radius = torch.clamp(self.radius.to(device=px.device, dtype=px.dtype), min=torch.finfo(px.dtype).eps)
        angles = torch.linspace(0, 2 * np.pi, self.num_sides + 1, device=px.device)[:-1]
        polygon = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
        polygon_distance = _convex_polygon_signed_distance_2d(torch.stack([px, py], dim=-1).reshape(-1, 2), polygon).reshape(px.shape)
        axial_distance = torch.abs(axial) - self.height.to(device=axial.device, dtype=axial.dtype) / 2
        outside = torch.sqrt(torch.clamp(polygon_distance, min=0.0) ** 2 + torch.clamp(axial_distance, min=0.0) ** 2)
        inside = torch.clamp(torch.maximum(polygon_distance, axial_distance), max=0.0)
        return outside + inside

    def to_mesh(self, segments=16):
        del segments
        device = self.device
        half_height = self.height.to(device=device, dtype=torch.float32) / 2
        radius = self.radius.to(device=device, dtype=torch.float32)
        side_count = self.num_sides
        angles = np.linspace(0, 2 * np.pi, side_count, endpoint=False)
        cos_angles = _constant_tensor(np.cos(angles), device=device)
        sin_angles = _constant_tensor(np.sin(angles), device=device)
        bottom = torch.stack([radius * cos_angles, radius * sin_angles, -half_height.expand(side_count)], dim=1)
        top = torch.stack([radius * cos_angles, radius * sin_angles, half_height.expand(side_count)], dim=1)
        zero = torch.tensor(0.0, device=device, dtype=torch.float32)
        caps = torch.stack([
            torch.stack([zero, zero, -half_height]),
            torch.stack([zero, zero, half_height]),
        ])
        vertices = torch.cat([bottom, top, caps], dim=0)
        vertices = self._apply_axis_transform(vertices, self.axis)
        vertices = self._transform_mesh_verts(vertices)
        bottom_center, top_center = 2 * side_count, 2 * side_count + 1
        faces = []
        for index in range(side_count):
            next_index = (index + 1) % side_count
            faces.extend([
                [index, next_index, side_count + next_index],
                [index, side_count + next_index, side_count + index],
                [bottom_center, next_index, index],
                [top_center, side_count + index, side_count + next_index],
            ])
        return vertices, _faces_tensor(faces, device=device)


class Torus(GeometryBase):
    kind = "torus"

    def __init__(self, position=(0, 0, 0), major_radius=1.0, minor_radius=0.25, axis="z", rotation=None, *, device=None):
        super().__init__(position=position, rotation=rotation, device=device)
        self.major_radius: torch.Tensor = _as_scalar(major_radius, device=device)
        self.minor_radius: torch.Tensor = _as_scalar(minor_radius, device=device)
        self.axis: str = self._validate_axis(axis)

    def signed_distance(self, xx, yy, zz):
        dx, dy, dz = self._local_coords(xx, yy, zz)
        axial, radial_a, radial_b = _axial_split(dx, dy, dz, self.axis)
        major_plane = torch.sqrt(radial_a ** 2 + radial_b ** 2)
        major_radius = self.major_radius.to(device=axial.device, dtype=axial.dtype)
        minor_radius = self.minor_radius.to(device=axial.device, dtype=axial.dtype)
        return torch.sqrt((major_plane - major_radius) ** 2 + axial ** 2) - minor_radius

    def to_mesh(self, segments=16):
        device = self.device
        major_radius = self.major_radius.to(device=device, dtype=torch.float32)
        minor_radius = self.minor_radius.to(device=device, dtype=torch.float32)
        major_segments, minor_segments = segments * 2, segments
        vertices = []
        for major in range(major_segments):
            theta = 2 * np.pi * major / major_segments
            for minor in range(minor_segments):
                phi = 2 * np.pi * minor / minor_segments
                vertices.append([
                    np.cos(phi) * np.cos(theta),
                    np.cos(phi) * np.sin(theta),
                    np.sin(phi),
                    np.cos(theta),
                    np.sin(theta),
                ])
        vertices = _constant_tensor(vertices, device=device)
        ring_radius = major_radius + minor_radius * vertices[:, 0]
        vertices = torch.stack([
            ring_radius * vertices[:, 3],
            ring_radius * vertices[:, 4],
            minor_radius * vertices[:, 2],
        ], dim=1)
        vertices = self._apply_axis_transform(vertices, self.axis)
        vertices = self._transform_mesh_verts(vertices)
        faces = []
        for major in range(major_segments):
            next_major = (major + 1) % major_segments
            for minor in range(minor_segments):
                next_minor = (minor + 1) % minor_segments
                v0 = major * minor_segments + minor
                v1 = major * minor_segments + next_minor
                v2 = next_major * minor_segments + minor
                v3 = next_major * minor_segments + next_minor
                faces.extend([[v0, v2, v1], [v1, v2, v3]])
        return vertices, _faces_tensor(faces, device=device)


class HollowBox(GeometryBase):
    kind = "hollow_box"

    def __init__(self, position=(0, 0, 0), outer_size=(1, 1, 1), inner_size=(0.8, 0.8, 0.8), rotation=None, *, device=None):
        super().__init__(position=position, rotation=rotation, device=device)
        self.outer_size: torch.Tensor = _as_vec3(outer_size, device=device)
        self.inner_size: torch.Tensor = _as_vec3(inner_size, device=device)

    @property
    def size(self):
        return tuple(self.outer_size.tolist())

    def signed_distance(self, xx, yy, zz):
        dx, dy, dz = self._local_coords(xx, yy, zz)
        outer_half_size = self.outer_size.to(device=dx.device, dtype=dx.dtype) / 2
        inner_half_size = self.inner_size.to(device=dx.device, dtype=dx.dtype) / 2
        outer = _box_signed_distance(dx, dy, dz, outer_half_size)
        inner = _box_signed_distance(dx, dy, dz, inner_half_size)
        return torch.maximum(outer, -inner)

    def to_mesh(self, segments=16):
        del segments
        outer = Box(position=self.position, size=self.outer_size, rotation=self.rotation, device=self.device)
        inner = Box(position=self.position, size=self.inner_size, rotation=self.rotation, device=self.device)
        outer_vertices, outer_faces = outer.to_mesh()
        inner_vertices, inner_faces = inner.to_mesh()
        inner_faces = torch.flip(inner_faces, dims=(1,))
        return (
            torch.cat([outer_vertices, inner_vertices], dim=0),
            torch.cat([outer_faces, inner_faces + outer_vertices.shape[0]], dim=0),
        )


def _as_polygon_vertices(value, *, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        vertices = value.to(device=device, dtype=torch.float32)
    else:
        vertices = torch.tensor([[float(u), float(v)] for u, v in value], dtype=torch.float32, device=device)
    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
        raise ValueError("vertices must have shape (N, 2) with N >= 3.")
    return vertices


def _as_axis_bounds(value, *, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        bounds = value.to(device=device, dtype=torch.float32)
    else:
        bounds = torch.tensor([float(value[0]), float(value[1])], dtype=torch.float32, device=device)
    if bounds.shape != (2,):
        raise ValueError("bounds must contain (min, max) along the extrusion axis.")
    return bounds


def _cross_2d_scalar(origin, first, second) -> float:
    return float((first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0]))


def _point_in_triangle_2d(point, a, b, c, eps=1.0e-12) -> bool:
    """Inclusive point-in-triangle test for a CCW triangle (a, b, c)."""
    return (
        _cross_2d_scalar(a, b, point) >= -eps
        and _cross_2d_scalar(b, c, point) >= -eps
        and _cross_2d_scalar(c, a, point) >= -eps
    )


def _ear_clip_faces_2d(polygon: np.ndarray) -> list[list[int]]:
    """Triangulate a simple CCW (N, 2) polygon into CCW index triangles via ear clipping."""
    remaining = list(range(polygon.shape[0]))
    faces: list[list[int]] = []
    while len(remaining) > 3:
        clipped = False
        for slot in range(len(remaining)):
            i0 = remaining[slot - 1]
            i1 = remaining[slot]
            i2 = remaining[(slot + 1) % len(remaining)]
            a, b, c = polygon[i0], polygon[i1], polygon[i2]
            if _cross_2d_scalar(a, b, c) <= 1.0e-12:
                continue
            others = (polygon[j] for j in remaining if j not in (i0, i1, i2))
            if any(_point_in_triangle_2d(p, a, b, c) for p in others):
                continue
            faces.append([i0, i1, i2])
            remaining.pop(slot)
            clipped = True
            break
        if not clipped:
            raise ValueError("Failed to triangulate polygon; vertices must describe a simple polygon.")
    faces.append(list(remaining))
    return faces


def _offset_polygon_2d(polygon: np.ndarray, offset: float) -> np.ndarray:
    """Approximate outward miter offset of a CCW (N, 2) polygon (negative offset shrinks)."""
    if offset == 0.0:
        return polygon
    previous_edges = polygon - np.roll(polygon, 1, axis=0)
    next_edges = np.roll(polygon, -1, axis=0) - polygon

    def _outward_normals(edges: np.ndarray) -> np.ndarray:
        normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
        return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)

    normal_previous = _outward_normals(previous_edges)
    normal_next = _outward_normals(next_edges)
    denom = np.maximum(1.0 + np.sum(normal_previous * normal_next, axis=1, keepdims=True), 1.0e-3)
    return polygon + offset * (normal_previous + normal_next) / denom


class PolySlab(GeometryBase):
    """Extruded 2D polygon with optional linear sidewall taper (Tidy3D semantics).

    ``vertices`` is the (N, 2) cross-section in the plane perpendicular to ``axis``
    (plane coordinates follow ``_axial_split``: axis z -> (x, y), y -> (x, z),
    x -> (y, z)), ``bounds`` is (min, max) along the axis, and a positive
    ``sidewall_angle`` (radians) shrinks the cross-section as the axis coordinate
    increases. ``reference_plane`` selects the axis position where the drawn
    polygon is exact. The interior uses the even-odd rule, so non-convex polygons
    are supported and vertex winding is irrelevant.
    """

    kind = "poly_slab"

    def __init__(
        self,
        vertices,
        bounds,
        axis="z",
        sidewall_angle=0.0,
        reference_plane="middle",
        position=(0, 0, 0),
        rotation=None,
        *,
        device=None,
    ):
        super().__init__(position=position, rotation=rotation, device=device)
        self.vertices: torch.Tensor = _as_polygon_vertices(vertices, device=device)
        self.bounds: torch.Tensor = _as_axis_bounds(bounds, device=device)
        if isinstance(axis, int) and not isinstance(axis, bool):
            if axis not in (0, 1, 2):
                raise ValueError("integer axis must be 0, 1, or 2.")
            axis = ("x", "y", "z")[axis]
        self.axis: str = self._validate_axis(axis)
        self.sidewall_angle: torch.Tensor = _as_scalar(sidewall_angle, device=device)
        if reference_plane not in ("bottom", "middle", "top"):
            raise ValueError("reference_plane must be 'bottom', 'middle', or 'top'.")
        self.reference_plane: str = reference_plane
        self.loop_sizes: tuple[int, ...] = (int(self.vertices.shape[0]),)

    def _reference_position(self, bounds: torch.Tensor) -> torch.Tensor:
        if self.reference_plane == "bottom":
            return bounds[0]
        if self.reference_plane == "top":
            return bounds[1]
        return 0.5 * (bounds[0] + bounds[1])

    def signed_distance(self, xx, yy, zz):
        dx, dy, dz = self._local_coords(xx, yy, zz)
        axial, pu, pv = _axial_split(dx, dy, dz, self.axis)
        vertices = self.vertices.to(device=pu.device, dtype=pu.dtype)
        loops = torch.split(vertices, list(self.loop_sizes), dim=0)
        points = torch.stack([pu, pv], dim=-1).reshape(-1, 2)
        polygon_distance = _polygon_loops_signed_distance_2d(points, loops).reshape(pu.shape)
        bounds = self.bounds.to(device=axial.device, dtype=axial.dtype)
        tan_angle = torch.tan(self.sidewall_angle.to(device=axial.device, dtype=axial.dtype))
        # Signed distance minus an offset is exact morphological erosion/dilation, so
        # the taper is a z-dependent inward offset of the cross-section.
        tapered_distance = polygon_distance + tan_angle * (axial - self._reference_position(bounds))
        axial_distance = torch.abs(axial - 0.5 * (bounds[0] + bounds[1])) - 0.5 * (bounds[1] - bounds[0])
        outside = torch.sqrt(torch.clamp(tapered_distance, min=0.0) ** 2 + torch.clamp(axial_distance, min=0.0) ** 2)
        inside = torch.clamp(torch.maximum(tapered_distance, axial_distance), max=0.0)
        return outside + inside

    def to_mesh(self, segments=16):
        del segments
        device = self.device
        polygon = self.vertices.detach().cpu().to(torch.float64).numpy()
        # Drop consecutive duplicate vertices (including a closed-ring wrap-around
        # repeat); coincident points would block every candidate ear below.
        keep = np.linalg.norm(polygon - np.roll(polygon, 1, axis=0), axis=1) > 1.0e-12
        polygon = polygon[keep]
        if polygon.shape[0] < 3:
            raise ValueError("vertices must contain at least 3 distinct points.")
        signed_area = 0.5 * float(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - np.roll(polygon[:, 0], -1) * polygon[:, 1]))
        if signed_area < 0.0:
            polygon = polygon[::-1].copy()
        bounds = self.bounds.detach().cpu().to(torch.float64).numpy()
        reference = {"bottom": bounds[0], "top": bounds[1], "middle": 0.5 * (bounds[0] + bounds[1])}[self.reference_plane]
        tan_angle = float(np.tan(self.sidewall_angle.detach().cpu().to(torch.float64).item()))
        bottom_ring = _offset_polygon_2d(polygon, -tan_angle * (bounds[0] - reference))
        top_ring = _offset_polygon_2d(polygon, -tan_angle * (bounds[1] - reference))
        count = polygon.shape[0]
        bottom = np.concatenate([bottom_ring, np.full((count, 1), bounds[0])], axis=1)
        top = np.concatenate([top_ring, np.full((count, 1), bounds[1])], axis=1)
        vertices = _constant_tensor(np.concatenate([bottom, top], axis=0), device=device)
        faces = []
        for index in range(count):
            next_index = (index + 1) % count
            faces.extend([
                [index, next_index, count + next_index],
                [index, count + next_index, count + index],
            ])
        for i0, i1, i2 in _ear_clip_faces_2d(polygon):
            faces.append([i0, i2, i1])
            faces.append([count + i0, count + i1, count + i2])
        # Map local (u, v, axial) columns to world axes following _axial_split
        # (axis z -> (x, y), y -> (x, z), x -> (y, z)); _remap_axis_torch would
        # mirror u/v for axis "x" and disagree with signed_distance/to_mask.
        vertices = vertices[:, {"z": [0, 1, 2], "y": [0, 2, 1], "x": [2, 0, 1]}[self.axis]]
        faces_tensor = _faces_tensor(faces, device=device)
        if self.axis == "y":
            # Odd column permutation reflects orientation; flip winding to keep
            # face normals outward.
            faces_tensor = torch.flip(faces_tensor, dims=(1,))
        vertices = self._transform_mesh_verts(vertices)
        return vertices, faces_tensor


class ComplexPolySlab(PolySlab):
    """PolySlab variant for non-simple cross-sections.

    Accepts a list of vertex loops; self-intersecting loops (e.g. bowties) and
    multiple loops are allowed. The interior is the even-odd rule over the combined
    crossing count of all loops, so loop winding is irrelevant and extra loops carve
    holes. The distance magnitude is the minimum distance over all edges of all loops.
    """

    kind = "complex_poly_slab"

    def __init__(
        self,
        loops,
        bounds,
        axis="z",
        sidewall_angle=0.0,
        reference_plane="middle",
        position=(0, 0, 0),
        rotation=None,
        *,
        device=None,
    ):
        loop_tensors = [_as_polygon_vertices(loop, device=device) for loop in loops]
        if not loop_tensors:
            raise ValueError("loops must contain at least one polygon loop.")
        super().__init__(
            torch.cat(loop_tensors, dim=0),
            bounds,
            axis=axis,
            sidewall_angle=sidewall_angle,
            reference_plane=reference_plane,
            position=position,
            rotation=rotation,
            device=device,
        )
        self.loop_sizes = tuple(int(loop.shape[0]) for loop in loop_tensors)

    def to_mesh(self, segments=16):
        raise NotImplementedError("ComplexPolySlab does not support mesh export.")
