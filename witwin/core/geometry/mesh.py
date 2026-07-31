"""Triangle mesh representation with OBJ loading and topology analysis."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np
import torch

from ..math import quat_to_rotation_matrix
from .base import (
    GeometryBase,
    _as_vec3,
    _resolve_device,
    _rotation_quaternion,
)
from .mesh_sdf import (
    _build_triangle_bvh,
    _mesh_sdf_available,
    _triangle_bvh_to_device,
    triangle_mesh_signed_distance_static_bvh,
    triangle_mesh_smooth_signed_distance,
    triangle_mesh_unsigned_distance_static_bvh,
    triangle_mesh_unsigned_distance,
)


def _vertices_array(vertices) -> np.ndarray:
    if isinstance(vertices, torch.Tensor):
        raise TypeError(
            "_vertices_array does not accept tensors; tensor-authored geometry "
            "must not be materialized on the host."
        )
    array = np.asarray(vertices, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError("vertices must have shape (N, 3) with N > 0.")
    return np.ascontiguousarray(array)


def _vertices_tensor(vertices, *, device=None) -> torch.Tensor:
    if isinstance(vertices, torch.Tensor):
        tensor = vertices
    else:
        tensor = torch.tensor(vertices, device=device, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != 3 or tensor.shape[0] == 0:
        raise ValueError("vertices must have shape (N, 3) with N > 0.")
    if not tensor.dtype.is_floating_point:
        raise TypeError("vertices must use a floating-point dtype.")
    return tensor


def _faces_array(faces, *, vertex_count: int) -> np.ndarray:
    if isinstance(faces, torch.Tensor):
        raise TypeError(
            "_faces_array does not accept tensors; tensor-authored topology "
            "must not be materialized on the host."
        )
    array = np.asarray(faces)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError("faces must have shape (M, 3) with M > 0.")
    if array.dtype not in {np.dtype(np.int32), np.dtype(np.int64)}:
        raise TypeError("faces must use int32 or int64.")
    if np.any(array < 0) or np.any(array >= vertex_count):
        raise ValueError("faces contains vertex indices outside the valid range.")
    return np.ascontiguousarray(array, dtype=np.int64)


def _faces_tensor(faces, *, device=None) -> torch.Tensor:
    if isinstance(faces, torch.Tensor):
        tensor = faces
    else:
        tensor = torch.as_tensor(faces, device=device)
    if tensor.ndim != 2 or tensor.shape[1] != 3 or tensor.shape[0] == 0:
        raise ValueError("faces must have shape (M, 3) with M > 0.")
    if tensor.dtype not in {torch.int32, torch.int64}:
        raise TypeError("faces must use torch.int32 or torch.int64.")
    return tensor


def _triangulate_obj_face(indices: list[int]) -> list[tuple[int, int, int]]:
    if len(indices) < 3:
        return []
    triangles: list[tuple[int, int, int]] = []
    anchor = indices[0]
    for index in range(1, len(indices) - 1):
        tri = (anchor, indices[index], indices[index + 1])
        if len({tri[0], tri[1], tri[2]}) == 3:
            triangles.append(tri)
    return triangles


def _load_obj(path: str | PathLike[str]) -> tuple[np.ndarray, np.ndarray]:
    obj_path = Path(path).expanduser().resolve()
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                tokens = line.split()
                if len(tokens) < 4:
                    raise ValueError(f"Invalid vertex record at {obj_path}:{line_no}")
                vertices.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
                continue
            if not line.startswith("f "):
                continue
            tokens = line.split()[1:]
            if len(tokens) < 3:
                raise ValueError(f"Invalid face record at {obj_path}:{line_no}")
            face_indices: list[int] = []
            for token in tokens:
                vertex_token = token.split("/")[0]
                if not vertex_token:
                    raise ValueError(f"Unsupported OBJ face token at {obj_path}:{line_no}")
                raw_index = int(vertex_token)
                vertex_index = raw_index - 1 if raw_index > 0 else len(vertices) + raw_index
                if vertex_index < 0 or vertex_index >= len(vertices):
                    raise ValueError(f"OBJ face index out of range at {obj_path}:{line_no}")
                face_indices.append(vertex_index)
            faces.extend(_triangulate_obj_face(face_indices))

    if not vertices:
        raise ValueError(f"OBJ file does not contain any vertices: {obj_path}")
    if not faces:
        raise ValueError(f"OBJ file does not contain any triangular faces: {obj_path}")
    return _vertices_array(vertices), _faces_array(faces, vertex_count=len(vertices))


def _mesh_topology_stats(vertices: np.ndarray, faces: np.ndarray) -> tuple[int, int, int, int, float]:
    edge_counts: dict[tuple[int, int], int] = {}
    edge_orientation_balance: dict[tuple[int, int], int] = {}

    bbox_extent = vertices.max(axis=0) - vertices.min(axis=0)
    bbox_diagonal = float(np.linalg.norm(bbox_extent))
    area_epsilon = max((bbox_diagonal * bbox_diagonal) * 1.0e-12, np.finfo(np.float32).eps)

    degenerate_faces = 0
    for tri in faces:
        a_idx, b_idx, c_idx = (int(tri[0]), int(tri[1]), int(tri[2]))
        if len({a_idx, b_idx, c_idx}) < 3:
            degenerate_faces += 1
            continue

        a = vertices[a_idx]
        b = vertices[b_idx]
        c = vertices[c_idx]
        twice_area = np.linalg.norm(np.cross(b - a, c - a))
        if twice_area <= area_epsilon:
            degenerate_faces += 1

        edges = ((a_idx, b_idx), (b_idx, c_idx), (c_idx, a_idx))
        for start, end in edges:
            edge = (start, end) if start < end else (end, start)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            orientation = 1 if start == edge[0] else -1
            edge_orientation_balance[edge] = edge_orientation_balance.get(edge, 0) + orientation

    boundary_edges = sum(1 for count in edge_counts.values() if count == 1)
    non_manifold_edges = sum(1 for count in edge_counts.values() if count > 2)
    inconsistent_edge_orientations = sum(
        1
        for edge, count in edge_counts.items()
        if count == 2 and edge_orientation_balance.get(edge, 0) != 0
    )
    signed_volume = float(
        np.einsum(
            "ij,ij->i",
            vertices[faces[:, 0]],
            np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]),
        ).sum()
        / 6.0
    )
    return (
        boundary_edges,
        non_manifold_edges,
        degenerate_faces,
        inconsistent_edge_orientations,
        signed_volume,
    )


def _mesh_topology_stats_torch(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> tuple[int, int, int, int, float]:
    if vertices.device.type != "cpu" or faces.device.type != "cpu":
        raise ValueError(
            "Tensor topology diagnostics require CPU-authored vertices and faces."
        )
    with torch.no_grad():
        face_indices = faces.to(dtype=torch.int64)
        if bool(torch.any(face_indices < 0)) or bool(
            torch.any(face_indices >= vertices.shape[0])
        ):
            raise ValueError(
                "faces contains vertex indices outside the valid range."
            )
        triangles = vertices[face_indices]
        bbox_extent = vertices.amax(dim=0) - vertices.amin(dim=0)
        bbox_diagonal = torch.linalg.vector_norm(bbox_extent)
        area_epsilon = torch.maximum(
            bbox_diagonal.square() * 1.0e-12,
            torch.tensor(
                torch.finfo(vertices.dtype).eps,
                device=vertices.device,
                dtype=vertices.dtype,
            ),
        )
        repeated_vertex = (
            (face_indices[:, 0] == face_indices[:, 1])
            | (face_indices[:, 1] == face_indices[:, 2])
            | (face_indices[:, 2] == face_indices[:, 0])
        )
        twice_area = torch.linalg.vector_norm(
            torch.linalg.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
                dim=-1,
            ),
            dim=-1,
        )
        degenerate_faces = int(
            torch.count_nonzero(repeated_vertex | (twice_area <= area_epsilon))
        )

        starts = torch.cat(
            (face_indices[:, 0], face_indices[:, 1], face_indices[:, 2])
        )
        ends = torch.cat(
            (face_indices[:, 1], face_indices[:, 2], face_indices[:, 0])
        )
        edges = torch.stack(
            (torch.minimum(starts, ends), torch.maximum(starts, ends)), dim=1
        )
        _, inverse, counts = torch.unique(
            edges, dim=0, return_inverse=True, return_counts=True
        )
        orientation = torch.where(
            starts < ends,
            torch.ones_like(starts),
            -torch.ones_like(starts),
        )
        balance = torch.zeros_like(counts)
        balance.scatter_add_(0, inverse, orientation)
        boundary_edges = int(torch.count_nonzero(counts == 1))
        non_manifold_edges = int(torch.count_nonzero(counts > 2))
        inconsistent_edge_orientations = int(
            torch.count_nonzero((counts == 2) & (balance != 0))
        )
        signed_volume = float(
            torch.sum(
                triangles[:, 0]
                * torch.linalg.cross(
                    triangles[:, 1], triangles[:, 2], dim=-1
                )
            )
            / 6.0
        )
    return (
        boundary_edges,
        non_manifold_edges,
        degenerate_faces,
        inconsistent_edge_orientations,
        signed_volume,
    )


def _tensor_revision(tensor: torch.Tensor) -> tuple:
    return (
        str(tensor.device),
        str(tensor.dtype),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.data_ptr(),
        getattr(tensor, "_version", None),
    )


class Mesh(GeometryBase):
    """Explicit triangle mesh geometry."""

    kind = "mesh"

    def __init__(
        self,
        vertices,
        faces,
        *,
        position=(0.0, 0.0, 0.0),
        scale=1.0,
        rotation=None,
        recenter: bool = True,
        fill_mode: str = "auto",
        surface_thickness: float | None = None,
        source_path: str | PathLike[str] | None = None,
        topology_diagnostics: bool | None = None,
        device=None,
    ):
        resolved_device = _resolve_device(
            device,
            vertices,
            faces,
            position,
            rotation,
            scale,
            surface_thickness,
        )
        super().__init__(
            position=position, rotation=rotation, device=resolved_device
        )

        normalized_fill_mode = str(fill_mode).lower()
        if normalized_fill_mode not in {"auto", "solid", "surface"}:
            raise ValueError("fill_mode must be 'auto', 'solid', or 'surface'.")
        if isinstance(surface_thickness, torch.Tensor):
            if surface_thickness.ndim != 0:
                raise ValueError("surface_thickness tensor must be scalar.")
            _resolve_device(resolved_device, surface_thickness)
        elif surface_thickness is not None and float(surface_thickness) <= 0.0:
            raise ValueError(
                "surface_thickness must be positive when provided."
            )

        vertices_tensor = _vertices_tensor(vertices, device=self.position.device)
        faces_tensor = _faces_tensor(faces, device=self.position.device)
        tensor_authored = isinstance(vertices, torch.Tensor) or isinstance(
            faces, torch.Tensor
        )
        tensor_topology_is_cpu = (
            vertices_tensor.device.type == "cpu"
            and faces_tensor.device.type == "cpu"
        )
        diagnostics_enabled = (
            not tensor_authored
            or bool(topology_diagnostics)
            or (
                topology_diagnostics is None
                and normalized_fill_mode == "auto"
                and tensor_topology_is_cpu
            )
        )
        if diagnostics_enabled:
            if tensor_authored:
                (
                    boundary_edges,
                    non_manifold_edges,
                    degenerate_faces,
                    inconsistent_edge_orientations,
                    signed_volume,
                ) = _mesh_topology_stats_torch(
                    vertices_tensor, faces_tensor
                )
                with torch.no_grad():
                    bbox_diagonal = float(
                        torch.linalg.vector_norm(
                            vertices_tensor.amax(dim=0)
                            - vertices_tensor.amin(dim=0)
                        )
                    )
            else:
                vertices_array = _vertices_array(vertices)
                faces_array = _faces_array(
                    faces, vertex_count=int(vertices_tensor.shape[0])
                )
                (
                    boundary_edges,
                    non_manifold_edges,
                    degenerate_faces,
                    inconsistent_edge_orientations,
                    signed_volume,
                ) = _mesh_topology_stats(vertices_array, faces_array)
                bbox_extent = (
                    vertices_array.max(axis=0)
                    - vertices_array.min(axis=0)
                )
                bbox_diagonal = float(np.linalg.norm(bbox_extent))
            volume_epsilon = max(
                (bbox_diagonal**3) * 1.0e-9, np.finfo(np.float32).eps
            )
            is_watertight = bool(
                boundary_edges == 0
                and non_manifold_edges == 0
                and degenerate_faces == 0
                and inconsistent_edge_orientations == 0
                and abs(signed_volume) > volume_epsilon
            )
        else:
            boundary_edges = None
            non_manifold_edges = None
            degenerate_faces = None
            inconsistent_edge_orientations = None
            signed_volume = None
            is_watertight = None

        self.scale: torch.Tensor = _as_vec3(scale, device=resolved_device)
        self._vertices_tensor: torch.Tensor = vertices_tensor
        self._faces_tensor: torch.Tensor = faces_tensor
        self.recenter: bool = bool(recenter)
        self.fill_mode: str = normalized_fill_mode
        self.topology_diagnostics: bool = diagnostics_enabled
        self.surface_thickness = surface_thickness
        self.source_path: str | None = None if source_path is None else str(Path(source_path).expanduser().resolve())
        self.boundary_edge_count: int | None = boundary_edges
        self.non_manifold_edge_count: int | None = non_manifold_edges
        self.degenerate_face_count: int | None = degenerate_faces
        self.inconsistent_edge_orientation_count: int | None = (
            inconsistent_edge_orientations
        )
        self.signed_volume: float | None = signed_volume
        self.enclosed_volume: float | None = (
            None if signed_volume is None else float(abs(signed_volume))
        )
        self.is_watertight: bool | None = is_watertight
        self._topology_diagnostic_revision: tuple | None = (
            (
                _tensor_revision(vertices_tensor),
                _tensor_revision(faces_tensor),
            )
            if diagnostics_enabled and tensor_authored
            else None
        )
        self._voxel_cache: dict = {}
        self._sdf_cache: dict = {}

    @property
    def vertices(self) -> torch.Tensor:
        """Return the authored vertex tensor without conversion or materialization."""

        return self._vertices_tensor

    @property
    def faces(self) -> torch.Tensor:
        """Return the authored face tensor without conversion or materialization."""

        return self._faces_tensor

    def _local_vertices_tensor(self, *, device=None, dtype=None) -> torch.Tensor:
        vertices = self._vertices_tensor
        if self.recenter:
            bounds_min = vertices.min(dim=0).values
            bounds_max = vertices.max(dim=0).values
            vertices = vertices - 0.5 * (bounds_min + bounds_max)
        if device is not None or dtype is not None:
            vertices = vertices.to(
                device=vertices.device if device is None else device,
                dtype=vertices.dtype if dtype is None else dtype,
            )
        return vertices

    def _world_vertices(self, *, device=None, dtype=None) -> torch.Tensor:
        local_vertices = self._local_vertices_tensor(device=device, dtype=dtype)
        scale = self.scale.to(device=local_vertices.device, dtype=local_vertices.dtype)
        rotation_value = self.rotation.to(
            device=local_vertices.device, dtype=local_vertices.dtype
        )
        rotation = quat_to_rotation_matrix(
            _rotation_quaternion(rotation_value)
        )
        position = self.position.to(device=local_vertices.device, dtype=local_vertices.dtype)
        return (local_vertices * scale) @ rotation.T + position

    @property
    def world_vertices(self) -> torch.Tensor:
        return self._world_vertices()

    @property
    def bounds_world(self) -> torch.Tensor:
        world_vertices = self.world_vertices
        world_min = world_vertices.min(dim=0).values
        world_max = world_vertices.max(dim=0).values
        return torch.stack((world_min, world_max))

    @classmethod
    def from_obj(
        cls,
        path: str | PathLike[str],
        *,
        position=(0.0, 0.0, 0.0),
        scale=1.0,
        rotation=None,
        recenter: bool = True,
        fill_mode: str = "auto",
        surface_thickness: float | None = None,
        device=None,
    ) -> "Mesh":
        vertices, faces = _load_obj(path)
        return cls(
            vertices,
            faces,
            position=position,
            scale=scale,
            rotation=rotation,
            recenter=recenter,
            fill_mode=fill_mode,
            surface_thickness=surface_thickness,
            source_path=path,
            topology_diagnostics=True,
            device=device,
        )

    @property
    def vertex_count(self) -> int:
        return int(self._vertices_tensor.shape[0])

    @property
    def face_count(self) -> int:
        return int(self._faces_tensor.shape[0])

    def has_trainable_geometry(self) -> bool:
        continuous = (
            self.position,
            self.rotation,
            self.scale,
            self._vertices_tensor,
            self.surface_thickness,
        )
        return any(
            tensor.requires_grad
            for tensor in continuous
            if isinstance(tensor, torch.Tensor)
        )

    @staticmethod
    def _tensor_state_key(tensor: torch.Tensor) -> tuple:
        return (
            str(tensor.device),
            str(tensor.dtype),
            tuple(tensor.shape),
            tensor.data_ptr(),
            getattr(tensor, "_version", None),
            bool(tensor.requires_grad),
        )

    def geometry_state_key(self) -> tuple:
        return (
            self.recenter,
            self.fill_mode,
            (
                self._tensor_state_key(self.surface_thickness)
                if isinstance(self.surface_thickness, torch.Tensor)
                else self.surface_thickness
            ),
            self.is_watertight,
            self.degenerate_face_count,
            self.inconsistent_edge_orientation_count,
            self._tensor_state_key(self.position),
            self._tensor_state_key(self.rotation),
            self._tensor_state_key(self.scale),
            self._tensor_state_key(self._vertices_tensor),
            self._tensor_state_key(self._faces_tensor),
        )

    def clear_voxel_cache(self) -> None:
        self._voxel_cache.clear()

    def clear_sdf_cache(self) -> None:
        self._sdf_cache.clear()

    def _sdf_query_data(self, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict | None]:
        use_cache = not self.has_trainable_geometry()
        cache_key = (str(device), str(dtype))
        geometry_key = self.geometry_state_key()
        if use_cache:
            cached = self._sdf_cache.get(cache_key)
            if cached is not None and cached["geometry_key"] == geometry_key:
                return cached["vertices"], cached["faces"], cached["triangles"], cached.get("bvh")
        else:
            self._sdf_cache.clear()

        vertices = self._world_vertices(device=device, dtype=dtype)
        faces = self._faces_tensor.to(device=device)
        triangles = vertices[faces].contiguous()
        bvh = None
        if (
            use_cache
            and device.type == "cuda"
            and dtype == torch.float32
            and self.vertices.device.type == "cpu"
            and self.faces.device.type == "cpu"
            and _mesh_sdf_available()
        ):
            cpu_vertices = self._world_vertices(
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            cpu_faces = self.faces.to(dtype=torch.int64)
            cpu_triangles = cpu_vertices[cpu_faces].contiguous()
            bvh = _triangle_bvh_to_device(
                _build_triangle_bvh(cpu_triangles),
                device=device,
            )

        if use_cache:
            self._sdf_cache[cache_key] = {
                "geometry_key": geometry_key,
                "vertices": vertices,
                "faces": faces,
                "triangles": triangles,
                "bvh": bvh,
            }
        return vertices, faces, triangles, bvh

    def _surface_band(self, xx: torch.Tensor, yy: torch.Tensor, zz: torch.Tensor) -> torch.Tensor:
        spacings = []
        for coord, dim in ((xx, 0), (yy, 1), (zz, 2)):
            if coord.ndim > dim and coord.shape[dim] > 1:
                delta = torch.diff(coord, dim=dim).abs()
                positive = delta[delta > 0]
                if positive.numel() > 0:
                    spacings.append(positive.min())
        if self.surface_thickness is not None:
            return torch.as_tensor(self.surface_thickness, device=xx.device, dtype=xx.dtype)
        if not spacings:
            return torch.as_tensor(1.0e-3, device=xx.device, dtype=xx.dtype)
        spacing_vec = torch.stack([spacing.to(device=xx.device, dtype=xx.dtype) for spacing in spacings])
        return 0.25 * torch.min(spacing_vec)

    def signed_distance(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        if self._topology_diagnostic_revision is not None:
            current_revision = (
                _tensor_revision(self.vertices),
                _tensor_revision(self.faces),
            )
            if current_revision != self._topology_diagnostic_revision:
                raise RuntimeError(
                    "Mesh topology diagnostics are stale after an in-place "
                    "tensor mutation; reconstruct the Mesh before SDF queries."
                )
        if self.fill_mode == "auto" and self.is_watertight is None:
            raise RuntimeError(
                "Mesh topology is unknown; signed-distance queries require "
                "fill_mode='solid' or 'surface', or CPU tensor diagnostics."
            )
        if x.ndim == y.ndim == z.ndim == 1:
            xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        else:
            xx, yy, zz = x, y, z

        points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        vertices, faces, triangles, bvh = self._sdf_query_data(device=points.device, dtype=points.dtype)
        use_static_bvh = (
            bvh is not None
            and not self.has_trainable_geometry()
            and points.device.type == "cuda"
            and points.dtype == torch.float32
            and not points.requires_grad
        )
        if self.fill_mode == "surface" or (self.fill_mode == "auto" and not self.is_watertight):
            if use_static_bvh:
                unsigned_distance = triangle_mesh_unsigned_distance_static_bvh(points, triangles, bvh)
            else:
                unsigned_distance = triangle_mesh_unsigned_distance(points, vertices, faces, _triangles=triangles)
            signed_distance = unsigned_distance - self._surface_band(xx, yy, zz)
        else:
            if use_static_bvh:
                signed_distance = triangle_mesh_signed_distance_static_bvh(points, triangles, bvh)
            else:
                signed_distance = triangle_mesh_smooth_signed_distance(points, vertices, faces, _triangles=triangles)
        return signed_distance.reshape(xx.shape)

    def to_mesh(self, segments=16, *, device=None):
        del segments
        target_device = self.vertices.device if device is None else torch.device(device)
        vertices = self._world_vertices(device=target_device)
        faces = (
            self.faces
            if self.faces.device == target_device
            else self.faces.to(device=target_device)
        )
        return vertices, faces
