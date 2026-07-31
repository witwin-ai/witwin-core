"""Torch-native triangle-mesh distance helpers with optional solver acceleration."""

from __future__ import annotations

import math

import numpy as np
import torch


_MESH_SDF_ACCELERATOR: object | None = None
_MESH_SDF_ACCELERATOR_ERROR: str | None = None
_MESH_SDF_ACCELERATOR_FAILED = False
_MESH_SDF_BVH_LEAF_SIZE = 8
_MESH_SDF_BVH_MIN_TRIANGLES = 1024


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum(a * b, dim=-1)


def _safe_signed_denom(value: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.where(value >= 0.0, value.clamp_min(eps), value.clamp_max(-eps))


def _register_mesh_sdf_accelerator(accelerator: object) -> None:
    """Install a solver-owned mesh-SDF accelerator.

    Core remains independently usable through its PyTorch implementation.
    Solver packages may register an accelerator when they are imported.
    """
    global _MESH_SDF_ACCELERATOR, _MESH_SDF_ACCELERATOR_ERROR
    global _MESH_SDF_ACCELERATOR_FAILED
    _MESH_SDF_ACCELERATOR = accelerator
    _MESH_SDF_ACCELERATOR_ERROR = None
    _MESH_SDF_ACCELERATOR_FAILED = False


def _get_mesh_sdf_accelerator():
    global _MESH_SDF_ACCELERATOR_ERROR, _MESH_SDF_ACCELERATOR_FAILED
    accelerator = _MESH_SDF_ACCELERATOR
    if accelerator is None or _MESH_SDF_ACCELERATOR_FAILED:
        return None
    try:
        if not accelerator.is_available():
            return None
    except Exception as exc:  # noqa: BLE001 - retain the PyTorch path
        _MESH_SDF_ACCELERATOR_ERROR = f"{type(exc).__name__}: {exc}"
        _MESH_SDF_ACCELERATOR_FAILED = True
        return None
    return accelerator


def _mesh_sdf_available() -> bool:
    return _get_mesh_sdf_accelerator() is not None


def _select_mesh_sdf_accelerator(
    points: torch.Tensor,
    vertices: torch.Tensor,
):
    accelerator = _MESH_SDF_ACCELERATOR
    if accelerator is None or not accelerator.supports(points, vertices):
        return None
    return _get_mesh_sdf_accelerator()


def _mesh_sdf_build_error() -> str | None:
    """Return the last solver-accelerator loading error, if any."""
    return _MESH_SDF_ACCELERATOR_ERROR


def _indexed_triangles(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    if faces.device != vertices.device:
        faces = faces.to(device=vertices.device)
    return vertices[faces].contiguous()


def _build_triangle_bvh(
    triangles: torch.Tensor,
    *,
    leaf_size: int = _MESH_SDF_BVH_LEAF_SIZE,
    min_triangles: int = _MESH_SDF_BVH_MIN_TRIANGLES,
) -> dict[str, np.ndarray] | None:
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape (T, 3, 3).")
    if triangles.device.type != "cpu":
        raise ValueError(
            "_build_triangle_bvh accepts CPU-authored triangles only; "
            "CUDA geometry must use the direct device path."
        )
    triangle_count = int(triangles.shape[0])
    if triangle_count < int(min_triangles):
        return None

    triangles_np = triangles.detach().to(dtype=torch.float32).numpy()
    tri_min = triangles_np.min(axis=1)
    tri_max = triangles_np.max(axis=1)
    centroids = 0.5 * (tri_min + tri_max)

    bbox_min_list: list[np.ndarray] = []
    bbox_max_list: list[np.ndarray] = []
    left_list: list[int] = []
    right_list: list[int] = []
    start_list: list[int] = []
    count_list: list[int] = []
    ordered_triangles: list[int] = []

    def build(indices: np.ndarray) -> int:
        node_index = len(bbox_min_list)
        bbox_min_list.append(tri_min[indices].min(axis=0).astype(np.float32))
        bbox_max_list.append(tri_max[indices].max(axis=0).astype(np.float32))
        left_list.append(-1)
        right_list.append(-1)
        start_list.append(-1)
        count_list.append(0)

        if len(indices) <= int(leaf_size):
            start = len(ordered_triangles)
            ordered_triangles.extend(int(index) for index in indices.tolist())
            start_list[node_index] = start
            count_list[node_index] = len(indices)
            return node_index

        centroid_extent = centroids[indices].max(axis=0) - centroids[indices].min(axis=0)
        split_axis = int(np.argmax(centroid_extent))
        order = np.argsort(centroids[indices, split_axis], kind="mergesort")
        ordered_indices = indices[order]
        split = len(ordered_indices) // 2

        if split <= 0 or split >= len(ordered_indices):
            start = len(ordered_triangles)
            ordered_triangles.extend(int(index) for index in ordered_indices.tolist())
            start_list[node_index] = start
            count_list[node_index] = len(ordered_indices)
            return node_index

        left_child = build(ordered_indices[:split])
        right_child = build(ordered_indices[split:])
        left_list[node_index] = left_child
        right_list[node_index] = right_child
        return node_index

    build(np.arange(triangle_count, dtype=np.int32))
    return {
        "bbox_min": np.asarray(bbox_min_list, dtype=np.float32),
        "bbox_max": np.asarray(bbox_max_list, dtype=np.float32),
        "left": np.asarray(left_list, dtype=np.int32),
        "right": np.asarray(right_list, dtype=np.int32),
        "start": np.asarray(start_list, dtype=np.int32),
        "count": np.asarray(count_list, dtype=np.int32),
        "triangle_indices": np.asarray(ordered_triangles, dtype=np.int32),
    }


def _triangle_bvh_to_device(bvh: dict[str, np.ndarray] | None, *, device: torch.device) -> dict[str, torch.Tensor] | None:
    if bvh is None:
        return None
    return {
        "bbox_min": torch.from_numpy(bvh["bbox_min"]).to(device=device, dtype=torch.float32).contiguous(),
        "bbox_max": torch.from_numpy(bvh["bbox_max"]).to(device=device, dtype=torch.float32).contiguous(),
        "left": torch.from_numpy(bvh["left"]).to(device=device, dtype=torch.int32).contiguous(),
        "right": torch.from_numpy(bvh["right"]).to(device=device, dtype=torch.int32).contiguous(),
        "start": torch.from_numpy(bvh["start"]).to(device=device, dtype=torch.int32).contiguous(),
        "count": torch.from_numpy(bvh["count"]).to(device=device, dtype=torch.int32).contiguous(),
        "triangle_indices": torch.from_numpy(bvh["triangle_indices"]).to(device=device, dtype=torch.int32).contiguous(),
    }


def _signed_distance_from_unsigned_and_winding(
    unsigned_distance: torch.Tensor,
    winding_angle: torch.Tensor,
    winding_beta: float,
) -> torch.Tensor:
    angle_threshold = unsigned_distance.new_tensor(2.0 * math.pi)
    beta = unsigned_distance.new_tensor(float(winding_beta))
    sign_factor = -torch.tanh((torch.abs(winding_angle) - angle_threshold) / beta)
    return unsigned_distance * sign_factor


def _point_triangle_squared_distance(points: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(points.dtype).eps
    ab = b - a
    ac = c - a
    bc = c - b

    ap = points - a
    bp = points - b
    cp = points - c

    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    denom = (va + vb + vc).clamp_min(eps)
    v = vb / denom
    w = vc / denom
    face_proj = a + ab * v.unsqueeze(-1) + ac * w.unsqueeze(-1)
    dist2 = _dot(points - face_proj, points - face_proj)

    vertex_a = (d1 <= 0.0) & (d2 <= 0.0)
    vertex_b = (d3 >= 0.0) & (d4 <= d3)
    vertex_c = (d6 >= 0.0) & (d5 <= d6)

    edge_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    edge_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    edge_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)

    ab_t = (d1 / _safe_signed_denom(d1 - d3, eps)).clamp(0.0, 1.0)
    ac_t = (d2 / _safe_signed_denom(d2 - d6, eps)).clamp(0.0, 1.0)
    bc_t = ((d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(eps)).clamp(0.0, 1.0)

    proj_ab = a + ab * ab_t.unsqueeze(-1)
    proj_ac = a + ac * ac_t.unsqueeze(-1)
    proj_bc = b + bc * bc_t.unsqueeze(-1)

    dist2 = torch.where(vertex_a, _dot(ap, ap), dist2)
    dist2 = torch.where(vertex_b, _dot(bp, bp), dist2)
    dist2 = torch.where(vertex_c, _dot(cp, cp), dist2)
    dist2 = torch.where(edge_ab, _dot(points - proj_ab, points - proj_ab), dist2)
    dist2 = torch.where(edge_ac, _dot(points - proj_ac, points - proj_ac), dist2)
    dist2 = torch.where(edge_bc, _dot(points - proj_bc, points - proj_bc), dist2)
    return dist2


def _triangle_mesh_unsigned_distance_torch_from_triangles(
    points: torch.Tensor,
    triangles: torch.Tensor,
    *,
    point_chunk_size: int | None = None,
    triangle_chunk_size: int | None = None,
) -> torch.Tensor:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")

    point_chunk = 8192 if point_chunk_size is None else int(point_chunk_size)
    triangle_chunk = 256 if triangle_chunk_size is None else int(triangle_chunk_size)
    outputs: list[torch.Tensor] = []
    smooth_eps = 1.0e-12

    for point_start in range(0, points.shape[0], point_chunk):
        point_stop = min(points.shape[0], point_start + point_chunk)
        point_block = points[point_start:point_stop]
        point_block_expanded = point_block[:, None, :]
        min_dist2 = None

        for triangle_start in range(0, triangles.shape[0], triangle_chunk):
            triangle_stop = min(triangles.shape[0], triangle_start + triangle_chunk)
            triangle_block = triangles[triangle_start:triangle_stop]
            a = triangle_block[None, :, 0, :]
            b = triangle_block[None, :, 1, :]
            c = triangle_block[None, :, 2, :]
            dist2 = _point_triangle_squared_distance(point_block_expanded, a, b, c)
            block_min = dist2.min(dim=1).values
            min_dist2 = block_min if min_dist2 is None else torch.minimum(min_dist2, block_min)

        outputs.append(torch.sqrt(min_dist2.clamp_min(0.0) + smooth_eps) - math.sqrt(smooth_eps))

    return torch.cat(outputs, dim=0)


def _triangle_mesh_winding_angle_torch_from_triangles(
    points: torch.Tensor,
    triangles: torch.Tensor,
    *,
    point_chunk_size: int | None = None,
    triangle_chunk_size: int | None = None,
) -> torch.Tensor:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")

    point_chunk = 4096 if point_chunk_size is None else int(point_chunk_size)
    triangle_chunk = 256 if triangle_chunk_size is None else int(triangle_chunk_size)
    outputs: list[torch.Tensor] = []
    eps = torch.finfo(points.dtype).eps

    for point_start in range(0, points.shape[0], point_chunk):
        point_stop = min(points.shape[0], point_start + point_chunk)
        point_block = points[point_start:point_stop]
        total_angle = point_block.new_zeros((point_block.shape[0],))
        point_block_expanded = point_block[:, None, :]

        for triangle_start in range(0, triangles.shape[0], triangle_chunk):
            triangle_stop = min(triangles.shape[0], triangle_start + triangle_chunk)
            triangle_block = triangles[triangle_start:triangle_stop]
            a = triangle_block[None, :, 0, :] - point_block_expanded
            b = triangle_block[None, :, 1, :] - point_block_expanded
            c = triangle_block[None, :, 2, :] - point_block_expanded

            la = torch.linalg.norm(a, dim=-1).clamp_min(eps)
            lb = torch.linalg.norm(b, dim=-1).clamp_min(eps)
            lc = torch.linalg.norm(c, dim=-1).clamp_min(eps)
            numerator = _dot(a, torch.cross(b, c, dim=-1))
            denominator = (
                la * lb * lc
                + _dot(a, b) * lc
                + _dot(b, c) * la
                + _dot(c, a) * lb
            )
            total_angle = total_angle + 2.0 * torch.atan2(numerator, _safe_signed_denom(denominator, eps)).sum(dim=1)

        outputs.append(total_angle)

    return torch.cat(outputs, dim=0)


def _triangle_mesh_unsigned_distance_torch(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    point_chunk_size: int | None = None,
    triangle_chunk_size: int | None = None,
    _triangles: torch.Tensor | None = None,
) -> torch.Tensor:
    triangles = _indexed_triangles(vertices, faces) if _triangles is None else _triangles
    return _triangle_mesh_unsigned_distance_torch_from_triangles(
        points,
        triangles,
        point_chunk_size=point_chunk_size,
        triangle_chunk_size=triangle_chunk_size,
    )


def _triangle_mesh_winding_angle_torch(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    point_chunk_size: int | None = None,
    triangle_chunk_size: int | None = None,
    _triangles: torch.Tensor | None = None,
) -> torch.Tensor:
    triangles = _indexed_triangles(vertices, faces) if _triangles is None else _triangles
    return _triangle_mesh_winding_angle_torch_from_triangles(
        points,
        triangles,
        point_chunk_size=point_chunk_size,
        triangle_chunk_size=triangle_chunk_size,
    )


def triangle_mesh_unsigned_distance_static_bvh(
    points: torch.Tensor,
    triangles: torch.Tensor,
    bvh: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    accelerator = _select_mesh_sdf_accelerator(points, triangles)
    if accelerator is not None:
        return accelerator.unsigned_distance_static_bvh(points, triangles, bvh)
    return _triangle_mesh_unsigned_distance_torch_from_triangles(points, triangles)


def triangle_mesh_signed_distance_static_bvh(
    points: torch.Tensor,
    triangles: torch.Tensor,
    bvh: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    accelerator = _select_mesh_sdf_accelerator(points, triangles)
    if accelerator is not None:
        return accelerator.signed_distance_static_bvh(points, triangles, bvh)
    unsigned_distance = _triangle_mesh_unsigned_distance_torch_from_triangles(
        points,
        triangles,
    )
    winding_angle = _triangle_mesh_winding_angle_torch_from_triangles(
        points,
        triangles,
    )
    return _signed_distance_from_unsigned_and_winding(
        unsigned_distance,
        winding_angle,
        winding_beta=0.5,
    )


def triangle_mesh_unsigned_distance(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    point_chunk_size: int | None = None,
    triangle_chunk_size: int | None = None,
    _triangles: torch.Tensor | None = None,
) -> torch.Tensor:
    triangles = _indexed_triangles(vertices, faces) if _triangles is None else _triangles
    accelerator = _select_mesh_sdf_accelerator(points, vertices)
    if accelerator is not None:
        return accelerator.unsigned_distance(points, triangles)

    return _triangle_mesh_unsigned_distance_torch_from_triangles(
        points,
        triangles,
        point_chunk_size=point_chunk_size,
        triangle_chunk_size=triangle_chunk_size,
    )


def triangle_mesh_winding_angle(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    point_chunk_size: int | None = None,
    triangle_chunk_size: int | None = None,
    _triangles: torch.Tensor | None = None,
) -> torch.Tensor:
    triangles = _indexed_triangles(vertices, faces) if _triangles is None else _triangles
    accelerator = _select_mesh_sdf_accelerator(points, vertices)
    if accelerator is not None:
        return accelerator.winding_angle(points, triangles)

    return _triangle_mesh_winding_angle_torch_from_triangles(
        points,
        triangles,
        point_chunk_size=point_chunk_size,
        triangle_chunk_size=triangle_chunk_size,
    )


def _triangle_mesh_smooth_signed_distance_torch(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    winding_beta: float = 0.5,
    point_chunk_size: int | None = None,
    triangle_chunk_size: int | None = None,
    _triangles: torch.Tensor | None = None,
) -> torch.Tensor:
    triangles = _indexed_triangles(vertices, faces) if _triangles is None else _triangles
    unsigned_distance = _triangle_mesh_unsigned_distance_torch_from_triangles(
        points,
        triangles,
        point_chunk_size=point_chunk_size,
        triangle_chunk_size=triangle_chunk_size,
    )
    winding_angle = _triangle_mesh_winding_angle_torch_from_triangles(
        points,
        triangles,
        point_chunk_size=point_chunk_size,
        triangle_chunk_size=triangle_chunk_size,
    )
    return _signed_distance_from_unsigned_and_winding(unsigned_distance, winding_angle, winding_beta)


def triangle_mesh_smooth_signed_distance(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    winding_beta: float = 0.5,
    point_chunk_size: int | None = None,
    triangle_chunk_size: int | None = None,
    _triangles: torch.Tensor | None = None,
) -> torch.Tensor:
    triangles = _indexed_triangles(vertices, faces) if _triangles is None else _triangles
    accelerator = _select_mesh_sdf_accelerator(points, vertices)
    if accelerator is not None:
        return accelerator.smooth_signed_distance(
            points,
            triangles,
            winding_beta=float(winding_beta),
        )

    return _triangle_mesh_smooth_signed_distance_torch(
        points,
        vertices,
        faces,
        winding_beta=winding_beta,
        point_chunk_size=point_chunk_size,
        triangle_chunk_size=triangle_chunk_size,
        _triangles=triangles,
    )
