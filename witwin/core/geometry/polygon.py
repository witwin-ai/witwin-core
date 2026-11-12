"""Shared differentiable polygon distance helpers."""

from __future__ import annotations

import torch


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
    # tiny is only a zero-length-edge guard; eps would clamp real sub-millimetre float32 edges.
    edge_norm_sq = torch.sum(edges**2, dim=-1).clamp_min(torch.finfo(points.dtype).tiny)
    t = torch.clamp(
        torch.sum(rel * edges[None, :, :], dim=-1) / edge_norm_sq[None, :], 0.0, 1.0
    )
    closest = polygon[None, :, :] + t[..., None] * edges[None, :, :]
    return (
        torch.sum((points[:, None, :] - closest) ** 2, dim=-1)
        .min(dim=1)
        .values.clamp_min(0.0)
    )


def convex_polygon_signed_distance_2d(
    points: torch.Tensor, polygon: torch.Tensor
) -> torch.Tensor:
    next_polygon = torch.roll(polygon, shifts=-1, dims=0)
    edges = next_polygon - polygon
    rel = points[:, None, :] - polygon[None, :, :]
    dist = torch.sqrt(
        _polygon_edge_distance_sq_2d(points, polygon, edges=edges, rel=rel)
    )

    polygon_area = 0.5 * torch.sum(_cross_2d(polygon, next_polygon))
    crosses = _cross_2d(edges[None, :, :], rel)
    # cross_k = |e_k| * (signed distance to edge line k), so the inside tolerance is a length
    # (1e-7 of the polygon diameter) times |e_k|: scale-free, unlike an absolute cross bound.
    detached = polygon.detach()
    diameter = (detached.amax(dim=0) - detached.amin(dim=0)).max()
    tolerance = 1.0e-7 * diameter * torch.linalg.vector_norm(edges.detach(), dim=-1)
    if polygon_area.detach().item() >= 0.0:
        inside = torch.all(crosses >= -tolerance[None, :], dim=1)
    else:
        inside = torch.all(crosses <= tolerance[None, :], dim=1)
    return torch.where(inside, -dist, dist)


def _polygon_crossing_count_2d(
    points: torch.Tensor, polygon: torch.Tensor
) -> torch.Tensor:
    """Count +u horizontal-ray crossings of a closed (M, 2) polygon loop for each (P, 2) point."""
    start = polygon
    end = torch.roll(polygon, shifts=-1, dims=0)
    pu = points[:, 0:1]
    pv = points[:, 1:2]
    straddles = (start[None, :, 1] <= pv) != (end[None, :, 1] <= pv)
    denom = _safe_signed_denom(end[:, 1] - start[:, 1], torch.finfo(points.dtype).tiny)
    t = (pv - start[None, :, 1]) / denom[None, :]
    crossing_u = start[None, :, 0] + t * (end[:, 0] - start[:, 0])[None, :]
    return (straddles & (pu < crossing_u)).sum(dim=1)


def polygon_loops_signed_distance_2d(points: torch.Tensor, loops) -> torch.Tensor:
    """Even-odd signed distance from (P, 2) points to one or more closed 2D polygon loops.

    Negative inside; the interior is defined by the even-odd rule over the combined
    crossing count of all loops, so winding direction is irrelevant and additional
    loops carve holes. Valid for arbitrary (including non-convex and self-intersecting)
    polygons.
    """
    distance_sq = torch.stack(
        [_polygon_edge_distance_sq_2d(points, loop) for loop in loops], dim=0
    )
    distance = torch.sqrt(distance_sq.min(dim=0).values)
    crossings = sum(_polygon_crossing_count_2d(points, loop) for loop in loops)
    inside = crossings % 2 == 1
    return torch.where(inside, -distance, distance)


def simple_polygon_orientation_2d(polygon) -> float:
    """Signed area of a simple (N, 2) host polygon; raise unless it is simple.

    Simple means at least three finite vertices, no repeated consecutive vertex
    (including the closing wrap), nonzero area, and no two edges that meet other
    than adjacent edges at their shared vertex. Tests run in fp64 on coordinates
    normalized by the bounding-box extent, so the result is scale-free.
    """
    import numpy as np

    points = np.asarray(polygon, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("polygon vertices must have shape (N, 2) with N >= 3.")
    if not np.isfinite(points).all():
        raise ValueError("polygon vertices must be finite.")
    extent = float(np.ptp(points, axis=0).max())
    if extent == 0.0:
        raise ValueError("polygon vertices must not all coincide.")
    p = (points - points.min(axis=0)) / extent
    q = np.roll(p, -1, axis=0)
    if (np.hypot(*(q - p).T) == 0.0).any():
        raise ValueError("polygon must not repeat consecutive vertices (including the closing vertex).")
    area = 0.5 * float(np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]))
    if abs(area) <= 1.0e-12:
        raise ValueError("polygon must have nonzero area.")
    count = p.shape[0]
    d = q - p

    def orient(a, b, c):
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])

    tol = 1.0e-12
    i, j = np.triu_indices(count, k=1)
    adjacent = (j == i + 1) | ((i == 0) & (j == count - 1))
    a0, a1, b0, b1 = p[i], q[i], p[j], q[j]
    o1, o2 = orient(a0, a1, b0), orient(a0, a1, b1)
    o3, o4 = orient(b0, b1, a0), orient(b0, b1, a1)
    proper = (o1 * o2 < -tol * tol) & (o3 * o4 < -tol * tol)

    def on_segment(s0, s1, r, o):
        lo, hi = np.minimum(s0, s1) - tol, np.maximum(s0, s1) + tol
        return (np.abs(o) <= tol) & (r >= lo).all(-1) & (r <= hi).all(-1)

    # Adjacent edges share exactly one vertex; any other contact is a self-touch.
    touch_b0 = on_segment(a0, a1, b0, o1)
    touch_b1 = on_segment(a0, a1, b1, o2)
    touch_a0 = on_segment(b0, b1, a0, o3)
    touch_a1 = on_segment(b0, b1, a1, o4)
    nonadjacent_hit = ~adjacent & (proper | touch_b0 | touch_b1 | touch_a0 | touch_a1)
    # Adjacent pair (i, i+1) shares q[i] == p[j]; a fold back over the edge also touches.
    forward = adjacent & (j == i + 1)
    wrap = adjacent & ~forward
    folded = (forward & (touch_b1 | touch_a0)) | (wrap & (touch_b0 | touch_a1))
    collinear_back = adjacent & (np.abs(d[i, 0] * d[j, 1] - d[i, 1] * d[j, 0]) <= tol) & (
        np.sum(d[i] * d[j], axis=-1) < 0.0)
    if (nonadjacent_hit | folded | collinear_back).any():
        raise ValueError("polygon must be simple: edges may meet only at shared adjacent vertices.")
    return area * extent * extent


def ear_clip_triangles_2d(polygon) -> list[tuple[int, int, int]]:
    """Counter-clockwise index triangles of a simple counter-clockwise (N, 2) host polygon.

    Zero-area (collinear) apexes are never clipped, so every triangle has positive
    area and every input vertex is used. Coordinates are normalized to unit extent.
    """
    import numpy as np

    points = np.asarray(polygon, dtype=np.float64)
    points = (points - points.min(axis=0)) / float(np.ptp(points, axis=0).max())

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    eps = 1.0e-14
    remaining = list(range(points.shape[0]))
    triangles = []
    while len(remaining) > 3:
        for slot in range(len(remaining)):
            i0, i1, i2 = remaining[slot - 1], remaining[slot], remaining[(slot + 1) % len(remaining)]
            a, b, c = points[i0], points[i1], points[i2]
            if cross(a, b, c) <= eps:
                continue
            if any(cross(a, b, points[k]) >= -eps and cross(b, c, points[k]) >= -eps
                   and cross(c, a, points[k]) >= -eps
                   for k in remaining if k not in (i0, i1, i2)):
                continue
            triangles.append((i0, i1, i2))
            remaining.pop(slot)
            break
        else:
            raise ValueError("Failed to triangulate polygon; vertices must describe a simple polygon.")
    a, b, c = (points[k] for k in remaining)
    if cross(a, b, c) <= eps:
        raise ValueError("Failed to triangulate polygon; vertices must describe a simple polygon.")
    triangles.append(tuple(remaining))
    return triangles
