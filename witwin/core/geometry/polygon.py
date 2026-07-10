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
    edge_norm_sq = torch.sum(edges**2, dim=-1).clamp_min(torch.finfo(points.dtype).eps)
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
    if polygon_area.detach().item() >= 0.0:
        inside = torch.all(crosses >= -1.0e-7, dim=1)
    else:
        inside = torch.all(crosses <= 1.0e-7, dim=1)
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
    denom = _safe_signed_denom(end[:, 1] - start[:, 1], torch.finfo(points.dtype).eps)
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
