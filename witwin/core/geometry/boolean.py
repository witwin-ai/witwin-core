"""Boolean (constructive solid geometry) composition of signed-distance geometry.

``Union``, ``Intersection`` and ``Difference`` combine child geometries through
their signed distance fields:

* sharp composition (``smoothness=0``) uses exact ``min`` / ``max``, so the
  zero level set is exactly the boolean of the child surfaces;
* smooth composition (``smoothness > 0`` or a tensor) replaces ``min`` / ``max``
  with the compact-support quadratic soft minimum
  ``smin(a, b; k) = mix(b, a, h) - k h (1 - h)``, ``h = clamp(1/2 + (b - a) / 2k)``.
  It equals the sharp result wherever the two operands differ by at least
  ``k`` (a length) and is C1 across the seam, which keeps shape gradients
  continuous near kinks.

Each node carries its own ``position`` / ``rotation`` frame with the same
semantics as the analytic primitives: children are authored in the node's local
frame, and the node maps that frame into the world by rotating about its local
origin and then translating by ``position``. The default identity frame means
children are given in world coordinates. A rigid frame preserves distances, so
the composed field stays a signed distance bound.

``Union`` also serves as the geometry group: a sharp union of any number of
children is the grouping container.

Surface meshes come from marching tetrahedra (Kuhn subdivision of a uniform
cube lattice) over the node's bounding box. This is a documented approximation:
no exact mesh boolean is performed, flat faces are recovered exactly but sharp
edges and corners are chamfered at the sampling scale. The resulting mesh is
closed, consistently outward oriented and welded, and its vertex positions are
differentiable in the child parameters through the interpolated field values.
"""

from __future__ import annotations

from functools import reduce
from itertools import permutations

import torch

from .base import (
    GeometryBase,
    _as_scalar,
    _resolve_device,
)


__all__ = ["BooleanGeometry", "Union", "Intersection", "Difference"]


def _smooth_min(a: torch.Tensor, b: torch.Tensor, k: torch.Tensor | None) -> torch.Tensor:
    if k is None:
        return torch.minimum(a, b)
    k = k.to(device=a.device, dtype=a.dtype).clamp(min=torch.finfo(a.dtype).eps)
    h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + h * (a - b) - k * h * (1.0 - h)


def _smooth_max(a: torch.Tensor, b: torch.Tensor, k: torch.Tensor | None) -> torch.Tensor:
    if k is None:
        return torch.maximum(a, b)
    return -_smooth_min(-a, -b, k)


def _geometry_tensors(geometry) -> list[torch.Tensor]:
    return [value for value in vars(geometry).values() if isinstance(value, torch.Tensor)]


def _geometry_bounds(geometry) -> torch.Tensor:
    """World axis-aligned bounds ``(2, 3)`` of a child geometry."""
    bounds = getattr(geometry, "bounds_world", None)
    if isinstance(bounds, torch.Tensor):
        return bounds
    vertices, _ = geometry.to_mesh()
    return torch.stack((vertices.amin(dim=0), vertices.amax(dim=0)))


# Kuhn subdivision of the unit cube into six tetrahedra sharing the main
# diagonal (0,0,0)-(1,1,1). Every cube uses the same split, so neighbouring
# cubes agree on their shared face diagonals and the lattice is conforming.
def _kuhn_tetrahedra() -> tuple[tuple[tuple[int, int, int], ...], ...]:
    tetrahedra = []
    for order in permutations(range(3)):
        corner = [0, 0, 0]
        path = [tuple(corner)]
        for axis in order:
            corner[axis] = 1
            path.append(tuple(corner))
        tetrahedra.append(tuple(path))
    return tuple(tetrahedra)


_KUHN_TETRAHEDRA = _kuhn_tetrahedra()
_CUBE_CORNERS = tuple((i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1))


def _marching_tetrahedra(signed_distance: torch.Tensor, origin: torch.Tensor, spacing: torch.Tensor):
    """Extract the closed zero level set of a node-sampled field.

    ``signed_distance`` has shape ``(nx + 1, ny + 1, nz + 1)``, negative inside.
    Returns local vertices ``(V, 3)`` interpolated along sign-changing lattice
    edges (differentiable in the field values) and outward faces ``(F, 3)``.
    """
    device = signed_distance.device
    dtype = signed_distance.dtype
    shape = signed_distance.shape
    nx, ny, nz = (int(size) - 1 for size in shape)
    stride_x = shape[1] * shape[2]
    stride_y = shape[2]
    node_count = shape[0] * shape[1] * shape[2]

    # Node values close to zero put several edge vertices next to one lattice
    # node and create sliver faces. Snap them 5% of a cell outside (a surface
    # shift of at most that much); the straight-through form keeps the gradient.
    snap = 5.0e-2 * spacing.detach().min()
    near_zero = signed_distance.detach().abs() < snap
    field = torch.where(near_zero, signed_distance + (snap - signed_distance).detach(), signed_distance)
    inside = field.detach() < 0

    corners = [inside[i:i + nx, j:j + ny, k:k + nz] for i, j, k in _CUBE_CORNERS]
    any_inside = reduce(torch.logical_or, corners)
    all_inside = reduce(torch.logical_and, corners)
    cube_i, cube_j, cube_k = torch.nonzero(any_inside & ~all_inside, as_tuple=True)
    if cube_i.numel() == 0:
        raise ValueError("Boolean geometry has an empty surface at the sampled resolution.")

    base = cube_i * stride_x + cube_j * stride_y + cube_k
    offsets = torch.tensor(
        [[i * stride_x + j * stride_y + k for i, j, k in tet] for tet in _KUHN_TETRAHEDRA],
        device=device,
        dtype=torch.int64,
    )
    tets = (base[:, None, None] + offsets[None]).reshape(-1, 4)
    flat_inside = inside.reshape(-1)
    tet_inside = flat_inside[tets]
    count = tet_inside.sum(dim=1)
    active = (count > 0) & (count < 4)
    tets, tet_inside, count = tets[active], tet_inside[active], count[active]

    # Sort each tetrahedron so its inside nodes come first.
    order = torch.argsort((~tet_inside).to(torch.int8), dim=1, stable=True)
    nodes = torch.gather(tets, 1, order)
    a, b, c, d = nodes.unbind(dim=1)

    one, three, two = count == 1, count == 3, count == 2
    # Every triangle is three (inside node, outside node) lattice edges plus the
    # tetrahedron it came from, which decides its outward orientation.
    triangle_edges = torch.cat(
        (
            torch.stack((torch.stack((a, b), -1), torch.stack((a, c), -1), torch.stack((a, d), -1)), 1)[one],
            torch.stack((torch.stack((a, d), -1), torch.stack((b, d), -1), torch.stack((c, d), -1)), 1)[three],
            torch.stack((torch.stack((a, c), -1), torch.stack((a, d), -1), torch.stack((b, d), -1)), 1)[two],
            torch.stack((torch.stack((a, c), -1), torch.stack((b, d), -1), torch.stack((b, c), -1)), 1)[two],
        )
    )
    triangle_inside_count = torch.cat((count[one], count[three], count[two], count[two]))
    triangle_nodes = torch.cat((nodes[one], nodes[three], nodes[two], nodes[two]))

    keys = triangle_edges[..., 0] * node_count + triangle_edges[..., 1]
    unique_keys, faces = torch.unique(keys.reshape(-1), return_inverse=True)
    faces = faces.reshape(-1, 3)

    def node_positions(index: torch.Tensor) -> torch.Tensor:
        ijk = torch.stack((index // stride_x, (index // stride_y) % shape[1], index % shape[2]), dim=-1)
        return origin + ijk.to(dtype) * spacing

    inner = unique_keys // node_count
    outer = unique_keys % node_count
    flat_field = field.reshape(-1)
    inner_value = flat_field[inner]
    outer_value = flat_field[outer]
    weight = (inner_value / (inner_value - outer_value)).unsqueeze(-1)
    inner_position = node_positions(inner)
    vertices = inner_position + weight * (node_positions(outer) - inner_position)

    with torch.no_grad():
        corner_positions = node_positions(triangle_nodes)
        inside_rank = torch.arange(4, device=device)[None, :] < triangle_inside_count[:, None]
        inside_weight = inside_rank.to(dtype)
        outside_weight = (~inside_rank).to(dtype)
        inside_centroid = (corner_positions * inside_weight[..., None]).sum(1) / inside_weight.sum(1, keepdim=True)
        outside_centroid = (corner_positions * outside_weight[..., None]).sum(1) / outside_weight.sum(1, keepdim=True)
        triangle = vertices.detach()[faces]
        normal = torch.linalg.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0], dim=-1)
        flip = (normal * (outside_centroid - inside_centroid)).sum(-1) < 0
    faces = torch.where(flip[:, None], faces[:, [0, 2, 1]], faces)
    return vertices, faces


class BooleanGeometry(GeometryBase):
    """Shared implementation of signed-distance boolean composition.

    Subclasses define the composition rule and the composed local bounds. The
    continuous tensors of every child (position, size, radius, mesh vertices,
    nested boolean parameters, ...) are additionally exposed as private tensor
    attributes of the node, so consumers that discover a geometry's parameters
    from its tensor attributes (device placement, trainability checks, adjoint
    registration) see the full tree.

    ``smoothness`` is an absolute blend length in metres, not a relative factor.
    """

    kind = "boolean"
    _minimum_children = 1

    def __init__(self, *geometries, smoothness=0.0, position=(0, 0, 0), rotation=None, device=None):
        if len(geometries) == 1 and isinstance(geometries[0], (tuple, list)):
            geometries = tuple(geometries[0])
        if len(geometries) < self._minimum_children:
            raise ValueError(
                f"{type(self).__name__} requires at least {self._minimum_children} child geometries."
            )
        for geometry in geometries:
            if not callable(getattr(geometry, "signed_distance", None)) or not callable(
                getattr(geometry, "to_mesh", None)
            ):
                raise TypeError(
                    f"{type(self).__name__} children must provide signed_distance() and to_mesh()."
                )
        if not isinstance(smoothness, torch.Tensor) and float(smoothness) < 0.0:
            raise ValueError("smoothness must be non-negative.")

        child_tensors = []
        seen = set()
        for geometry in geometries:
            for tensor in _geometry_tensors(geometry):
                if id(tensor) not in seen:
                    seen.add(id(tensor))
                    child_tensors.append(tensor)
        resolved_device = _resolve_device(device, position, rotation, smoothness, child_tensors)
        super().__init__(position=position, rotation=rotation, device=resolved_device)
        self.geometries: tuple = tuple(geometries)
        self.smooth: bool = isinstance(smoothness, torch.Tensor) or float(smoothness) > 0.0
        self.smoothness: torch.Tensor = _as_scalar(smoothness, device=resolved_device)
        for index, tensor in enumerate(child_tensors):
            setattr(self, f"_child_tensor_{index}", tensor)

    def _blend(self) -> torch.Tensor | None:
        return self.smoothness if self.smooth else None

    def _compose(self, distances: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def _compose_local_bounds(self, bounds: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def has_trainable_geometry(self) -> bool:
        return any(
            value.requires_grad for value in vars(self).values() if isinstance(value, torch.Tensor)
        )

    def local_signed_distance(self, xx, yy, zz):
        """Composed signed distance at coordinates given in the node's local frame."""
        return self._compose([geometry.signed_distance(xx, yy, zz) for geometry in self.geometries])

    def signed_distance(self, xx, yy, zz):
        return self.local_signed_distance(*self._local_coords(xx, yy, zz))

    def _local_bounds(self) -> torch.Tensor:
        bounds = [_geometry_bounds(geometry) for geometry in self.geometries]
        dtype = reduce(torch.promote_types, (bound.dtype for bound in bounds))
        return self._compose_local_bounds([bound.to(device=self.device, dtype=dtype) for bound in bounds])

    @property
    def bounds_world(self) -> torch.Tensor:
        """World axis-aligned bounds ``(2, 3)`` (min row, max row) of the composed solid.

        Union bounds are the hull of the child bounds (grown by the smooth blend
        reach), intersection bounds the overlap of the child bounds, and
        difference bounds those of the base geometry. A rotated node returns the
        bounds of its rotated local box, which can be loose but always encloses
        the solid. Differentiable in the child and node parameters.
        """
        local = self._local_bounds()
        corners = torch.stack(
            [torch.stack([local[i, 0], local[j, 1], local[k, 2]]) for i, j, k in _CUBE_CORNERS]
        )
        world = self._transform_mesh_verts(corners)
        return torch.stack((world.amin(dim=0), world.amax(dim=0)))

    def to_mesh(self, segments=16, *, resolution=None):
        """Tessellate the composed surface by marching tetrahedra.

        ``resolution`` is the number of lattice cells across the longest side of
        the local bounding box (default ``4 * segments``). See the module
        docstring for the approximation this implies.
        """
        cells = int(4 * segments if resolution is None else resolution)
        if cells < 2:
            raise ValueError("resolution must be at least 2.")
        with torch.no_grad():
            local = self._local_bounds().detach()
        dtype = local.dtype if local.dtype.is_floating_point else torch.float32
        local = local.to(dtype=dtype)
        extent = (local[1] - local[0]).clamp(min=0.0)
        if not bool(torch.all(extent > 0)):
            raise ValueError(f"{type(self).__name__} has empty bounds and no surface.")
        step = extent.max() / cells
        # Three padding cells keep every lattice boundary node outside the solid,
        # and the half-step-odd placement keeps nodes off the bounding planes.
        counts = [int(torch.ceil(extent[axis] / step).item()) + 3 for axis in range(3)]
        spacing = torch.full((3,), float(step), device=local.device, dtype=dtype)
        center = 0.5 * (local[0] + local[1])
        origin = center - 0.5 * spacing * torch.tensor(counts, device=local.device, dtype=dtype)
        axes = [
            origin[axis] + spacing[axis] * torch.arange(counts[axis] + 1, device=local.device, dtype=dtype)
            for axis in range(3)
        ]
        xx, yy, zz = torch.meshgrid(*axes, indexing="ij")
        field = self.local_signed_distance(xx, yy, zz)
        local_vertices, faces = _marching_tetrahedra(field, origin, spacing)
        return self._transform_mesh_verts(local_vertices), faces


class Union(BooleanGeometry):
    """Union of child geometries; a sharp union is also the geometry group."""

    kind = "union"

    def _compose(self, distances):
        blend = self._blend()
        return reduce(lambda left, right: _smooth_min(left, right, blend), distances)

    def _compose_local_bounds(self, bounds):
        stacked = torch.stack(bounds)
        lower = stacked[:, 0].amin(dim=0)
        upper = stacked[:, 1].amax(dim=0)
        if self.smooth:
            # Each pairwise soft minimum lowers the field by at most k / 4.
            reach = 0.25 * (len(bounds) - 1) * self.smoothness.to(device=lower.device, dtype=lower.dtype)
            lower = lower - reach
            upper = upper + reach
        return torch.stack((lower, upper))


class Intersection(BooleanGeometry):
    """Intersection of child geometries."""

    kind = "intersection"

    def _compose(self, distances):
        blend = self._blend()
        return reduce(lambda left, right: _smooth_max(left, right, blend), distances)

    def _compose_local_bounds(self, bounds):
        stacked = torch.stack(bounds)
        lower = stacked[:, 0].amax(dim=0)
        upper = torch.maximum(stacked[:, 1].amin(dim=0), lower)
        return torch.stack((lower, upper))


class Difference(BooleanGeometry):
    """``base`` minus the union of ``tools``.

    ``smoothness`` is an absolute blend length in metres, not a relative factor.
    """

    kind = "difference"
    _minimum_children = 2

    def __init__(self, base, *tools, smoothness=0.0, position=(0, 0, 0), rotation=None, device=None):
        if len(tools) == 1 and isinstance(tools[0], (tuple, list)):
            tools = tuple(tools[0])
        super().__init__(
            base, *tools, smoothness=smoothness, position=position, rotation=rotation, device=device
        )

    @property
    def base(self):
        return self.geometries[0]

    @property
    def tools(self) -> tuple:
        return self.geometries[1:]

    def _compose(self, distances):
        blend = self._blend()
        removed = reduce(lambda left, right: _smooth_min(left, right, blend), distances[1:])
        return _smooth_max(distances[0], -removed, blend)

    def _compose_local_bounds(self, bounds):
        return bounds[0]
