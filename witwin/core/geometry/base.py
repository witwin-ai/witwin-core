"""Shared geometry base classes and tensor helpers."""

from __future__ import annotations

import warnings
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from ..math import (
    quat_from_euler,
    quat_identity,
    quat_to_rotation_matrix,
    quat_to_rotation_matrix_np,
)


def _devices_match(left: torch.device, right: torch.device) -> bool:
    if left.type != right.type:
        return False
    return (
        left.index is None
        or right.index is None
        or left.index == right.index
    )


def _tensor_leaves(value):
    """Inspect authored containers without converting tensor values."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_leaves(item)
    elif isinstance(value, np.ndarray) and value.dtype == object:
        for item in value.flat:
            yield from _tensor_leaves(item)


def _resolve_device(explicit, *values) -> torch.device | None:
    tensors = (tensor for value in values for tensor in _tensor_leaves(value))
    resolved = torch.device(explicit) if explicit is not None else None
    for tensor in tensors:
        if resolved is None:
            resolved = tensor.device
            continue
        if not _devices_match(resolved, tensor.device):
            raise ValueError(
                f"Geometry tensor device {tensor.device} conflicts with "
                f"resolved device {resolved}."
            )
        if resolved.index is None:
            resolved = tensor.device
    return resolved



def _stack_scalar_sequence(value, *, device=None) -> torch.Tensor | None:
    """Preserve tensor ancestry, promoted dtype, and authored device in vectors.

    Python-only inputs keep their existing float32 conversion. When tensors
    are present, Python constants use the promoted tensor dtype, as if authored
    with ``tensor.new_tensor`` before ``torch.stack``.
    """
    tensors = tuple(_tensor_leaves(value))
    if not tensors:
        return None
    resolved_device = _resolve_device(device, value)
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    components = []
    for item in value:
        if isinstance(item, torch.Tensor):
            if item.shape != ():
                raise ValueError("Geometry sequence tensor components must be scalar.")
            components.append(item.to(dtype=dtype))
        elif tuple(_tensor_leaves(item)):
            raise ValueError("Geometry sequence tensor components must be scalar.")
        else:
            component = torch.as_tensor(item, dtype=dtype, device=resolved_device)
            if component.shape != ():
                raise ValueError("Geometry sequence components must be scalar.")
            components.append(component)
    return torch.stack(components)


def _as_position(value, *, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        _resolve_device(device, value)
        if value.shape != (3,):
            raise ValueError("position must have shape (3,).")
        return value
    seq = list(value)
    stacked = _stack_scalar_sequence(seq, device=device)
    if stacked is not None:
        if stacked.shape != (3,):
            raise ValueError("vector sequence must contain three scalar components.")
        return stacked
    return torch.tensor([float(v) for v in seq], dtype=torch.float32, device=device)


def _as_rotation(value, *, device=None) -> torch.Tensor:
    """Accept quaternion (4,), Euler tuple (3,), or None -> identity."""
    if value is None:
        return quat_identity(device=device)
    if isinstance(value, torch.Tensor):
        _resolve_device(device, value)
        if value.shape in {(3,), (4,)}:
            return value
        raise ValueError(f"rotation tensor must be shape (4,) quaternion or (3,) Euler, got {value.shape}")
    seq = list(value)
    stacked = _stack_scalar_sequence(seq, device=device)
    if stacked is not None:
        if stacked.shape not in {(3,), (4,)}:
            raise ValueError("rotation sequence must contain three or four scalar components.")
        return stacked
    if len(seq) == 4:
        return torch.tensor([float(v) for v in seq], dtype=torch.float32, device=device)
    if len(seq) == 3:
        return quat_from_euler(float(seq[0]), float(seq[1]), float(seq[2]), device=device)
    raise ValueError(f"rotation must have 3 (Euler) or 4 (quaternion) elements, got {len(seq)}")


def _as_scalar(value, *, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        _resolve_device(device, value)
        if value.shape != ():
            raise ValueError("scalar tensor must have shape ().")
        return value
    return torch.tensor(float(value), dtype=torch.float32, device=device)


def _as_vec3(value, *, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        _resolve_device(device, value)
        if value.shape != (3,):
            raise ValueError("vector tensor must have shape (3,).")
        return value
    if isinstance(value, (int, float)):
        scalar = float(value)
        return torch.tensor([scalar, scalar, scalar], dtype=torch.float32, device=device)
    seq = list(value)
    stacked = _stack_scalar_sequence(seq, device=device)
    if stacked is not None:
        if stacked.shape != (3,):
            raise ValueError("vector sequence must contain three scalar components.")
        return stacked
    return torch.tensor([float(v) for v in seq], dtype=torch.float32, device=device)


def _rotation_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape == (4,):
        return rotation
    return quat_from_euler(
        rotation[0],
        rotation[1],
        rotation[2],
        device=rotation.device,
        dtype=rotation.dtype,
    )


def _rotate_coords(xx, yy, zz, position: torch.Tensor, rotation: torch.Tensor):
    """Translate + inverse-rotate 3D grid coordinates into local frame."""
    dx = xx - position[0]
    dy = yy - position[1]
    dz = zz - position[2]
    rotation_matrix = quat_to_rotation_matrix(_rotation_quaternion(rotation))
    rotation_inverse = rotation_matrix.T
    dx_rot = rotation_inverse[0, 0] * dx + rotation_inverse[0, 1] * dy + rotation_inverse[0, 2] * dz
    dy_rot = rotation_inverse[1, 0] * dx + rotation_inverse[1, 1] * dy + rotation_inverse[1, 2] * dz
    dz_rot = rotation_inverse[2, 0] * dx + rotation_inverse[2, 1] * dy + rotation_inverse[2, 2] * dz
    return dx_rot, dy_rot, dz_rot


def _axial_split(dx, dy, dz, axis: str):
    """Return (axial, radial_a, radial_b) for a named axis."""
    if axis == "z":
        return dz, dx, dy
    if axis == "y":
        return dy, dx, dz
    return dx, dy, dz


def _apply_rotation_np(vertices: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Apply a (3,3) rotation matrix to (N,3) vertices."""
    return (vertices @ rotation_matrix.T).astype(np.float32)


def _constant_tensor(data, *, device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, device=device)


def _remap_axis_torch(vertices: torch.Tensor, axis: str) -> torch.Tensor:
    """Remap z-up vertices to the axial/radial order used by _axial_split."""
    if axis == "y":
        return vertices[:, [0, 2, 1]]
    if axis == "x":
        return vertices[:, [2, 0, 1]]
    return vertices


def _coordinate_spacing(coord: torch.Tensor, dim: int) -> torch.Tensor | None:
    if not isinstance(coord, torch.Tensor) or coord.ndim <= dim or coord.shape[dim] <= 1:
        return None
    delta = torch.diff(coord, dim=dim).abs()
    positive = delta[delta > 0]
    if positive.numel() == 0:
        return None
    return positive.min()


def _default_beta_from_coords(xx, yy, zz, *, reference: torch.Tensor) -> torch.Tensor:
    spacings = [
        spacing
        for spacing in (
            _coordinate_spacing(xx, 0),
            _coordinate_spacing(yy, 1),
            _coordinate_spacing(zz, 2),
        )
        if spacing is not None
    ]
    if not spacings:
        raise ValueError(
            "Cannot infer beta without coordinate spacing; pass an explicit "
            "beta (a smoothing length in metres) for point-like queries."
        )

    spacing = spacings[0].to(device=reference.device, dtype=reference.dtype)
    for candidate in spacings[1:]:
        spacing = torch.minimum(spacing, candidate.to(device=reference.device, dtype=reference.dtype))
    return spacing * 0.05


def _resolve_beta(beta, xx, yy, zz, *, reference: torch.Tensor) -> torch.Tensor:
    if beta is None:
        resolved = _default_beta_from_coords(xx, yy, zz, reference=reference)
    else:
        resolved = torch.as_tensor(beta, device=reference.device, dtype=reference.dtype)
    # Beta is a length: dimensionless eps would broaden positive, small-scale
    # smoothing widths. tiny is only a divide-by-zero guard.
    return torch.clamp(resolved, min=torch.finfo(reference.dtype).tiny)


def occupancy_from_signed_distance(signed_distance: torch.Tensor, *, xx, yy, zz, offset=0.0, beta=None) -> torch.Tensor:
    """Soft occupancy with offset and beta expressed as lengths in metres.

    With beta=None, use 5% of the smallest coordinate spacing; queries without
    spacing must supply beta. Below sqrt(finfo(dtype).tiny), use a hard step
    (0.5 on the surface) and warn. Its distance, offset and beta gradients are
    zero; downstream density gradients remain available. This threshold bounds
    the reciprocal width by 1/sqrt(tiny), leaving substantial headroom for
    upstream gradient amplification instead of overflowing near the surface.
    """
    offset_tensor = torch.as_tensor(offset, device=signed_distance.device, dtype=signed_distance.dtype)
    beta_tensor = _resolve_beta(beta, xx, yy, zz, reference=signed_distance)
    hard = beta_tensor < torch.finfo(signed_distance.dtype).tiny ** 0.5
    use_hard = bool(hard.any())
    if use_hard:
        warnings.warn(
            "beta below sqrt(finfo(dtype).tiny): using hard occupancy with zero "
            "distance, offset and beta gradients to avoid gradient overflow.",
            RuntimeWarning,
            stacklevel=2,
        )
        # Do not evaluate the tiny division even in an unselected where branch:
        # its backward can overflow and contaminate zero geometry gradients.
        beta_tensor = torch.where(hard, torch.ones_like(beta_tensor), beta_tensor)
    occupancy = 0.5 * (1.0 - torch.tanh((signed_distance - offset_tensor) / beta_tensor))
    if use_hard:
        step = 0.5 * (1.0 - torch.sign(signed_distance - offset_tensor))
        occupancy = torch.where(hard, step, occupancy)
    return occupancy.clamp(0.0, 1.0)


@runtime_checkable
class GeometrySpec(Protocol):
    """Minimal geometry contract consumed by solver-specific scenes."""

    kind: str

    def to_mesh(self, segments: int = 16) -> tuple:
        ...


class GeometryBase:
    """Base class for all geometry primitives."""

    kind: str = "base"

    def __init__(self, position=(0, 0, 0), rotation=None, *, device=None):
        resolved_device = _resolve_device(device, position, rotation)
        self.position: torch.Tensor = _as_position(
            position, device=resolved_device
        )
        self.rotation: torch.Tensor = _as_rotation(
            rotation, device=resolved_device
        )

    @property
    def device(self):
        return self.position.device

    def to_dict(self, arrays=None) -> dict:
        from .serialize import geometry_to_dict

        return geometry_to_dict(self, arrays=arrays)

    @classmethod
    def from_dict(cls, d, arrays=None, device=None):
        from .serialize import geometry_from_dict

        return geometry_from_dict(d, arrays=arrays, device=device)

    def _local_coords(self, xx, yy, zz):
        return _rotate_coords(xx, yy, zz, self.position, self.rotation)

    def _rotation_matrix_np(self) -> np.ndarray:
        return quat_to_rotation_matrix_np(_rotation_quaternion(self.rotation))

    @staticmethod
    def _validate_axis(axis: str) -> str:
        axis_name = str(axis).lower()
        if axis_name not in {"x", "y", "z"}:
            raise ValueError("axis must be 'x', 'y', or 'z'.")
        return axis_name

    def _apply_axis_transform(
        self, vertices: torch.Tensor, faces: torch.Tensor, axis: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The y remap swaps two coordinates; x is an even cyclic permutation.
        if axis == "y":
            faces = faces[:, [0, 2, 1]]
        return _remap_axis_torch(vertices, axis), faces

    def _transform_mesh_verts(self, vertices: torch.Tensor) -> torch.Tensor:
        if not isinstance(vertices, torch.Tensor):
            vertices = torch.as_tensor(vertices, dtype=torch.float32, device=self.device)
        rotation = self.rotation.to(device=vertices.device, dtype=vertices.dtype)
        rotation_matrix = quat_to_rotation_matrix(_rotation_quaternion(rotation))
        position = self.position.to(device=vertices.device, dtype=vertices.dtype)
        return vertices @ rotation_matrix.T + position

    def with_material(self, material, **kwargs):
        from ..material import Structure

        return Structure(geometry=self, material=material, **kwargs)

    def signed_distance(self, xx, yy, zz):
        raise NotImplementedError

    def to_mask(self, xx, yy, zz, offset=0.0, beta=None):
        """Occupancy with offset and beta expressed as lengths in metres.

        See :func:`occupancy_from_signed_distance` for coordinate spacing
        requirements and the near-zero-beta gradient guard.
        """
        signed_distance = self.signed_distance(xx, yy, zz)
        return occupancy_from_signed_distance(
            signed_distance,
            xx=xx,
            yy=yy,
            zz=zz,
            offset=offset,
            beta=beta,
        )

    def to_mesh(self, segments=16):
        raise NotImplementedError

    def __repr__(self):
        fields = ", ".join(f"{key}={value}" for key, value in self.__dict__.items() if not key.startswith("_"))
        return f"{type(self).__name__}({fields})"
