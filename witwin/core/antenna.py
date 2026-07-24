from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Callable
from numbers import Integral, Real
from typing import Any, Literal, Mapping

import torch

from .identity import AntennaId, reserve_antenna_id
from .scalars import ScalarLike


_PATTERN_KINDS = frozenset(
    {"isotropic", "vertical", "horizontal", "custom"}
)


@dataclass(frozen=True, slots=True)
class AntennaPattern:
    """Solver-neutral antenna pattern identity and custom field callable."""

    kind: str = "isotropic"
    custom: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.kind not in _PATTERN_KINDS:
            raise ValueError(
                f"pattern kind must be one of {sorted(_PATTERN_KINDS)}."
            )
        if (self.kind == "custom") != (self.custom is not None):
            raise ValueError(
                "custom pattern requires exactly one custom callable."
            )


def _tensor_leaf(
    value,
    *,
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    tensor = (
        value
        if isinstance(value, torch.Tensor)
        else torch.tensor(value, dtype=torch.float32)
    )
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if not tensor.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating-point dtype.")
    return tensor


def antenna_orientation_matrix(
    orientation: torch.Tensor | None,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return the differentiable local-to-world antenna rotation matrix.

    Euler orientation uses ``(yaw, pitch, roll)`` radians with intrinsic Z-Y-X
    rotations. Quaternion orientation uses scalar-first ``(w, x, y, z)``.
    """

    if orientation is None:
        return torch.eye(3, device=reference.device, dtype=reference.dtype)
    if orientation.shape == (4,):
        w, x, y, z = orientation.unbind()
        return torch.stack(
            (
                torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y))),
                torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x))),
                torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))),
            )
        )
    yaw, pitch, roll = orientation.unbind()
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    return torch.stack(
        (
            torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr)),
            torch.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr)),
            torch.stack((-sp, cp * sr, cp * cr)),
        )
    )


@dataclass(frozen=True, init=False)
class AntennaState:
    """Solver-neutral logical TX/RX and antenna-array state.

    ``orientation`` maps endpoint-local coordinates to world coordinates.
    Euler values are ``(yaw, pitch, roll)`` radians under intrinsic Z-Y-X;
    quaternions are unit ``(w, x, y, z)`` values. ``polarization`` and
    ``element_positions`` are authored in endpoint-local Cartesian coordinates.
    A solver projects them into world coordinates with ``orientation_matrix``.
    """

    antenna_id: AntennaId
    role: Literal["tx", "rx"]
    position: torch.Tensor
    orientation: torch.Tensor | None
    polarization: torch.Tensor | None
    element_positions: torch.Tensor | None
    weights: torch.Tensor | None
    synthetic_array: bool
    pattern: AntennaPattern
    power_w: ScalarLike | None
    metadata: Mapping[str, Any]

    def __init__(
        self,
        antenna_id: AntennaId | int,
        role: Literal["tx", "rx"],
        position,
        *,
        orientation=None,
        polarization=None,
        element_positions=None,
        weights=None,
        synthetic_array: bool = True,
        pattern: AntennaPattern | str = "isotropic",
        power_w: ScalarLike | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if role not in {"tx", "rx"}:
            raise ValueError("role must be 'tx' or 'rx'.")
        resolved_position = _tensor_leaf(
            position, name="position", shape=(3,)
        )
        if orientation is None:
            resolved_orientation = None
        else:
            resolved_orientation = (
                orientation
                if isinstance(orientation, torch.Tensor)
                else torch.tensor(orientation, dtype=torch.float32)
            )
            if resolved_orientation.shape not in {(3,), (4,)}:
                raise ValueError(
                    "orientation must have shape (3,) for Euler angles or "
                    "(4,) for a quaternion."
                )
            if not resolved_orientation.dtype.is_floating_point:
                raise TypeError("orientation must use a floating-point dtype.")
        resolved_polarization = (
            None
            if polarization is None
            else _tensor_leaf(polarization, name="polarization", shape=(3,))
        )
        if element_positions is not None:
            resolved_element_positions = (
                element_positions
                if isinstance(element_positions, torch.Tensor)
                else torch.tensor(element_positions, dtype=torch.float32)
            )
            if (
                resolved_element_positions.ndim != 2
                or resolved_element_positions.shape[1] != 3
            ):
                raise ValueError("element_positions must have shape (A, 3).")
            if not resolved_element_positions.dtype.is_floating_point:
                raise TypeError(
                    "element_positions must use a floating-point dtype."
                )
        else:
            resolved_element_positions = None
        if weights is not None:
            resolved_weights = (
                weights
                if isinstance(weights, torch.Tensor)
                else torch.tensor(weights, dtype=torch.float32)
            )
            if resolved_weights.ndim != 1:
                raise ValueError("weights must have shape (A,).")
            if not (
                resolved_weights.dtype.is_floating_point
                or resolved_weights.dtype.is_complex
            ):
                raise TypeError(
                    "weights must use a floating-point or complex dtype."
                )
            if (
                resolved_element_positions is not None
                and resolved_weights.shape[0]
                != resolved_element_positions.shape[0]
            ):
                raise ValueError(
                    "weights must have one value per antenna element."
                )
        else:
            resolved_weights = None
        continuous_tensors = tuple(
            value
            for value in (
                resolved_position,
                resolved_orientation,
                resolved_polarization,
                resolved_element_positions,
                resolved_weights,
                power_w,
            )
            if isinstance(value, torch.Tensor)
        )
        if any(
            value.device != continuous_tensors[0].device
            for value in continuous_tensors[1:]
        ):
            raise ValueError("Antenna tensor leaves must be on the same device.")
        if isinstance(power_w, torch.Tensor):
            if power_w.ndim != 0 or not power_w.dtype.is_floating_point:
                raise TypeError(
                    "power_w must be a scalar floating-point tensor."
                )

        object.__setattr__(
            self, "antenna_id", reserve_antenna_id(int(antenna_id))
        )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "position", resolved_position)
        object.__setattr__(self, "orientation", resolved_orientation)
        object.__setattr__(self, "polarization", resolved_polarization)
        object.__setattr__(
            self, "element_positions", resolved_element_positions
        )
        object.__setattr__(self, "weights", resolved_weights)
        object.__setattr__(self, "synthetic_array", bool(synthetic_array))
        resolved_pattern = (
            AntennaPattern(pattern) if isinstance(pattern, str) else pattern
        )
        if not isinstance(resolved_pattern, AntennaPattern):
            raise TypeError("pattern must be an AntennaPattern or string.")
        object.__setattr__(self, "pattern", resolved_pattern)
        object.__setattr__(self, "power_w", power_w)
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(metadata or {}))
        )

    def orientation_matrix(self) -> torch.Tensor:
        """Return the local-to-world rotation without detaching tensor leaves."""

        return antenna_orientation_matrix(
            self.orientation,
            reference=self.position,
        )

    def polarization_world(self) -> torch.Tensor | None:
        """Return the world-frame polarization while preserving autograd."""

        if self.polarization is None:
            return None
        rotation = self.orientation_matrix().to(
            device=self.polarization.device,
            dtype=self.polarization.dtype,
        )
        return rotation @ self.polarization


@dataclass(frozen=True, init=False)
class ReceiverGrid(AntennaState):
    """A row-major planar receiver grid with one stable RX endpoint identity.

    Sample ``(row, column)`` is ``origin + row * spacing[0] * x_axis
    + column * spacing[1] * y_axis``. Axes are world-frame directions and are
    intentionally independent of the antenna-local orientation.
    """

    x_axis: torch.Tensor
    y_axis: torch.Tensor
    shape: tuple[int, int]
    spacing: torch.Tensor

    def __init__(
        self,
        antenna_id: AntennaId | int,
        origin,
        x_axis,
        y_axis,
        shape,
        spacing,
        *,
        orientation=None,
        polarization=None,
        element_positions=None,
        weights=None,
        synthetic_array: bool = True,
        pattern: AntennaPattern | str = "isotropic",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            antenna_id,
            "rx",
            origin,
            orientation=orientation,
            polarization=polarization,
            element_positions=element_positions,
            weights=weights,
            synthetic_array=synthetic_array,
            pattern=pattern,
            metadata=metadata,
        )
        resolved_x_axis = (
            x_axis
            if isinstance(x_axis, torch.Tensor)
            else torch.tensor(
                x_axis,
                device=self.position.device,
                dtype=self.position.dtype,
            )
        )
        resolved_y_axis = (
            y_axis
            if isinstance(y_axis, torch.Tensor)
            else torch.tensor(
                y_axis,
                device=self.position.device,
                dtype=self.position.dtype,
            )
        )
        for name, axis in (
            ("x_axis", resolved_x_axis),
            ("y_axis", resolved_y_axis),
        ):
            if axis.shape != (3,):
                raise ValueError(f"{name} must have shape (3,).")
            if not axis.dtype.is_floating_point:
                raise TypeError(f"{name} must use a floating-point dtype.")
        if isinstance(spacing, torch.Tensor):
            resolved_spacing = spacing
        else:
            if (
                len(spacing) != 2
                or any(not isinstance(value, Real) for value in spacing)
                or any(float(value) <= 0.0 for value in spacing)
            ):
                raise ValueError("spacing must contain two positive values.")
            resolved_spacing = torch.tensor(
                spacing,
                device=self.position.device,
                dtype=self.position.dtype,
            )
        if resolved_spacing.shape != (2,):
            raise ValueError("spacing must have shape (2,).")
        if not resolved_spacing.dtype.is_floating_point:
            raise TypeError("spacing must use a floating-point dtype.")
        resolved_shape = tuple(shape)
        if (
            len(resolved_shape) != 2
            or any(
                not isinstance(value, Integral)
                or isinstance(value, bool)
                or int(value) <= 0
                for value in resolved_shape
            )
        ):
            raise ValueError("shape must contain two positive integers.")
        continuous = (
            self.position,
            resolved_x_axis,
            resolved_y_axis,
            resolved_spacing,
        )
        if any(value.device != continuous[0].device for value in continuous[1:]):
            raise ValueError("ReceiverGrid tensor leaves must be on the same device.")
        object.__setattr__(self, "x_axis", resolved_x_axis)
        object.__setattr__(self, "y_axis", resolved_y_axis)
        object.__setattr__(
            self, "shape", (int(resolved_shape[0]), int(resolved_shape[1]))
        )
        object.__setattr__(self, "spacing", resolved_spacing)

    @property
    def origin(self) -> torch.Tensor:
        """Return the exact tensor used as the inherited endpoint position."""

        return self.position

    def points(self) -> torch.Tensor:
        """Return row-major ``(rows * columns, 3)`` world sample positions."""

        rows, columns = self.shape
        row = torch.arange(
            rows, device=self.origin.device, dtype=self.origin.dtype
        )
        column = torch.arange(
            columns, device=self.origin.device, dtype=self.origin.dtype
        )
        row_grid, column_grid = torch.meshgrid(row, column, indexing="ij")
        spacing = self.spacing.to(
            device=self.origin.device, dtype=self.origin.dtype
        )
        return (
            self.origin
            + row_grid.reshape(-1, 1) * spacing[0] * self.x_axis
            + column_grid.reshape(-1, 1) * spacing[1] * self.y_axis
        )


__all__ = [
    "AntennaPattern",
    "AntennaState",
    "ReceiverGrid",
    "antenna_orientation_matrix",
]
