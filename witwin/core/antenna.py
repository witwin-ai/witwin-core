from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Callable
from typing import Any, Literal, Mapping

import torch

from .identity import AntennaId, reserve_antenna_id


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


@dataclass(frozen=True, init=False)
class AntennaState:
    """Solver-neutral logical TX/RX and antenna-array state."""

    antenna_id: AntennaId
    role: Literal["tx", "rx"]
    position: torch.Tensor
    orientation: torch.Tensor | None
    polarization: torch.Tensor | None
    element_positions: torch.Tensor | None
    weights: torch.Tensor | None
    synthetic_array: bool
    pattern: AntennaPattern
    power_w: Any | None
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
        power_w: Any | None = None,
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


__all__ = ["AntennaPattern", "AntennaState"]
