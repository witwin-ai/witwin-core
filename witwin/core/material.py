from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from .geometry.base import GeometrySpec

VACUUM_PERMITTIVITY = 8.8541878128e-12


def _coerce_scalar(value: Any, *, name: str) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _numeric_scalar(value: Any) -> float | None:
    for candidate in (
        lambda item: float(item),
        lambda item: float(item.item()),
        lambda item: float(item[0]),
    ):
        try:
            return candidate(value)
        except (TypeError, ValueError, IndexError, KeyError, AttributeError):
            continue
    return None


def _coerce_nonnegative(value: Any, *, name: str) -> Any:
    scalar = _numeric_scalar(value)
    if scalar is not None and scalar < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _is_numeric_close(value: Any, target: float) -> bool:
    scalar = _numeric_scalar(value)
    if scalar is None:
        return False
    return bool(np.isclose(scalar, target))


@dataclass(frozen=True)
class MaterialCapabilities:
    conductive: bool = False
    magnetic: bool = False
    anisotropic: bool = False
    dispersive: bool = False


@dataclass(frozen=True)
class StaticMaterialSample:
    eps_r: Any
    mu_r: Any = 1.0
    sigma_e: Any = 0.0


@dataclass(frozen=True)
class FrequencyMaterialSample:
    eps_r: Any
    mu_r: Any = 1.0
    sigma_e: Any = 0.0


@runtime_checkable
class MaterialSpec(Protocol):
    name: str | None

    def capabilities(self) -> MaterialCapabilities:
        ...

    def evaluate_static(self) -> StaticMaterialSample:
        ...

    def evaluate_at_frequency(self, frequency: float) -> FrequencyMaterialSample:
        ...


@dataclass(frozen=True, init=False)
class Material(MaterialSpec):
    eps_r: Any
    mu_r: Any
    sigma_e: Any
    name: str | None

    def __init__(
        self,
        eps_r: Any = 1.0,
        mu_r: Any = 1.0,
        sigma_e: Any = 0.0,
        name: str | None = None,
    ):
        object.__setattr__(self, "eps_r", _coerce_scalar(eps_r, name="eps_r"))
        object.__setattr__(self, "mu_r", _coerce_scalar(mu_r, name="mu_r"))
        object.__setattr__(self, "sigma_e", _coerce_nonnegative(sigma_e, name="sigma_e"))
        object.__setattr__(self, "name", None if name is None else str(name))

    def capabilities(self) -> MaterialCapabilities:
        return MaterialCapabilities(
            conductive=not _is_numeric_close(self.sigma_e, 0.0),
            magnetic=not _is_numeric_close(self.mu_r, 1.0),
            anisotropic=False,
            dispersive=False,
        )

    def evaluate_static(self) -> StaticMaterialSample:
        return StaticMaterialSample(
            eps_r=self.eps_r,
            mu_r=self.mu_r,
            sigma_e=self.sigma_e,
        )

    def evaluate_at_frequency(self, frequency: float) -> FrequencyMaterialSample:
        resolved_frequency = float(frequency)
        if resolved_frequency < 0.0:
            raise ValueError("frequency must be >= 0.")
        sigma_scalar = _numeric_scalar(self.sigma_e)
        if resolved_frequency == 0.0:
            if sigma_scalar is not None and not np.isclose(sigma_scalar, 0.0):
                raise ValueError("Conductive materials require frequency > 0.")
            eps_scalar = _numeric_scalar(self.eps_r)
            mu_scalar = _numeric_scalar(self.mu_r)
            return FrequencyMaterialSample(
                eps_r=self.eps_r if eps_scalar is None else complex(eps_scalar, 0.0),
                mu_r=self.mu_r if mu_scalar is None else mu_scalar,
                sigma_e=self.sigma_e if sigma_scalar is None else sigma_scalar,
            )

        eps_scalar = _numeric_scalar(self.eps_r)
        mu_scalar = _numeric_scalar(self.mu_r)
        if eps_scalar is None or sigma_scalar is None:
            eps_r = self.eps_r
            mu_r = self.mu_r
            sigma_e = self.sigma_e
        else:
            eps_r = complex(
                eps_scalar,
                -sigma_scalar / (2.0 * np.pi * resolved_frequency * VACUUM_PERMITTIVITY),
            )
            mu_r = self.mu_r if mu_scalar is None else mu_scalar
            sigma_e = self.sigma_e if sigma_scalar is None else sigma_scalar
        return FrequencyMaterialSample(
            eps_r=eps_r,
            mu_r=mu_r,
            sigma_e=sigma_e,
        )


@dataclass(frozen=True, init=False)
class Structure:
    geometry: GeometrySpec
    material: MaterialSpec
    name: str | None
    priority: int
    enabled: bool
    tags: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __init__(
        self,
        geometry: GeometrySpec,
        material: MaterialSpec,
        name: str | None = None,
        priority: int = 0,
        enabled: bool = True,
        tags=(),
        metadata: Mapping[str, Any] | None = None,
    ):
        if not isinstance(geometry, GeometrySpec):
            raise TypeError("Structure geometry must implement GeometrySpec (kind and to_mesh()).")
        if not isinstance(material, MaterialSpec):
            raise TypeError(
                "Structure material must implement MaterialSpec "
                "(name, capabilities(), evaluate_static(), and evaluate_at_frequency())."
            )
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "material", material)
        object.__setattr__(self, "name", None if name is None else str(name))
        object.__setattr__(self, "priority", int(priority))
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "tags", tuple(str(tag) for tag in tags))
        object.__setattr__(self, "metadata", MappingProxyType(dict(metadata or {})))
