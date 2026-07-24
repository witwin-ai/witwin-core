from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch

from .identity import MaterialId, new_material_id, reserve_material_id
from .scalars import ComplexLike, ScalarLike

VACUUM_PERMITTIVITY = 8.8541878128e-12


def _version_value(value: int) -> int:
    resolved = int(value)
    if resolved < 0:
        raise ValueError("version must be non-negative.")
    return resolved


def _validate_scalar(
    value: ScalarLike,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> ScalarLike:
    """Validate Python scalars without observing tensor values on the host."""

    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar tensor.")
        if not value.dtype.is_floating_point:
            raise TypeError(f"{name} must use a floating-point tensor dtype.")
        return value
    if not isinstance(value, Real):
        return value
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None:
        invalid = scalar <= minimum if strict_minimum else scalar < minimum
        if invalid:
            relation = ">" if strict_minimum else ">="
            raise ValueError(f"{name} must be {relation} {minimum}.")
    if maximum is not None and scalar > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")
    return value


def _is_numeric_close(value: ScalarLike, target: float) -> bool:
    if isinstance(value, torch.Tensor) or not isinstance(value, Real):
        return False
    return bool(np.isclose(float(value), target))


def _complex_permittivity(
    eps_r: ScalarLike,
    sigma_e: ScalarLike,
    frequency_hz: ScalarLike,
) -> ComplexLike:
    tensors = tuple(
        value
        for value in (eps_r, sigma_e, frequency_hz)
        if isinstance(value, torch.Tensor)
    )
    tensor = tensors[0] if tensors else None
    if tensor is None:
        frequency = float(frequency_hz)
        if frequency < 0.0:
            raise ValueError("frequency_hz must be >= 0.")
        if frequency == 0.0:
            if not _is_numeric_close(sigma_e, 0.0):
                raise ValueError("Conductive materials require frequency_hz > 0.")
            return complex(float(eps_r), 0.0)
        return complex(
            float(eps_r),
            -float(sigma_e)
            / (2.0 * math.pi * frequency * VACUUM_PERMITTIVITY),
        )

    if not tensor.dtype.is_floating_point:
        raise TypeError("Material tensors must use a floating-point dtype.")
    if any(value.device != tensor.device for value in tensors[1:]):
        raise ValueError(
            "Material and frequency tensors must be on the same device."
        )
    if any(value.dtype != tensor.dtype for value in tensors[1:]):
        raise ValueError(
            "Material and frequency tensors must use the same dtype."
        )
    real_dtype = tensor.dtype
    device = tensor.device
    eps_tensor = torch.as_tensor(eps_r, device=device, dtype=real_dtype)
    sigma_tensor = torch.as_tensor(sigma_e, device=device, dtype=real_dtype)
    frequency_tensor = torch.as_tensor(
        frequency_hz, device=device, dtype=real_dtype
    )
    imaginary = -sigma_tensor / (
        2.0 * math.pi * frequency_tensor * VACUUM_PERMITTIVITY
    )
    return torch.complex(eps_tensor, imaginary)


@dataclass(frozen=True)
class MaterialCapabilities:
    conductive: bool = False
    perfect_conductor: bool = False
    magnetic: bool = False
    anisotropic: bool = False
    dispersive: bool = False
    layered: bool = False
    rough: bool = False
    phase_screen: bool = False


@dataclass(frozen=True)
class StaticMaterialSample:
    eps_r: ScalarLike
    mu_r: ScalarLike = 1.0
    sigma_e: ScalarLike = 0.0


@dataclass(frozen=True)
class FrequencyMaterialSample:
    eps_r: ScalarLike
    mu_r: ScalarLike = 1.0
    sigma_e: ScalarLike = 0.0


@runtime_checkable
class DispersionSpec(Protocol):
    def complex_eps(self, frequency_hz: ScalarLike) -> ComplexLike:
        ...


@dataclass(frozen=True)
class PowerLawDispersion:
    """Solver-neutral power-law material dispersion."""

    reference_frequency_hz: ScalarLike
    eps_r_exponent: ScalarLike = 0.0
    sigma_e_exponent: ScalarLike = 0.0

    def __post_init__(self) -> None:
        _validate_scalar(
            self.reference_frequency_hz,
            name="reference_frequency_hz",
            minimum=0.0,
            strict_minimum=True,
        )

    def scale(self, frequency_hz: ScalarLike) -> tuple[ScalarLike, ScalarLike]:
        ratio = frequency_hz / self.reference_frequency_hz
        return ratio**self.eps_r_exponent, ratio**self.sigma_e_exponent

    def complex_eps(self, frequency_hz: ScalarLike) -> ComplexLike:
        raise TypeError(
            "PowerLawDispersion requires the owning PhysicalMaterial base values; "
            "use PhysicalMaterial.evaluate_at_frequency()."
        )


@dataclass(frozen=True)
class MaterialLayer:
    """One homogeneous logical layer in a physical material."""

    thickness_m: ScalarLike
    eps_r: ScalarLike = 1.0
    sigma_e: ScalarLike = 0.0
    mu_r: ScalarLike = 1.0
    dispersion: DispersionSpec | None = None

    def __post_init__(self) -> None:
        _validate_scalar(
            self.thickness_m,
            name="thickness_m",
            minimum=0.0,
            strict_minimum=True,
        )
        _validate_scalar(
            self.eps_r, name="eps_r", minimum=0.0, strict_minimum=True
        )
        _validate_scalar(
            self.mu_r, name="mu_r", minimum=0.0, strict_minimum=True
        )
        _validate_scalar(self.sigma_e, name="sigma_e", minimum=0.0)
        if self.dispersion is not None and not isinstance(
            self.dispersion, DispersionSpec
        ):
            raise TypeError("dispersion must implement DispersionSpec.")

    def evaluate_at_frequency(
        self, frequency_hz: ScalarLike
    ) -> FrequencyMaterialSample:
        if isinstance(self.dispersion, PowerLawDispersion):
            eps_scale, sigma_scale = self.dispersion.scale(frequency_hz)
            eps_r = _complex_permittivity(
                self.eps_r * eps_scale,
                self.sigma_e * sigma_scale,
                frequency_hz,
            )
        elif self.dispersion is not None:
            eps_r = self.dispersion.complex_eps(frequency_hz)
        else:
            eps_r = _complex_permittivity(
                self.eps_r, self.sigma_e, frequency_hz
            )
        return FrequencyMaterialSample(
            eps_r=eps_r,
            mu_r=self.mu_r,
            sigma_e=self.sigma_e,
        )


@dataclass(frozen=True)
class SurfaceRoughness:
    """Logical Gaussian-correlated surface roughness."""

    rms_height_m: ScalarLike
    correlation_length_x_m: ScalarLike
    correlation_length_y_m: ScalarLike
    principal_axis_rad: ScalarLike = 0.0
    correlation: str = "gaussian"

    def __post_init__(self) -> None:
        _validate_scalar(
            self.rms_height_m, name="rms_height_m", minimum=0.0
        )
        _validate_scalar(
            self.correlation_length_x_m,
            name="correlation_length_x_m",
            minimum=0.0,
            strict_minimum=True,
        )
        _validate_scalar(
            self.correlation_length_y_m,
            name="correlation_length_y_m",
            minimum=0.0,
            strict_minimum=True,
        )
        _validate_scalar(self.principal_axis_rad, name="principal_axis_rad")
        if self.correlation != "gaussian":
            raise ValueError("correlation must be 'gaussian'.")


@dataclass(frozen=True)
class PhaseScreen:
    """Logical phase-screen descriptor; native storage remains solver-owned."""

    height: torch.Tensor
    height_scale_m: ScalarLike
    height_offset_m: ScalarLike = 0.0
    realization_id: int = 0
    mode: str = "realization_coherent"
    correlation: SurfaceRoughness | None = None
    quadrature_tolerance: ScalarLike = 1.0e-4

    def __post_init__(self) -> None:
        if not isinstance(self.height, torch.Tensor):
            raise TypeError("height must be a torch.Tensor.")
        if self.height.ndim != 2 or self.height.numel() == 0:
            raise ValueError("height must have shape (H, W) with H, W > 0.")
        if not self.height.dtype.is_floating_point:
            raise TypeError("height must use a floating-point dtype.")
        _validate_scalar(
            self.height_scale_m,
            name="height_scale_m",
            minimum=0.0,
            strict_minimum=True,
        )
        _validate_scalar(self.height_offset_m, name="height_offset_m")
        if int(self.realization_id) < 0:
            raise ValueError("realization_id must be non-negative.")
        if self.mode not in {"realization_coherent", "ensemble_bsdf"}:
            raise ValueError(
                "mode must be 'realization_coherent' or 'ensemble_bsdf'."
            )
        _validate_scalar(
            self.quadrature_tolerance,
            name="quadrature_tolerance",
            minimum=0.0,
            strict_minimum=True,
        )


@runtime_checkable
class MaterialSpec(Protocol):
    name: str | None

    def capabilities(self) -> MaterialCapabilities:
        ...

    def evaluate_static(self) -> StaticMaterialSample:
        ...

    def evaluate_at_frequency(
        self, frequency: ScalarLike
    ) -> FrequencyMaterialSample:
        ...


@dataclass(frozen=True, init=False)
class PhysicalMaterial(MaterialSpec):
    """Canonical solver-neutral electromagnetic and surface material."""

    eps_r: ScalarLike
    mu_r: ScalarLike
    sigma_e: ScalarLike
    name: str | None
    material_id: MaterialId
    version: int
    thickness_m: ScalarLike
    layers: tuple[MaterialLayer, ...]
    geometry_mode: str
    roughness_front: SurfaceRoughness | None
    roughness_back: SurfaceRoughness | None
    dispersion: DispersionSpec | PowerLawDispersion | None
    gain: ScalarLike
    scattering_coefficient: ScalarLike
    xpd_coefficient: ScalarLike
    conductor_model: Literal["finite", "perfect"]

    def __init__(
        self,
        eps_r: ScalarLike = 1.0,
        mu_r: ScalarLike = 1.0,
        sigma_e: ScalarLike = 0.0,
        name: str | None = None,
        *,
        material_id: MaterialId | int | None = None,
        version: int = 0,
        thickness_m: ScalarLike = 0.1,
        layers=(),
        geometry_mode: str = "thin_sheet",
        roughness_front: SurfaceRoughness | None = None,
        roughness_back: SurfaceRoughness | None = None,
        dispersion: DispersionSpec | PowerLawDispersion | None = None,
        gain: ScalarLike = 1.0,
        scattering_coefficient: ScalarLike = 0.0,
        xpd_coefficient: ScalarLike = 0.0,
        conductor_model: Literal["finite", "perfect"] = "finite",
    ):
        _validate_scalar(
            eps_r, name="eps_r", minimum=0.0, strict_minimum=True
        )
        _validate_scalar(
            mu_r, name="mu_r", minimum=0.0, strict_minimum=True
        )
        _validate_scalar(sigma_e, name="sigma_e", minimum=0.0)
        _validate_scalar(
            thickness_m,
            name="thickness_m",
            minimum=0.0,
            strict_minimum=True,
        )
        _validate_scalar(gain, name="gain", minimum=0.0)
        _validate_scalar(
            scattering_coefficient,
            name="scattering_coefficient",
            minimum=0.0,
            maximum=1.0,
        )
        _validate_scalar(
            xpd_coefficient,
            name="xpd_coefficient",
            minimum=0.0,
            maximum=1.0,
        )
        normalized_layers = tuple(layers)
        if any(not isinstance(layer, MaterialLayer) for layer in normalized_layers):
            raise TypeError("layers must contain only MaterialLayer values.")
        if geometry_mode not in {"thin_sheet", "volumetric"}:
            raise ValueError(
                "geometry_mode must be 'thin_sheet' or 'volumetric'."
            )
        if roughness_front is not None and not isinstance(
            roughness_front, SurfaceRoughness
        ):
            raise TypeError("roughness_front must be a SurfaceRoughness or None.")
        if roughness_back is not None and not isinstance(
            roughness_back, SurfaceRoughness
        ):
            raise TypeError("roughness_back must be a SurfaceRoughness or None.")
        if dispersion is not None and not isinstance(
            dispersion, (DispersionSpec, PowerLawDispersion)
        ):
            raise TypeError("dispersion must implement DispersionSpec.")
        if conductor_model not in {"finite", "perfect"}:
            raise ValueError(
                "conductor_model must be 'finite' or 'perfect'."
            )

        object.__setattr__(self, "eps_r", eps_r)
        object.__setattr__(self, "mu_r", mu_r)
        object.__setattr__(self, "sigma_e", sigma_e)
        object.__setattr__(self, "name", None if name is None else str(name))
        object.__setattr__(
            self,
            "material_id",
            new_material_id()
            if material_id is None
            else reserve_material_id(int(material_id)),
        )
        object.__setattr__(self, "version", _version_value(version))
        object.__setattr__(self, "thickness_m", thickness_m)
        object.__setattr__(self, "layers", normalized_layers)
        object.__setattr__(self, "geometry_mode", geometry_mode)
        object.__setattr__(self, "roughness_front", roughness_front)
        object.__setattr__(self, "roughness_back", roughness_back)
        object.__setattr__(self, "dispersion", dispersion)
        object.__setattr__(self, "gain", gain)
        object.__setattr__(
            self, "scattering_coefficient", scattering_coefficient
        )
        object.__setattr__(self, "xpd_coefficient", xpd_coefficient)
        object.__setattr__(self, "conductor_model", conductor_model)

    def capabilities(self) -> MaterialCapabilities:
        return MaterialCapabilities(
            conductive=self.conductor_model == "perfect"
            or not _is_numeric_close(self.sigma_e, 0.0),
            perfect_conductor=self.conductor_model == "perfect",
            magnetic=not _is_numeric_close(self.mu_r, 1.0),
            anisotropic=False,
            dispersive=self.dispersion is not None
            or any(layer.dispersion is not None for layer in self.layers),
            layered=bool(self.layers),
            rough=self.roughness_front is not None
            or self.roughness_back is not None,
            phase_screen=False,
        )

    @classmethod
    def perfect_conductor(
        cls,
        name: str | None = "perfect_conductor",
        **kwargs,
    ) -> PhysicalMaterial:
        """Create an explicit PEC specification without infinite parameters."""

        if "conductor_model" in kwargs:
            raise TypeError(
                "perfect_conductor() fixes conductor_model='perfect'."
            )
        return cls(
            eps_r=1.0,
            mu_r=1.0,
            sigma_e=0.0,
            name=name,
            conductor_model="perfect",
            **kwargs,
        )

    def evaluate_static(self) -> StaticMaterialSample:
        return StaticMaterialSample(
            eps_r=self.eps_r,
            mu_r=self.mu_r,
            sigma_e=self.sigma_e,
        )

    def evaluate_at_frequency(
        self, frequency: ScalarLike
    ) -> FrequencyMaterialSample:
        eps_r = self.eps_r
        sigma_e = self.sigma_e
        if isinstance(self.dispersion, PowerLawDispersion):
            eps_scale, sigma_scale = self.dispersion.scale(frequency)
            eps_r = eps_r * eps_scale
            sigma_e = sigma_e * sigma_scale
            resolved_eps = _complex_permittivity(eps_r, sigma_e, frequency)
        elif self.dispersion is not None:
            resolved_eps = self.dispersion.complex_eps(frequency)
        else:
            resolved_eps = _complex_permittivity(eps_r, sigma_e, frequency)
        return FrequencyMaterialSample(
            eps_r=resolved_eps,
            mu_r=self.mu_r,
            sigma_e=sigma_e,
        )
