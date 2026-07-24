from __future__ import annotations

import torch
import pytest

from witwin.core import (
    MaterialLayer,
    PhaseScreen,
    PhysicalMaterial,
    PowerLawDispersion,
    SurfaceRoughness,
)


def test_physical_material_expresses_channel_logical_surface_fields():
    eps_r = torch.nn.Parameter(torch.tensor(4.0))
    sigma_e = torch.nn.Parameter(torch.tensor(0.01))
    thickness = torch.nn.Parameter(torch.tensor(0.05))
    roughness = SurfaceRoughness(
        rms_height_m=0.001,
        correlation_length_x_m=0.02,
        correlation_length_y_m=0.03,
    )
    layer = MaterialLayer(
        thickness_m=thickness,
        eps_r=eps_r,
        sigma_e=sigma_e,
    )
    material = PhysicalMaterial(
        eps_r=eps_r,
        sigma_e=sigma_e,
        thickness_m=thickness,
        layers=(layer,),
        geometry_mode="thin_sheet",
        roughness_front=roughness,
        roughness_back=roughness,
        gain=0.8,
        scattering_coefficient=0.2,
        xpd_coefficient=0.1,
    )

    assert material.eps_r is eps_r
    assert material.sigma_e is sigma_e
    assert material.thickness_m is thickness
    assert material.layers[0].eps_r is eps_r
    assert material.layers[0].thickness_m is thickness
    assert material.geometry_mode == "thin_sheet"
    assert material.roughness_front is roughness
    assert material.roughness_back is roughness
    assert material.capabilities().layered
    assert material.capabilities().rough


def test_material_frequency_evaluation_keeps_autograd_path():
    eps_r = torch.nn.Parameter(torch.tensor(3.0))
    sigma_e = torch.nn.Parameter(torch.tensor(0.02))
    frequency = torch.nn.Parameter(torch.tensor(10.0e9))
    material = PhysicalMaterial(eps_r=eps_r, sigma_e=sigma_e)

    sample = material.evaluate_at_frequency(frequency)
    loss = sample.eps_r.real + sample.eps_r.imag
    loss.backward()

    assert eps_r.grad is not None
    assert sigma_e.grad is not None
    assert frequency.grad is not None


def test_power_law_dispersion_is_solver_neutral_and_differentiable():
    eps_r = torch.nn.Parameter(torch.tensor(3.0))
    sigma_e = torch.nn.Parameter(torch.tensor(0.01))
    dispersion = PowerLawDispersion(
        reference_frequency_hz=10.0e9,
        eps_r_exponent=0.2,
        sigma_e_exponent=0.1,
    )
    material = PhysicalMaterial(
        eps_r=eps_r,
        sigma_e=sigma_e,
        dispersion=dispersion,
    )

    sample = material.evaluate_at_frequency(torch.tensor(20.0e9))
    sample.eps_r.abs().backward()

    assert eps_r.grad is not None
    assert sigma_e.grad is not None
    assert material.capabilities().dispersive


def test_layer_power_law_dispersion_evaluates_from_layer_base_values():
    eps_r = torch.nn.Parameter(torch.tensor(3.0))
    sigma_e = torch.nn.Parameter(torch.tensor(0.01))
    layer = MaterialLayer(
        thickness_m=0.05,
        eps_r=eps_r,
        sigma_e=sigma_e,
        dispersion=PowerLawDispersion(
            reference_frequency_hz=10.0e9,
            eps_r_exponent=0.2,
            sigma_e_exponent=0.1,
        ),
    )

    sample = layer.evaluate_at_frequency(torch.tensor(20.0e9))
    sample.eps_r.abs().backward()

    assert eps_r.grad is not None
    assert sigma_e.grad is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)
def test_material_frequency_evaluation_rejects_mixed_tensor_devices():
    material = PhysicalMaterial(
        eps_r=torch.tensor(3.0),
        sigma_e=torch.tensor(0.01, device="cuda"),
    )

    with pytest.raises(ValueError, match="same device"):
        material.evaluate_at_frequency(torch.tensor(10.0e9, device="cuda"))


def test_phase_screen_keeps_height_tensor_identity():
    height = torch.nn.Parameter(torch.zeros((4, 5)))
    phase_screen = PhaseScreen(
        height=height,
        height_scale_m=0.01,
        quadrature_tolerance=2.0e-4,
    )

    assert phase_screen.height is height
    assert phase_screen.height.requires_grad
    assert phase_screen.quadrature_tolerance == 2.0e-4
