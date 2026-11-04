from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from witwin.core import Box
from witwin.core.geometry.base import _resolve_beta, occupancy_from_signed_distance


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_box_cell_profile_is_identical_at_metre_and_micrometre_scales(dtype):
    profiles = []
    for scale in (1.0, 1.0e-6):
        cell = 0.04 * scale
        box = Box(
            position=torch.tensor([-cell, 0.0, 0.0], dtype=dtype),
            size=torch.full((3,), 2 * cell, dtype=dtype),
        )
        x = cell * torch.tensor([0, 1, 2, 5, 15], dtype=dtype)
        profiles.append(box.to_mask(x, torch.zeros_like(x), torch.zeros_like(x), beta=0.05 * cell))
    assert torch.equal(profiles[0], profiles[1])
    assert torch.equal(profiles[0], torch.tensor([0.5, 0, 0, 0, 0], dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("beta_value", [5.0e-3, 2.0e-9])
def test_normal_beta_values_and_gradients_match_main_formula_bitwise(dtype, beta_value):
    scale = beta_value / 5.0e-3
    position = torch.tensor([-scale / 2, 0, 0], dtype=dtype, requires_grad=True)
    size = torch.full((3,), scale, dtype=dtype, requires_grad=True)
    beta = torch.tensor(beta_value, dtype=dtype, requires_grad=True)
    offset = torch.tensor(0.1 * beta_value, dtype=dtype, requires_grad=True)
    density = torch.tensor(16.0, dtype=dtype, requires_grad=True)
    x = beta_value * torch.tensor([-2, -0.5, 0, 0.5, 2], dtype=dtype)
    y = torch.zeros_like(x)
    box = Box(position=position, size=size)
    sdf = box.signed_distance(x, y, y)

    # Freeze main a302ac9's arithmetic. Its eps floor is inactive in all but
    # fp32 micrometre coordinates, where removing that floor IS the scale fix;
    # that case must match the same tanh formula with the authored beta.
    reference_beta = torch.clamp(beta, min=torch.finfo(dtype).eps)
    if beta_value < torch.finfo(dtype).eps:
        reference_beta = beta
    expected = (0.5 * (1.0 - torch.tanh((sdf - offset) / reference_beta))).clamp(0.0, 1.0)
    actual = box.to_mask(x, y, y, offset=offset, beta=beta)
    parameters = (position, size, beta, offset, density)
    expected_grad = torch.autograd.grad((density * expected).sum(), parameters, retain_graph=True)
    actual_grad = torch.autograd.grad((density * actual).sum(), parameters)

    assert torch.equal(actual, expected)
    assert bool(((actual > 0) & (actual < 1)).all())
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        assert torch.isfinite(actual_value).all()
        assert torch.equal(actual_value, expected_value)
    assert actual_grad[0][0] != 0
    assert actual_grad[1][0] != 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_beta_resolution_preserves_positive_lengths_and_only_floors_at_tiny(dtype):
    reference = torch.tensor(0.0, dtype=dtype)
    tiny = torch.finfo(dtype).tiny
    beta = torch.tensor([-1.0, 0.0, tiny, 1.0e-30, 2.0e-9, 5.0e-3], dtype=dtype)
    actual = _resolve_beta(beta, reference, reference, reference, reference=reference)
    assert torch.equal(actual[:2], torch.full((2,), tiny, dtype=dtype))
    assert torch.equal(actual[2:], beta[2:])


@pytest.mark.parametrize("shape", [(), (1,), (1, 1, 1), (2, 2, 2)])
@pytest.mark.parametrize("entrypoint", ["to_mask", "occupancy"])
def test_spacingless_queries_require_explicit_beta(shape, entrypoint):
    x = torch.zeros(shape)
    box = Box()

    def query(beta=None):
        if entrypoint == "to_mask":
            return box.to_mask(x, x, x, beta=beta)
        return occupancy_from_signed_distance(box.signed_distance(x, x, x), xx=x, yy=x, zz=x, beta=beta)

    with pytest.raises(ValueError, match="coordinate spacing; pass an explicit beta"):
        query()
    assert torch.equal(query(beta=0.01), torch.ones(shape))


@pytest.mark.parametrize("scale", [1.0, 1.0e-6])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_default_beta_uses_coordinate_spacing_at_each_scale_and_axis(scale, axis):
    axes = [torch.zeros(1) for _ in range(3)]
    axes[axis] = scale * torch.tensor([-0.1, 0.0, 0.1])
    coords = torch.meshgrid(*axes, indexing="ij")
    spacing = torch.diff(axes[axis]).abs().min()
    resolved = _resolve_beta(None, *coords, reference=coords[0])
    assert torch.equal(resolved, 0.05 * spacing)
    position = torch.zeros(3)
    position[axis] = -0.5 * scale + 0.0025 * scale
    box = Box(position=position, size=(scale, scale, scale))
    assert torch.equal(box.to_mask(*coords), box.to_mask(*coords, beta=0.05 * spacing))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("beta_value", ["tiny", 1.0e-30, 1.0e-12])
def test_surface_geometry_and_density_gradients_are_finite(dtype, beta_value):
    beta_value = torch.finfo(dtype).tiny if beta_value == "tiny" else beta_value
    guarded = beta_value < torch.finfo(dtype).tiny ** 0.5
    position = torch.zeros(3, dtype=dtype, requires_grad=True)
    size = torch.full((3,), 2.0, dtype=dtype, requires_grad=True)
    beta = torch.tensor(beta_value, dtype=dtype, requires_grad=True)
    density = torch.tensor(16.0, dtype=dtype, requires_grad=True)
    box = Box(position=position, size=size)
    x, y = torch.tensor(1.0, dtype=dtype), torch.tensor(0.0, dtype=dtype)
    warning = pytest.warns(RuntimeWarning, match="hard occupancy") if guarded else nullcontext()
    with warning:
        occupancy = box.to_mask(x, y, y, beta=beta)
    grads = torch.autograd.grad(density * occupancy, (position, size, beta, density))
    assert occupancy == 0.5
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert grads[-1] == 0.5
    if guarded:
        assert all(torch.count_nonzero(grad) == 0 for grad in grads[:-1])
    else:
        assert grads[0][0] > 0
        assert grads[1][0] > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("beta_value", ["tiny", 1.0e-30, 1.0e-12])
def test_near_surface_density_and_distance_gradients_are_finite(dtype, beta_value):
    beta_value = torch.finfo(dtype).tiny if beta_value == "tiny" else beta_value
    guarded = beta_value < torch.finfo(dtype).tiny ** 0.5
    sdf = (beta_value * torch.tensor([-2, -0.5, 0, 0.5, 2], dtype=dtype)).requires_grad_()
    density = torch.full_like(sdf, 16.0, requires_grad=True)
    beta = torch.tensor(beta_value, dtype=dtype, requires_grad=True)
    offset = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    warning = pytest.warns(RuntimeWarning, match="hard occupancy") if guarded else nullcontext()
    with warning:
        occupancy = occupancy_from_signed_distance(sdf, xx=sdf, yy=sdf, zz=sdf, beta=beta, offset=offset)
    grads = torch.autograd.grad((density * occupancy).sum(), (sdf, density, beta, offset))
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert torch.equal(grads[1], occupancy)
    if guarded:
        assert torch.equal(occupancy, sdf.new_tensor([1, 1, 0.5, 0, 0]))
        assert all(torch.count_nonzero(grads[index]) == 0 for index in (0, 2, 3))
    else:
        assert (grads[0] < 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hard_guard_handles_mixed_beta_without_poisoning_smooth_gradients(dtype):
    threshold = torch.finfo(dtype).tiny ** 0.5
    beta = torch.tensor([torch.finfo(dtype).tiny, threshold, 5.0e-3], dtype=dtype, requires_grad=True)
    sdf = torch.zeros(3, dtype=dtype, requires_grad=True)
    with pytest.warns(RuntimeWarning, match="hard occupancy"):
        occupancy = occupancy_from_signed_distance(sdf, xx=sdf, yy=sdf, zz=sdf, beta=beta)
    distance_grad, beta_grad = torch.autograd.grad((16.0 * occupancy).sum(), (sdf, beta))
    assert torch.isfinite(distance_grad).all()
    assert torch.isfinite(beta_grad).all()
    assert distance_grad[0] == 0
    assert torch.equal(distance_grad[1:], -8.0 / beta.detach()[1:])
