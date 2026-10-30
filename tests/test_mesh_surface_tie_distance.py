"""Signed distance of solid meshes at query points lying on the surface.

The points come from two generated scene meshes (p16s positions 1981 and 2827)
where a quarter-cell grid point lies within 2e-7 wavelengths of a triangle and
the winding number is about 1. Core 0.4.5 returns +31 um and +74 um there,
which is far larger than the true distance; the unsigned-distance path still
clamps length-squared denominators at an absolute epsilon and adds a fixed
1 um square-root smoothing. Tracked for the 0.4.6 epsilon cleanup.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from witwin.core import Mesh

_DATA = np.load(Path(__file__).parent / "data" / "mesh_surface_tie_points.npz")


@pytest.mark.xfail(strict=True, reason="unsigned-distance absolute epsilons (core 0.4.6 cleanup)")
@pytest.mark.parametrize("case", [0, 1])
def test_on_surface_point_has_near_zero_signed_distance(case):
    vertices = torch.as_tensor(_DATA[f"v{case}"])
    faces = torch.as_tensor(_DATA[f"f{case}"])
    mesh = Mesh(vertices, faces, recenter=False, scale=1.0, fill_mode="solid", topology_diagnostics=False, device="cpu")
    x, y, z = (torch.tensor([value], dtype=torch.float32) for value in _DATA[f"axes{case}"])
    distance = float(mesh.signed_distance(x, y, z).flatten()[0])
    diagonal = float(torch.linalg.vector_norm(vertices.amax(0) - vertices.amin(0)))
    assert abs(distance) <= 1.0e-5 * diagonal
