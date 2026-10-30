"""Axis orientation, SDF agreement, and exact 0.4.2 z-mesh compatibility."""

import math
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

from witwin.core import Cone, Cylinder, Prism, Pyramid, Torus


ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = 64


@pytest.fixture(scope="module")
def release_042():
    """Import the unmodified release's base and primitives under a separate package."""
    package_name = "witwin.core._to_mesh_axis_042"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "witwin/core/geometry")]
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, package_name, package)
        for name in ("base", "primitives"):
            path = f"witwin/core/geometry/{name}.py"
            source = subprocess.check_output(
                ["git", "show", f"e50282b:{path}"], cwd=ROOT, text=True
            )
            module = types.ModuleType(f"{package_name}.{name}")
            module.__package__ = package_name
            monkeypatch.setitem(sys.modules, module.__name__, module)
            exec(compile(source, f"e50282b:{path}", "exec"), module.__dict__)
        yield module


# At 64 segments, circular cross sections lose 0.161% volume; the torus
# (128 major / 64 minor segments) loses 0.201%. Polyhedra are exact to float32.
@pytest.mark.parametrize(
    "primitive,kwargs,analytic_volume,rel_tol",
    [
        (Cone, dict(radius=1.0, height=1.0), math.pi / 3, 0.002),
        (Cylinder, dict(radius=1.0, height=1.0), math.pi, 0.002),
        (Pyramid, dict(base_size=1.0, height=1.0), 1 / 3, 1e-6),
        (Prism, dict(radius=1.0, height=1.0, num_sides=5),
         2.5 * math.sin(2 * math.pi / 5), 1e-6),
        (Torus, dict(major_radius=1.0, minor_radius=0.25),
         2 * math.pi**2 * 0.25**2, 0.003),
    ],
    ids=["cone", "cylinder", "pyramid", "prism", "torus"],
)
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_to_mesh_axis_orientation(
    primitive, kwargs, analytic_volume, rel_tol, axis, release_042
):
    geometry = primitive(axis=axis, **kwargs)
    vertices, faces = geometry.to_mesh(segments=SEGMENTS)
    volume = torch.linalg.det(vertices.double()[faces]).sum().item() / 6
    assert volume > 0
    assert volume == pytest.approx(analytic_volume, rel=rel_tol)
    assert geometry.signed_distance(*vertices.unbind(dim=1)).abs().max().item() <= 1e-5

    if axis == "z":
        original = getattr(release_042, primitive.__name__)(axis=axis, **kwargs)
        original_vertices, original_faces = original.to_mesh(segments=SEGMENTS)
        assert torch.equal(vertices, original_vertices)
        assert torch.equal(faces, original_faces)
