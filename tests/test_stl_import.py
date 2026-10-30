from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from witwin.core import Box, Mesh


def _cube_triangles(size=1.0):
    vertices, faces = Box(size=(size, size, size)).to_mesh()
    return vertices.numpy()[faces.numpy()].astype(np.float32)


def _write_ascii(path, triangles, *, polygons=None):
    lines = ["solid cube"]
    for triangle in triangles:
        lines += ["  facet normal 0 0 0", "    outer loop"]
        lines += [f"      vertex {x!r} {y!r} {z!r}" for x, y, z in triangle.tolist()]
        lines += ["    endloop", "  endfacet"]
    for polygon in polygons or ():
        lines += ["  facet normal 0 0 0", "    outer loop"]
        lines += [f"      vertex {x!r} {y!r} {z!r}" for x, y, z in polygon]
        lines += ["    endloop", "  endfacet"]
    lines.append("endsolid cube")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _write_binary(path, triangles, *, header=b"binary cube"):
    payload = bytearray(header.ljust(80, b" ")[:80])
    payload += struct.pack("<I", len(triangles))
    for triangle in triangles:
        payload += struct.pack("<3f", 0.0, 0.0, 0.0)
        payload += struct.pack("<9f", *triangle.reshape(-1).tolist())
        payload += struct.pack("<H", 0)
    path.write_bytes(bytes(payload))
    return path


def test_ascii_and_binary_stl_load_identical_welded_watertight_meshes(tmp_path):
    triangles = _cube_triangles()
    ascii_mesh = Mesh.from_stl(_write_ascii(tmp_path / "cube_ascii.stl", triangles))
    binary_mesh = Mesh.from_stl(_write_binary(tmp_path / "cube_binary.stl", triangles))
    for mesh in (ascii_mesh, binary_mesh):
        assert mesh.vertex_count == 8
        assert mesh.face_count == 12
        assert mesh.is_watertight
        assert mesh.enclosed_volume == pytest.approx(1.0, rel=1.0e-6)
        assert mesh.source_path is not None
    torch.testing.assert_close(ascii_mesh.vertices, binary_mesh.vertices, rtol=0, atol=0)
    torch.testing.assert_close(ascii_mesh.faces, binary_mesh.faces, rtol=0, atol=0)
    # Corner winding is preserved: welded faces reproduce the authored triangles.
    torch.testing.assert_close(ascii_mesh.vertices[ascii_mesh.faces], torch.from_numpy(triangles))
    sd = ascii_mesh.signed_distance(torch.zeros(1), torch.zeros(1), torch.zeros(1))
    assert float(sd.reshape(-1)[0]) == pytest.approx(-0.5, abs=1.0e-4)


def test_binary_stl_with_solid_header_is_detected_by_size(tmp_path):
    triangles = _cube_triangles()
    mesh = Mesh.from_stl(_write_binary(tmp_path / "solid_header.stl", triangles, header=b"solid exported"))
    assert mesh.vertex_count == 8 and mesh.is_watertight


def test_ascii_polygon_facets_are_fan_triangulated(tmp_path):
    triangles = _cube_triangles()
    # Replace the two -z triangles with one quad facet.
    bottom = [tuple(v) for v in ((-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, -0.5, -0.5))]
    mesh = Mesh.from_stl(_write_ascii(tmp_path / "quad.stl", triangles[2:], polygons=[bottom]))
    assert mesh.face_count == 12
    assert mesh.is_watertight


def test_unit_scale_converts_file_units_without_changing_topology(tmp_path):
    path = _write_binary(tmp_path / "cube_mm.stl", _cube_triangles(size=2.0))
    reference = Mesh.from_stl(path, recenter=False)
    scaled = Mesh.from_stl(path, unit_scale=1.0e-3, scale=(1.0, 2.0, 1.0), recenter=False)
    assert scaled.is_watertight
    torch.testing.assert_close(scaled.vertices, reference.vertices)
    torch.testing.assert_close(
        scaled.world_vertices,
        reference.vertices * torch.tensor([1.0e-3, 2.0e-3, 1.0e-3]),
    )
    torch.testing.assert_close(
        scaled.bounds_world,
        torch.tensor([[-1.0e-3, -2.0e-3, -1.0e-3], [1.0e-3, 2.0e-3, 1.0e-3]]),
    )
    with pytest.raises(ValueError):
        Mesh.from_stl(path, unit_scale=0.0)


def test_weld_tolerance_merges_nearly_coincident_corners(tmp_path):
    triangles = _cube_triangles()
    triangles[0, 0] += np.float32(2.0e-6)
    path = _write_ascii(tmp_path / "jitter.stl", triangles)
    exact = Mesh.from_stl(path)
    assert exact.vertex_count == 9
    assert not exact.is_watertight
    welded = Mesh.from_stl(path, weld_tolerance=1.0e-4)
    assert welded.vertex_count == 8
    assert welded.is_watertight
    with pytest.raises(ValueError):
        Mesh.from_stl(path, weld_tolerance=0.0)


def test_welding_drops_collapsed_facets(tmp_path):
    triangles = _cube_triangles()
    sliver = np.array([[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5 + 1.0e-7], [0.5, 0.5, 0.5]]], dtype=np.float32)
    mesh = Mesh.from_stl(_write_binary(tmp_path / "sliver.stl", np.concatenate([triangles, sliver])), weld_tolerance=1.0e-5)
    assert mesh.face_count == 12
    assert mesh.is_watertight


def test_invalid_stl_files_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        Mesh.from_stl(tmp_path / "missing.stl")
    garbage = tmp_path / "garbage.stl"
    garbage.write_bytes(b"\x00\x01not an stl")
    with pytest.raises(ValueError):
        Mesh.from_stl(garbage)
    empty = tmp_path / "empty.stl"
    empty.write_text("solid empty\nendsolid empty\n", encoding="ascii")
    with pytest.raises(ValueError):
        Mesh.from_stl(empty)
    broken = tmp_path / "broken.stl"
    broken.write_text("solid x\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nendloop\nendfacet\nendsolid x\n")
    with pytest.raises(ValueError):
        Mesh.from_stl(broken)
