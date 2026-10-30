"""Solid-mesh winding signs across scales; unsigned-distance compatibility."""

import math
from pathlib import Path
import subprocess
import types

import numpy as np
import pytest
import torch

from witwin.core import Cone, Cylinder, Mesh
from witwin.core.geometry.mesh_sdf import (
    triangle_mesh_unsigned_distance,
    triangle_mesh_winding_angle,
)


def _repro():
    # Original mesh168.npz: retain the authored vertices, faces and three queries.
    vertices = torch.tensor([
        [0.004571450082693127, 0.00826040783945992, 0.004040021226138734],
        [0.0039788146001336604, 0.008915778365174489, 0.0034037648473543808],
        [0.006423184286207385, 0.009801802661296814, 0.003902941099184283],
        [0.005830548803647919, 0.010457173187011382, 0.0032666847203999295],
        [0.005457028891399427, 0.01028209359164398, 0.004365917697375018],
        [0.00431259414604651, 0.009329459043999342, 0.0044506382287066255],
        [0.006089404740294535, 0.00938812198247196, 0.002856067907886396],
        [0.004944969994941619, 0.008435487434827324, 0.0029407882491636447],
        [0.004916962064257014, 0.009612744187762414, 0.0026720415035986185],
        [0.0045261320010479755, 0.010165248932334142, 0.003605179895052712],
        [0.00587586688529307, 0.008552332094137161, 0.003701526051485951],
        [0.0054850368220840315, 0.009104836838708889, 0.004634664442940045],
        [0.004112577518317082, 0.008452781621221766, 0.0037339268986099727],
        [0.006289421368023964, 0.010264799405249537, 0.003572779047928691],
        [0.006069872093667184, 0.010161890291712987, 0.004218892025846947],
        [0.0057215294424897, 0.010547107669610575, 0.0038449099013988935],
        [0.00430876797170376, 0.008695936407105816, 0.004349263233950857],
        [0.003960425700634994, 0.009081154545220838, 0.003975281109502803],
        [0.004829298351131654, 0.009884253563480739, 0.004540820542189135],
        [0.006441573185706053, 0.009636426481250465, 0.00333142483703586],
        [0.006093230914637286, 0.010021644619365487, 0.0029574425225334486],
        [0.004680469443851346, 0.008170473356860728, 0.0034617960451397696],
        [0.004332126792673863, 0.008555690734758316, 0.0030878137306373584],
        [0.005572700535209392, 0.008833327462990564, 0.0027658854043495283],
        [0.004315663904042939, 0.009247664969677952, 0.00292984834018702],
        [0.005404086209005097, 0.010153673861691836, 0.002849274414846379],
        [0.005556238026687186, 0.009525301697910198, 0.0026079202039535486],
        [0.00488355620968592, 0.008965356705316913, 0.0026577174867168174],
        [0.004085939879556341, 0.00957241921422042, 0.0034783334350791746],
        [0.005174362184518499, 0.010478428106234304, 0.0033977595097385337],
        [0.004954812530053002, 0.010375518992697754, 0.0040438724876567895],
        [0.004282130713051736, 0.009815574760321901, 0.004093670150528775],
        [0.004637369296568399, 0.009982085184779017, 0.0030482371911586333],
        [0.005227636701822547, 0.008239152920236999, 0.00390894643680013],
        [0.0063160590067847045, 0.009145161812250883, 0.003828372511459489],
        [0.00611986817328931, 0.008902006266149402, 0.003213035796009888],
        [0.005447186356288044, 0.008342062033773549, 0.003262833458881874],
        [0.004997912677335949, 0.008563907164779467, 0.004457431531692284],
        [0.006086334982298106, 0.009469916056793351, 0.004376857606351643],
        [0.005518442676655126, 0.00975222432115439, 0.004648988459821846],
        [0.0048457608596538596, 0.009192279328561105, 0.004698785742585115],
        [0.005764629589772647, 0.008735495841692286, 0.004258468945434388],
    ], dtype=torch.float32)
    faces = torch.tensor([
        [0, 37, 16], [37, 11, 40], [16, 40, 5], [37, 40, 16],
        [0, 16, 12], [16, 5, 17], [12, 17, 1], [16, 17, 12],
        [0, 12, 21], [12, 1, 22], [21, 22, 7], [12, 22, 21],
        [0, 21, 33], [21, 7, 36], [33, 36, 10], [21, 36, 33],
        [0, 33, 37], [33, 10, 41], [37, 41, 11], [33, 41, 37],
        [1, 17, 28], [17, 5, 31], [28, 31, 9], [17, 31, 28],
        [5, 40, 18], [40, 11, 39], [18, 39, 4], [40, 39, 18],
        [11, 41, 38], [41, 10, 34], [38, 34, 2], [41, 34, 38],
        [10, 36, 35], [36, 7, 23], [35, 23, 6], [36, 23, 35],
        [7, 22, 27], [22, 1, 24], [27, 24, 8], [22, 24, 27],
        [3, 29, 15], [29, 9, 30], [15, 30, 4], [29, 30, 15],
        [3, 15, 13], [15, 4, 14], [13, 14, 2], [15, 14, 13],
        [3, 13, 20], [13, 2, 19], [20, 19, 6], [13, 19, 20],
        [3, 20, 25], [20, 6, 26], [25, 26, 8], [20, 26, 25],
        [3, 25, 29], [25, 8, 32], [29, 32, 9], [25, 32, 29],
        [4, 30, 18], [30, 9, 31], [18, 31, 5], [30, 31, 18],
        [2, 14, 38], [14, 4, 39], [38, 39, 11], [14, 39, 38],
        [6, 19, 35], [19, 2, 34], [35, 34, 10], [19, 34, 35],
        [8, 26, 27], [26, 6, 23], [27, 23, 7], [26, 23, 27],
        [9, 32, 28], [32, 8, 24], [28, 24, 1], [32, 24, 28],
    ])
    points = torch.tensor([
        [0.00448394487490265, 0.008768603310920737, 0.0035871558999221198],
        [0.00448394487490265, 0.008768603310920737, 0.0037864423388066816],
        [0.00448394487490265, 0.0089678897498053, 0.0035871558999221198],
    ], dtype=torch.float32)
    return vertices, faces, points


def _icosphere(subdivisions):
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ])
    for _ in range(subdivisions):
        edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=-1)
        unique_edges, inverse = np.unique(edges.reshape(-1, 2), axis=0, return_inverse=True)
        midpoints = vertices[unique_edges].mean(axis=1)
        midpoints /= np.linalg.norm(midpoints, axis=1, keepdims=True)
        ab, bc, ca = (inverse.reshape(-1, 3) + len(vertices)).T
        a, b, c = faces.T
        faces = np.stack([
            np.stack([a, ab, ca], axis=1), np.stack([ab, b, bc], axis=1),
            np.stack([ca, bc, c], axis=1), np.stack([ab, bc, ca], axis=1),
        ], axis=1).reshape(-1, 3)
        vertices = np.concatenate([vertices, midpoints])
    return torch.tensor(vertices * [1.0, 0.37, 0.12], dtype=torch.float32), torch.tensor(faces)


@pytest.fixture(scope="module")
def before():
    """Load the frozen pre-fix implementation for bitwise comparisons."""
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        ["git", "show", "4c384e0:witwin/core/geometry/mesh_sdf.py"], cwd=root, text=True
    )
    module = types.ModuleType("mesh_sdf_before_solid_sign_fix")
    exec(compile(source, "4c384e0:mesh_sdf.py", "exec"), module.__dict__)
    return module


def _reference_winding(points, vertices, faces):
    # Independent float64 solid angles, without dimensional clamps or chunking.
    a, b, c = np.moveaxis(
        vertices.double().numpy()[faces.numpy()][None] - points.double().numpy()[:, None, None],
        2, 0,
    )
    la, lb, lc = (np.linalg.norm(v, axis=-1) for v in (a, b, c))
    numerator = np.einsum("...i,...i->...", a, np.cross(b, c))
    denominator = la * lb * lc + (
        (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
    )
    return torch.from_numpy((2.0 * np.arctan2(numerator, denominator)).sum(axis=1))


def test_repro_three_points_are_inside():
    vertices, faces, points = _repro()
    mesh = Mesh(vertices, faces, recenter=False, fill_mode="solid")
    assert (_reference_winding(points, vertices, faces).abs() > 2 * math.pi).all()
    assert (mesh.signed_distance(*points.T[:, :, None]).flatten() < 0).all()


@pytest.mark.parametrize("kind", ["repro", "ico1", "ico2", "cylinder", "cone"])
@pytest.mark.parametrize("scale", [1.0e-3, 1.0, 100.0], ids=["1mm", "1m", "100m"])
def test_sign_reference_and_unsigned_compatibility(kind, scale, before):
    if kind == "repro":
        vertices, faces, _ = _repro()
    elif kind.startswith("ico"):
        vertices, faces = _icosphere(int(kind[-1]))
    else:
        primitive = Cylinder if kind == "cylinder" else Cone
        vertices, faces = primitive(radius=0.37, height=1.0).to_mesh(segments=32)
    center = (vertices.amax(0) + vertices.amin(0)) / 2
    diagonal = torch.linalg.vector_norm(vertices.amax(0) - vertices.amin(0))
    vertices = (vertices - center) / diagonal * scale
    extent = vertices.amax(0) - vertices.amin(0)
    points = (torch.rand((512, 3), generator=torch.Generator().manual_seed(168)) - 0.5) * extent * 1.5
    # Sample both sides close to every face, beyond the relative exclusion band.
    triangles = vertices[faces]
    normals = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
    centroids = triangles.mean(1)
    points = torch.cat([points, centroids + normals * (scale * 1e-5), centroids - normals * (scale * 1e-5)])
    # Distance is used only to exclude surface queries, in normalized float64 units.
    distance = before.triangle_mesh_unsigned_distance(points.double() / scale, vertices.double() / scale, faces)
    # Undo the unsigned path's square-root smoothing for the exclusion threshold.
    distance = torch.sqrt(distance.square() + 2e-6 * distance)
    points = points[distance > 1e-6]
    reference = _reference_winding(points, vertices, faces)
    inside = reference.abs() > 2 * math.pi
    assert inside.any() and (~inside).any()

    # Auto fill also exercises the scale-dependent area and volume diagnostics.
    mesh = Mesh(vertices, faces, recenter=False)
    numpy_mesh = Mesh(vertices.numpy(), faces.numpy(), recenter=False)
    assert mesh.is_watertight and numpy_mesh.is_watertight
    signed = mesh.signed_distance(*points.T[:, :, None]).flatten()
    winding = triangle_mesh_winding_angle(points, vertices, faces)
    assert torch.equal(signed < 0, inside)
    unsigned = triangle_mesh_unsigned_distance(points, vertices, faces)
    assert torch.equal(unsigned, before.triangle_mesh_unsigned_distance(points, vertices, faces))
    factor = -torch.tanh((winding.abs() - 2 * math.pi) / 0.5)
    assert torch.equal(signed.abs(), unsigned * factor.abs())
    old_winding = before.triangle_mesh_winding_angle(points, vertices, faces)
    unchanged = ((winding - old_winding).abs() / (4 * math.pi)) <= 1e-6
    if scale >= 1.0:
        assert unchanged.any()
    if unchanged.any():
        old_signed = before.triangle_mesh_smooth_signed_distance(points, vertices, faces)
        assert torch.equal(signed[unchanged], old_signed[unchanged])


def test_winding_is_differentiable_in_vertices():
    vertices, faces = _icosphere(1)
    vertices = (vertices.double() * 1e-3).requires_grad_()
    points = torch.tensor([[0.2, 0.1, 0.3]], dtype=torch.float64) * 1e-3
    # An open patch avoids a constant integer winding and tests nonzero derivatives.
    patch = faces[:3]
    assert torch.autograd.gradcheck(
        lambda v: triangle_mesh_winding_angle(points, v, patch), (vertices,), eps=1e-8
    )
