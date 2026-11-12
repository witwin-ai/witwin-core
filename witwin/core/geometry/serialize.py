"""Constructor-based geometry dictionaries for ``witwin.scene/1``."""

from __future__ import annotations

from inspect import Parameter, signature

import torch

from .base import GeometryBase, _rotation_quaternion
from .boolean import Difference, Intersection, Union
from .mesh import Mesh
from .primitives import Box, Cone, Cylinder, Ellipsoid, ExtrudedPolygon, HollowBox, Prism, Pyramid, Sphere, Torus


_GEOMETRY_TYPES = (
    Box, Sphere, Cylinder, Ellipsoid, Cone, Pyramid, Prism, ExtrudedPolygon, Torus, HollowBox,
    Mesh, Union, Intersection, Difference,
)
_TYPES_BY_KIND = {cls.kind: cls for cls in _GEOMETRY_TYPES}


def geometry_to_dict(geometry, arrays=None) -> dict:
    """Serialize constructor fields, optionally storing large tensors in ``arrays``."""
    return _geometry_to_dict(geometry, arrays, "geometry")


def _geometry_to_dict(geometry, arrays, path):
    cls = type(geometry)
    if cls not in _GEOMETRY_TYPES:
        raise TypeError(f"not in witwin.scene/1: {cls.__name__}")
    result = {"type": cls.kind}
    for name in signature(cls).parameters:
        if name == "device":
            continue
        value = getattr(geometry, name)
        if name == "rotation":
            value = _rotation_quaternion(value)
        if name == "smoothness" and not geometry.smooth:
            # A hard boolean was authored with a Python number <= 0; a tensor would turn it smooth.
            value = value.item()
        result[name] = _encode(value, arrays, f"{path}/{name}")
    return result


def _encode(value, arrays, path):
    if isinstance(value, GeometryBase):
        return _geometry_to_dict(value, arrays, path)
    if isinstance(value, torch.Tensor):
        if arrays is not None and value.numel() > 4096:
            arrays[path] = value
            return {"array": path}
        return _tensor_to_dict(value)
    if isinstance(value, (tuple, list)):
        return [_encode(item, arrays, f"{path}/{index}") for index, item in enumerate(value)]
    return value


def _tensor_to_dict(tensor) -> dict:
    """Inline tensor: dtype name plus nested values that JSON writes at full precision."""
    return {"type": "Tensor", "dtype": str(tensor.dtype).removeprefix("torch."),
            "values": tensor.detach().cpu().tolist()}


def _tensor_from_dict(d, device=None) -> torch.Tensor:
    return torch.tensor(d["values"], dtype=getattr(torch, d["dtype"]), device=device)


def _legacy_tensor_value(value, device=None):
    """Read a bare number or numeric list from a document written without tensor dtypes.

    Those writers emitted every tensor element as a Python float, which JSON stores at full
    precision. A value that float32 cannot represent exactly came from a float64 tensor and is
    rebuilt as one; every other value is returned unchanged, so the constructor converts it
    exactly as before (Python floats to float32, integer lists to int64).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, list)) or not _numeric(value):
        return value
    wide = torch.tensor(value, dtype=torch.float64)
    if torch.equal(wide.to(torch.float32).to(torch.float64), wide):
        return value
    return wide.to(device=device)


def _numeric(value):
    if isinstance(value, list):
        return all(_numeric(item) for item in value)
    return type(value) in (int, float)


def _is_tensor_dict(value):
    return isinstance(value, dict) and value.get("type") == "Tensor"


def geometry_from_dict(d, arrays=None, device=None):
    """Rebuild geometry through its constructor; mesh paths are provenance only."""
    cls = _TYPES_BY_KIND[d["type"]]
    legacy = not any(_is_tensor_dict(d[name]) for name in signature(cls).parameters if name in d)
    values = {}
    for name in signature(cls).parameters:
        if name == "device":
            continue
        values[name] = _decode(d[name], arrays, device)
        if legacy:
            values[name] = _legacy_tensor_value(values[name], device)
    if cls is Mesh:
        # Host arrays run the host topology checks, and the host path stores float32 vertices.
        # Other vertex dtypes were tensor-authored and stay tensors; an explicit False skips checks.
        vertices = torch.as_tensor(values["vertices"])
        host = d["topology_diagnostics"] and vertices.dtype == torch.float32
        for name in ("vertices", "faces"):
            values[name] = (torch.as_tensor(values[name]).cpu().numpy() if host
                            else torch.as_tensor(values[name], device=device))
    args = []
    kwargs = {"device": device}
    for name, parameter in signature(cls).parameters.items():
        if name == "device":
            continue
        if parameter.kind == Parameter.VAR_POSITIONAL:
            args.extend(values[name])
        elif parameter.kind in {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}:
            args.append(values[name])
        else:
            kwargs[name] = values[name]
    return cls(*args, **kwargs)


def _decode(value, arrays, device):
    if isinstance(value, dict):
        if "array" in value:
            tensor = arrays[value["array"]]
            return tensor if device is None else tensor.to(device=device)
        if _is_tensor_dict(value):
            return _tensor_from_dict(value, device)
        return geometry_from_dict(value, arrays=arrays, device=device)
    if isinstance(value, list):
        return [_decode(item, arrays, device) for item in value]
    return value
