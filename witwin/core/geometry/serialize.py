"""Constructor-based geometry dictionaries for ``witwin.scene/1``."""

from __future__ import annotations

from inspect import Parameter, signature

import torch

from .base import GeometryBase, _rotation_quaternion
from .boolean import Difference, Intersection, Union
from .mesh import Mesh
from .primitives import Box, Cone, Cylinder, Ellipsoid, HollowBox, Prism, Pyramid, Sphere, Torus


_GEOMETRY_TYPES = (
    Box, Sphere, Cylinder, Ellipsoid, Cone, Pyramid, Prism, Torus, HollowBox,
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
        result[name] = _encode(value, arrays, f"{path}/{name}")
    return result


def _encode(value, arrays, path):
    if isinstance(value, GeometryBase):
        return _geometry_to_dict(value, arrays, path)
    if isinstance(value, torch.Tensor):
        if arrays is not None and value.numel() > 4096:
            arrays[path] = value
            return {"array": path}
        return value.detach().cpu().tolist()
    if isinstance(value, (tuple, list)):
        return [_encode(item, arrays, f"{path}/{index}") for index, item in enumerate(value)]
    return value


def geometry_from_dict(d, arrays=None, device=None):
    """Rebuild geometry through its constructor; mesh paths are provenance only."""
    cls = _TYPES_BY_KIND[d["type"]]
    args = []
    kwargs = {"device": device}
    for name, parameter in signature(cls).parameters.items():
        if name == "device":
            continue
        value = _decode(d[name], arrays, device)
        if cls is Mesh and name in {"vertices", "faces"}:
            # Mesh stores the resolved diagnostics flag. Host arrays always run the
            # (host-only) topology checks; device tensors with an explicit False skip them.
            value = (torch.as_tensor(value).cpu().numpy() if d["topology_diagnostics"]
                     else torch.as_tensor(value, device=device))
        if parameter.kind == Parameter.VAR_POSITIONAL:
            args.extend(value)
        elif parameter.kind in {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}:
            args.append(value)
        else:
            kwargs[name] = value
    return cls(*args, **kwargs)


def _decode(value, arrays, device):
    if isinstance(value, dict):
        if "array" in value:
            tensor = arrays[value["array"]]
            return tensor if device is None else tensor.to(device=device)
        return geometry_from_dict(value, arrays=arrays, device=device)
    if isinstance(value, list):
        return [_decode(item, arrays, device) for item in value]
    return value
