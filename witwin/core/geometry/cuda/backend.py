"""Loader for the native CUDA mesh-SDF extension.

Exposes a module object that is call-compatible with the former SlangTorch
module: ``module.<kernelName>(**kwargs).launchRaw(blockSize=..., gridSize=...)``.
The kernels compute their own launch grid, so ``launchRaw`` ignores the block /
grid hints and simply dispatches the compiled function.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


_COMPILED_EXTENSION: Any | None = None


def is_available() -> bool:
    return bool(torch.cuda.is_available())


def get_compiled_extension(*, verbose: bool = False) -> Any:
    global _COMPILED_EXTENSION
    if _COMPILED_EXTENSION is None:
        from .build import build_extension

        _COMPILED_EXTENSION = build_extension(verbose=verbose)
    return _COMPILED_EXTENSION


def _query_mesh_unsigned_distance(*, triangles, points, unsignedDistance, closestTriangleIndex):
    get_compiled_extension().query_mesh_unsigned_distance(
        triangles, points, unsignedDistance, closestTriangleIndex
    )


def _query_mesh_distance_and_winding(*, triangles, points, unsignedDistance, windingAngle, closestTriangleIndex):
    get_compiled_extension().query_mesh_distance_and_winding(
        triangles, points, unsignedDistance, windingAngle, closestTriangleIndex
    )


def _query_mesh_unsigned_distance_bvh(
    *,
    triangles,
    points,
    nodeBBoxMin,
    nodeBBoxMax,
    nodeLeft,
    nodeRight,
    nodeStart,
    nodeCount,
    triangleIndices,
    unsignedDistance,
    closestTriangleIndex,
):
    get_compiled_extension().query_mesh_unsigned_distance_bvh(
        triangles,
        points,
        nodeBBoxMin,
        nodeBBoxMax,
        nodeLeft,
        nodeRight,
        nodeStart,
        nodeCount,
        triangleIndices,
        unsignedDistance,
        closestTriangleIndex,
    )


def _query_mesh_parity_sign_bvh(
    *,
    triangles,
    points,
    nodeBBoxMin,
    nodeBBoxMax,
    nodeLeft,
    nodeRight,
    nodeStart,
    nodeCount,
    triangleIndices,
    jitterScale,
    inside,
):
    get_compiled_extension().query_mesh_parity_sign_bvh(
        triangles,
        points,
        nodeBBoxMin,
        nodeBBoxMax,
        nodeLeft,
        nodeRight,
        nodeStart,
        nodeCount,
        triangleIndices,
        float(jitterScale),
        inside,
    )


def _backward_mesh_unsigned_distance(
    *, triangles, points, closestTriangleIndex, gradUnsignedDistance, gradTriangles, gradPoints
):
    get_compiled_extension().backward_mesh_unsigned_distance(
        triangles, points, closestTriangleIndex, gradUnsignedDistance, gradTriangles, gradPoints
    )


def _backward_mesh_distance_and_winding(
    *,
    triangles,
    points,
    closestTriangleIndex,
    gradUnsignedDistance,
    gradWindingAngle,
    gradTriangles,
    gradPoints,
):
    get_compiled_extension().backward_mesh_distance_and_winding(
        triangles,
        points,
        closestTriangleIndex,
        gradUnsignedDistance,
        gradWindingAngle,
        gradTriangles,
        gradPoints,
    )


_KERNELS: dict[str, Callable[..., None]] = {
    "queryMeshUnsignedDistance": _query_mesh_unsigned_distance,
    "queryMeshDistanceAndWinding": _query_mesh_distance_and_winding,
    "queryMeshUnsignedDistanceBVH": _query_mesh_unsigned_distance_bvh,
    "queryMeshParitySignBVH": _query_mesh_parity_sign_bvh,
    "backwardMeshUnsignedDistance": _backward_mesh_unsigned_distance,
    "backwardMeshDistanceAndWinding": _backward_mesh_distance_and_winding,
}


class _Launch:
    __slots__ = ("_kernel", "_kwargs")

    def __init__(self, kernel: Callable[..., None], kwargs: dict):
        self._kernel = kernel
        self._kwargs = kwargs

    def launchRaw(self, *, blockSize=None, gridSize=None):  # noqa: N802 - kernel launch entry point
        del blockSize, gridSize
        self._kernel(**self._kwargs)
        return None


class NativeMeshSDFModule:
    def __getattr__(self, name: str):
        kernel = _KERNELS.get(name)
        if kernel is None:
            raise AttributeError(f"Native CUDA mesh SDF backend does not implement kernel {name!r}.")

        def bind(**kwargs):
            return _Launch(kernel, kwargs)

        bind.__name__ = name
        object.__setattr__(self, name, bind)
        return bind


_NATIVE_MODULE = NativeMeshSDFModule()


def get_native_mesh_sdf_module(*, verbose: bool = False) -> NativeMeshSDFModule:
    """Compile the extension (once) and return the launch-compatible module."""
    if not is_available():
        raise RuntimeError("Native CUDA mesh SDF backend requires torch.cuda.is_available() to be True.")
    get_compiled_extension(verbose=verbose)
    return _NATIVE_MODULE
