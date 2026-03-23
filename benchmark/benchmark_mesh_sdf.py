from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from witwin.core import Box, Mesh


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _asset_dir() -> Path:
    return _repo_root() / "maxwell" / "tests" / "assets"


def _grid_plane_mesh(*, resolution: int = 48, size: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(-size / 2.0, size / 2.0, resolution + 1, dtype=torch.float32)
    vertices = []
    for iy in range(resolution + 1):
        for ix in range(resolution + 1):
            vertices.append((float(axis[ix]), float(axis[iy]), 0.0))

    faces = []
    stride = resolution + 1
    for iy in range(resolution):
        for ix in range(resolution):
            v00 = iy * stride + ix
            v10 = v00 + 1
            v01 = v00 + stride
            v11 = v01 + 1
            faces.append((v00, v01, v10))
            faces.append((v10, v01, v11))

    return torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.int64)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_forward(fn, *, device: torch.device, warmup: int, repeat: int) -> tuple[float, float]:
    for _ in range(warmup):
        _ = fn()
        _sync(device)

    times_ms: list[float] = []
    peaks_mb: list[float] = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        _ = fn()
        _sync(device)
        end = time.perf_counter()
        times_ms.append((end - start) * 1.0e3)
        if device.type == "cuda":
            peaks_mb.append(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))

    peak_mb = max(peaks_mb) if peaks_mb else 0.0
    return statistics.mean(times_ms), peak_mb


def _measure_forward_backward(fn, params, *, device: torch.device, warmup: int, repeat: int) -> tuple[float, float]:
    for _ in range(warmup):
        for param in params:
            if param.grad is not None:
                param.grad.zero_()
        loss = fn().sum()
        loss.backward()
        _sync(device)

    times_ms: list[float] = []
    peaks_mb: list[float] = []
    for _ in range(repeat):
        for param in params:
            if param.grad is not None:
                param.grad.zero_()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        loss = fn().sum()
        loss.backward()
        _sync(device)
        end = time.perf_counter()
        times_ms.append((end - start) * 1.0e3)
        if device.type == "cuda":
            peaks_mb.append(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))

    peak_mb = max(peaks_mb) if peaks_mb else 0.0
    return statistics.mean(times_ms), peak_mb


def _format_row(case: str, triangles: int | str, points: int, forward_ms: float, backward_ms: float | None, peak_mb: float) -> str:
    backward_text = "-" if backward_ms is None else f"{backward_ms:8.3f}"
    return f"| {case:<30} | {str(triangles):>9} | {points:>8} | {forward_ms:>10.3f} | {backward_text} | {peak_mb:>11.1f} |"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark core primitive and mesh SDF occupancy paths.")
    parser.add_argument("--device", default="cuda", help="torch device, default: cuda")
    parser.add_argument("--warmup", type=int, default=2, help="warmup iterations per case")
    parser.add_argument("--repeat", type=int, default=5, help="timed iterations per case")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    asset_dir = _asset_dir()
    beta = 0.02

    print("| case                           | triangles |   points | forward ms | fwd+bwd ms | peak mem MB |")
    print("| ------------------------------ | ---------:| -------: | ---------: | ---------: | ----------: |")

    primitive_axis = torch.linspace(-0.8, 0.8, 96, device=device, dtype=torch.float32)
    primitive_box = Box(position=(0.0, 0.0, 0.0), size=(0.8, 0.6, 0.4), device=device)
    primitive_points = int(primitive_axis.numel() ** 3)
    primitive_forward_ms, primitive_peak_mb = _measure_forward(
        lambda: primitive_box.to_mask(primitive_axis, primitive_axis, primitive_axis, beta=beta),
        device=device,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    print(_format_row("primitive_box_forward", "-", primitive_points, primitive_forward_ms, None, primitive_peak_mb))

    cube_mesh = Mesh.from_obj(asset_dir / "cube.obj", fill_mode="solid", recenter=True, device=device)
    cube_axis = torch.linspace(-1.0, 1.0, 96, device=device, dtype=torch.float32)
    cube_points = int(cube_axis.numel() ** 3)
    cube_forward_ms, cube_peak_mb = _measure_forward(
        lambda: cube_mesh.to_mask(cube_axis, cube_axis, cube_axis, beta=beta),
        device=device,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    print(_format_row("mesh_cube_static_forward", cube_mesh.face_count, cube_points, cube_forward_ms, None, cube_peak_mb))

    teapot_mesh = Mesh.from_obj(asset_dir / "teapot.obj", fill_mode="solid", recenter=True, device=device)
    teapot_axis = torch.linspace(-1.2, 1.2, 64, device=device, dtype=torch.float32)
    teapot_points = int(teapot_axis.numel() ** 3)
    teapot_forward_ms, teapot_peak_mb = _measure_forward(
        lambda: teapot_mesh.to_mask(teapot_axis, teapot_axis, teapot_axis, beta=beta),
        device=device,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    print(_format_row("mesh_teapot_static_forward", teapot_mesh.face_count, teapot_points, teapot_forward_ms, None, teapot_peak_mb))

    plane_vertices, plane_faces = _grid_plane_mesh(resolution=48, size=1.0)
    plane_mesh = Mesh(plane_vertices, plane_faces, fill_mode="surface", recenter=False, device=device)
    plane_x = torch.linspace(-0.8, 0.8, 64, device=device, dtype=torch.float32)
    plane_y = torch.linspace(-0.8, 0.8, 64, device=device, dtype=torch.float32)
    plane_z = torch.linspace(-0.2, 0.2, 16, device=device, dtype=torch.float32)
    plane_points = int(plane_x.numel() * plane_y.numel() * plane_z.numel())
    plane_forward_ms, plane_peak_mb = _measure_forward(
        lambda: plane_mesh.to_mask(plane_x, plane_y, plane_z, beta=beta),
        device=device,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    print(_format_row("mesh_plane_medium_forward", plane_mesh.face_count, plane_points, plane_forward_ms, None, plane_peak_mb))

    trainable_vertices = torch.nn.Parameter(teapot_mesh._vertices_tensor.detach().clone())
    trainable_mesh = Mesh(
        trainable_vertices,
        teapot_mesh._faces_tensor.detach().clone(),
        fill_mode="solid",
        recenter=teapot_mesh.recenter,
        device=device,
    )
    trainable_forward_ms, _ = _measure_forward(
        lambda: trainable_mesh.to_mask(teapot_axis, teapot_axis, teapot_axis, beta=beta),
        device=device,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    trainable_fwd_bwd_ms, trainable_peak_mb = _measure_forward_backward(
        lambda: trainable_mesh.to_mask(teapot_axis, teapot_axis, teapot_axis, beta=beta),
        params=[trainable_vertices],
        device=device,
        warmup=max(1, args.warmup),
        repeat=max(1, args.repeat // 2),
    )
    print(_format_row("mesh_teapot_trainable", trainable_mesh.face_count, teapot_points, trainable_forward_ms, trainable_fwd_bwd_ms, trainable_peak_mb))


if __name__ == "__main__":
    main()
