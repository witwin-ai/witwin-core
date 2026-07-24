# WiTwin Core

WiTwin Core is the shared logical-world and geometry package of the WiTwin stack. It defines solver-neutral scene, material, antenna, dynamics, snapshot, and geometry contracts. Solver packages compile those contracts into their own runtime resources and results.

## Get Started

CPython 3.10-3.14, PyTorch 2.10 or newer, and an NVIDIA GPU are supported for the main WiTwin simulation workflows.

```bash
pip install witwin
```

## Prebuilt CUDA Support

Release wheels are built for Linux x86_64 and Windows x86_64 with CUDA 12.8. Each platform wheel carries one PyTorch Stable ABI native library built against PyTorch 2.10 and reused across supported Python and PyTorch versions; CI currently verifies PyTorch 2.10-2.12. The fat binaries contain native code for compute capabilities 7.0, 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, 10.1, and 12.0, plus compute 12.0 PTX for forward-compatible Blackwell execution. This includes native coverage for RTX 2080-class Turing GPUs and current data-center and RTX/RTX PRO Blackwell families.

Linux wheels target `manylinux_2_35_x86_64`. The installed NVIDIA driver must support the CUDA 12.x runtime supplied by PyTorch; the CUDA toolkit is only needed for source/JIT builds.

For full CUDA 12.8 and Blackwell support, use at least driver 570.26 on Linux or 570.65 on Windows. Pre-Blackwell systems can use NVIDIA's CUDA 12.x minor-version compatibility floor (525.60.13 on Linux or 528.33 on Windows), subject to NVIDIA's compatibility-mode feature limits.

## What It Provides

- Canonical `Scene`, `SceneSnapshot`, `Structure`, and stable logical IDs
- Solver-neutral `PhysicalMaterial`, layers, roughness, dispersion, and phase-screen assignments
- Logical TX/RX antenna state, rigid motion, trajectories, and deformation state
- Runtime-checkable `GeometrySpec` and `MaterialSpec` contracts
- Shared analytic geometry primitives and differentiable mesh/SDF utilities
- Granular topology, geometry, material, and assignment versions

Core owns the logical `Scene`; each solver owns its compiled scene lifecycle, native resources, caches, and results. Radar-specific SMPL geometry lives in `witwin.radar.geometry`; Maxwell-specific `PolySlab` geometry lives in `witwin.maxwell.geometry`.

## 0.4 Migration

- `Mesh.vertices`, `Mesh.faces`, `Mesh.world_vertices`, and
  `Mesh.bounds_world` are tensor-valued contracts. `bounds_world` has shape
  `(2, 3)`, with minimum coordinates in row 0 and maximum coordinates in row 1.
- Tensor-authored geometry is never implicitly moved to the requested device.
  Place authored tensors explicitly before construction; conflicting devices
  fail loudly.
- CUDA-authored meshes do not run host topology diagnostics or build a host BVH.
  `fill_mode="auto"` therefore requires an explicit topology decision before
  an SDF query.

## Related Solvers

- [WiTwin Maxwell](https://github.com/witwin-ai/witwin-maxwell)
- [WiTwin Radar](https://github.com/witwin-ai/witwin-radar)

## Developer

<a href="http://xingyuchen.me/">
  <img src="https://github.com/Asixa.png" alt="Xingyu Chen" width="48" height="48" style="border-radius:50%;">
</a>

[Xingyu Chen](http://xingyuchen.me/)
