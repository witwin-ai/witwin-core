# WiTwin Core

WiTwin Core is the shared data and geometry package of the WiTwin stack. It defines the geometry and material contracts consumed by solver packages without imposing a common solver-specific scene lifecycle.

## Get Started

CPython 3.10-3.14, PyTorch 2.10-2.11, and an NVIDIA GPU are supported for the main WiTwin simulation workflows.

```bash
pip install witwin
```

## Prebuilt CUDA Support

Release wheels are built for Linux x86_64 and Windows x86_64 with CUDA 12.8. Each wheel carries separate PyTorch 2.10/cu128 and 2.11/cu128 native-extension variants, selected at runtime by the active PyTorch ABI. The fat binaries contain native code for compute capabilities 7.0, 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, 10.1, and 12.0, plus compute 12.0 PTX for forward-compatible Blackwell execution. This includes native coverage for RTX 2080-class Turing GPUs and current data-center and RTX/RTX PRO Blackwell families.

Linux wheels target `manylinux_2_35_x86_64`. The installed NVIDIA driver must support the CUDA 12.x runtime supplied by PyTorch; the CUDA toolkit is only needed for source/JIT builds.

For full CUDA 12.8 and Blackwell support, use at least driver 570.26 on Linux or 570.65 on Windows. Pre-Blackwell systems can use NVIDIA's CUDA 12.x minor-version compatibility floor (525.60.13 on Linux or 528.33 on Windows), subject to NVIDIA's compatibility-mode feature limits.

## What It Provides

- Runtime-checkable `GeometrySpec` and `MaterialSpec` contracts
- Shared analytic geometry primitives and differentiable mesh/SDF utilities
- Reusable `Material` and `Structure` values for solver-owned scenes

Each solver owns its public `Scene` implementation. Radar-specific SMPL geometry lives in `witwin.radar.geometry`; Maxwell-specific `PolySlab` geometry lives in `witwin.maxwell.geometry`.

## Related Solvers

- [WiTwin Maxwell](https://github.com/witwin-ai/witwin-maxwell)
- [WiTwin Radar](https://github.com/witwin-ai/witwin-radar)

## Developer

<a href="http://xingyuchen.me/">
  <img src="https://github.com/Asixa.png" alt="Xingyu Chen" width="48" height="48" style="border-radius:50%;">
</a>

[Xingyu Chen](http://xingyuchen.me/)
