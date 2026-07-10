# WiTwin Core

WiTwin Core is the shared data and geometry package of the WiTwin stack. It defines the geometry and material contracts consumed by solver packages without imposing a common solver-specific scene lifecycle.

## Get Started

Python 3.10+ and an NVIDIA GPU are required for the main WiTwin simulation workflows.

```bash
pip install witwin
```

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
