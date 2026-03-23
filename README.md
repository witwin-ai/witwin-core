# WiTwin Core

WiTwin Core is the core package of the WiTwin stack. It provides the shared foundations used across WiTwin projects, with a focus on common scene representation, geometry primitives, mesh utilities, and state management that can be reused by different simulation modules.

## Get Started

Python 3.10+ and an NVIDIA GPU are required for the main WiTwin simulation workflows.

```bash
pip install witwin
```

## What It Provides

- Shared scene and geometry primitives in `witwin.core`
- Common state management for simulation-facing scene objects
- Reusable mesh, structure, and material building blocks for WiTwin solvers

## Related Solvers

- [WiTwin Maxwell](../maxwell/README.md)
- [WiTwin Radar](../radar/README.md)
