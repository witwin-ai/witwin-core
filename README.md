# WiTwin Core

WiTwin Core is the shared logical-world and geometry package of the WiTwin stack. It defines solver-neutral scene, material, antenna, dynamics, snapshot, and geometry contracts. Solver packages compile those contracts into their own runtime resources and results.

## Get Started

Core supports CPython 3.10-3.14 and PyTorch 2.10 or newer.

```bash
pip install witwin
```

## Pure Python Distribution

Core ships as a platform-independent `py3-none-any` wheel and contains no C++ or
CUDA sources. Mesh SDF evaluation always has a PyTorch implementation. Solver
packages may register their own optional accelerators; Maxwell owns and packages
the CUDA mesh-SDF implementation used by Maxwell workflows.

## What It Provides

- Canonical `Scene`, `SceneSnapshot`, `Structure`, and stable logical IDs
- Solver-neutral `PhysicalMaterial`, layers, roughness, dispersion, and phase-screen assignments
- Logical TX/RX antenna state, planar receiver grids, rigid motion, trajectories, and deformation state
- Stable structure, surface, material, assignment, primitive, and antenna identities
- Typed mesh UV topology and explicit finite/perfect conductor material identity
- Runtime-checkable `GeometrySpec` and `MaterialSpec` contracts
- Shared analytic geometry primitives and differentiable mesh/SDF utilities
- Granular topology, geometry, material, and assignment versions

Core owns the logical `Scene`; each solver owns its compiled scene lifecycle, native resources, caches, and results. Radar-specific SMPL geometry lives in `witwin.radar.geometry`; Maxwell-specific `PolySlab` geometry lives in `witwin.maxwell.geometry`.

## Stable API

The supported public API is the set of names exported by `witwin.core.__all__`.
Applications should import contracts, geometry types, identity allocators, and
math utilities directly from `witwin.core`. CI locks this symbol set so adding,
removing, or renaming a public symbol requires an intentional API change.
Modules and names outside that export set are implementation details and may be
reorganized between releases.

## Coordinate and Motion Semantics

Antenna orientation maps endpoint-local coordinates into world coordinates.
Euler orientation is `(yaw, pitch, roll)` in radians with intrinsic Z-Y-X
rotations; quaternion orientation is scalar-first `(w, x, y, z)`.
Polarization and antenna element positions are endpoint-local, while
`ReceiverGrid.x_axis` and `y_axis` are world-frame grid directions. Core's
Torch rotation and projection helpers preserve autograd.

Snapshot deformation is composed before motion: absolute deformation vertices
replace authored local mesh vertices, or offsets are added to them; authored
geometry transforms are applied next. `RigidMotion.rotation` then left-composes
the authored rotation and its world-frame translation is added last. Endpoint
motion follows the same rule after antenna-local orientation.

`PhysicalMaterial(conductor_model="perfect")`, or the equivalent
`PhysicalMaterial.perfect_conductor()`, carries explicit PEC identity while
keeping finite logical material parameters. Solver-owned compilers map that
identity to their native material ABI; Core does not encode native model IDs or
effective conductivity constants.

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
- `Material` is removed. `PhysicalMaterial` is the only public name for the
  material contract; the two were aliases of one class. Replace
  `from witwin.core import Material` with
  `from witwin.core import PhysicalMaterial`.
- `Structure` and `MaterialAssignment` now live in `witwin.core.structure`
  rather than `witwin.core.material`, which keeps only material specifications.
  Both remain exported from `witwin.core`, so package-level imports are
  unaffected.
- Continuous physical quantities are annotated `ScalarLike`
  (`float | torch.Tensor`) instead of `Any`. `ScalarLike` and `ComplexLike` are
  exported from `witwin.core`. This is a typing change only; the accepted
  runtime values are unchanged.
- `witwin.core.identity` documents its two allocation modes. Use
  `reserve_*_id(value)` whenever identities must be reproducible across
  processes; `new_*_id()` draws from a process-global counter and is
  order-dependent.

## Related Solvers

- [WiTwin Maxwell](https://github.com/witwin-ai/witwin-maxwell)
- [WiTwin Radar](https://github.com/witwin-ai/witwin-radar)

## Developer

<a href="http://xingyuchen.me/">
  <img src="https://github.com/Asixa.png" alt="Xingyu Chen" width="48" height="48" style="border-radius:50%;">
</a>

[Xingyu Chen](http://xingyuchen.me/)
