# Core feature inventory

- Release 0.4.3 fixes `to_mesh()` axis remapping and outward face winding for Cone, Cylinder, Pyramid, Prism and Torus, including Prism x-axis mesh agreement with its signed distance. Z-axis vertices, faces and ordering remain bit-identical to 0.4.2.

- Release 0.4.2 adds signed-distance boolean geometry composition (`Union`, `Intersection`, `Difference`, `BooleanGeometry`) and binary/ASCII STL import (`Mesh.from_stl`). Both are additive exports; existing geometry, mesh and scene contracts are unchanged, and the optional Maxwell integration still selects `witwin-maxwell>=0.5.1,<0.6`. Local wheel validation does not change an installed environment.

- Release 0.4.1 includes nested-Tensor geometry ancestry and fixed assignment-control validation; the optional Maxwell integration selects `witwin-maxwell>=0.5.1,<0.6`. Its fourteen core source files preserve the qualified implementation, with only the module version constant changed. Local wheel validation does not change an installed environment.

- Shared logical scenes, snapshots, material, antenna and motion contracts remain solver neutral.
- Analytic primitives and mesh/SDF utilities retain native PyTorch tensor workflows.
- Primitive position, size, Euler and quaternion sequences preserve nested Tensor ancestry, promoted dtype and inferred device; direct Tensor inputs keep their existing behavior.
- Scalar Python inputs retain their established geometry representation. Nonscalar coordinate components and conflicting tensor devices reject explicitly.
- CPU coverage includes nested-versus-stacked gradients, float64, metadata-only device contracts, serialization and SDF finite differences. Genuine CUDA geometry and solver tests remain separate checks.

- Structure priority and enabled controls reject trainable Tensor ancestry before discrete conversion, including nested/object-array containers; fixed controls preserve native geometry and material gradients. CPU155 and four original CUDA contracts pass on the same proposed core bytes.

- Boolean geometry composition: `Union`, `Intersection` and `Difference` (base minus a union of tools) combine child signed distance fields with exact min/max, or with a compact-support quadratic soft minimum when `smoothness` (length, float or trainable tensor) is set. Nodes carry primitive-style `position`/`rotation` frames, expose `bounds_world`, nest freely, and surface every child tensor as a node tensor attribute so gradients and solver parameter scans reach the child parameters. A sharp `Union` is the geometry group. `to_mesh(resolution=...)` extracts a closed, outward-oriented, welded surface by marching tetrahedra (an approximation that chamfers sharp edges at the lattice scale; no exact mesh boolean).

- `Mesh.from_stl(path, unit_scale=..., weld_tolerance=...)` loads binary and ASCII STL with a pure NumPy parser (binary detected by record size, polygon facets fan-triangulated), welds coincident facet corners (exact by default, lattice tolerance in file units on request) and drops collapsed facets so topology diagnostics see closed surfaces. `unit_scale` folds into `Mesh.scale`, keeping diagnostics at file scale; other arguments follow `Mesh.from_obj`.
