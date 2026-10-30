# Geometry dicts for `witwin.scene/1` (witwin-core part)

Design (approved by the user 2026-10-30):
`/home/xingyu/code/witwin/instamax/docs/plans/mw_scene_tokenization.md` §2. This repo provides the
geometry layer; witwin-maxwell builds `Scene.to_dict/from_dict` on top of it in a separate change.

## Scope

`geometry_to_dict(geometry, arrays=None) -> dict` and `geometry_from_dict(d, arrays=None, device=None)`
in a new module `witwin/core/geometry/serialize.py`, exported from `witwin.core`. Also add thin
`GeometryBase.to_dict(arrays=None)` and `GeometryBase.from_dict(d, arrays=None, device=None)` (a
classmethod that dispatches on `d["type"]`), both delegating to the module.

Types: every `GeometryBase` subclass in `witwin/core/geometry/` today: Box, Sphere, Cylinder,
Ellipsoid, Cone, Pyramid, Prism, Torus, HollowBox, Mesh, and the boolean geometries Union,
Intersection and Difference, which serialize their operand geometries recursively. Any other
subclass raises `TypeError("not in witwin.scene/1: <class>")`.

## Format

- `{"type": "<kind>", "position": [x, y, z], "rotation": [w, x, y, z], ...type fields}`.
  `type` is the class's `kind` string. The rotation is exactly the quaternion that
  `_rotation_quaternion(self.rotation)` returns, whatever form the user passed. Do not normalize it
  further, so existing behaviour for non-unit inputs is unchanged.
- Type fields are exactly the class's constructor parameters (`inspect.signature`), except
  `position`, `rotation` and `device`, read from the stored attributes, with every value written
  even when it equals the default. No hand-maintained key lists: the serializer derives the key
  set from the signature, and a parameter without a readable attribute is an error found by the
  round-trip test, not skipped.
- Floats are Python floats, vectors are lists, tensors are lists. With `arrays` (a dict) given,
  any tensor with more than 4096 elements is stored as `{"array": "<key>"}` and the tensor goes to
  `arrays[key]` (keys unique within one call, e.g. a path such as `"geometry/0/vertices"`).
  `from_dict` resolves the same references from `arrays`.
- Mesh: serialize the raw authored vertices and the actual constructor fields (scale, recenter,
  position, rotation, fill_mode, surface_thickness, topology_diagnostics), so `from_dict` reruns
  the same constructor path and rebuilds identical geometry. `source_path` is written as provenance
  only; it is not re-read.
- The output is JSON-safe: `json.dumps(d)` works when `arrays` is None.

## Acceptance tests (new file tests/test_geometry_serialize.py, nothing else)

1. `test_round_trip_every_geometry`: one non-default instance of every listed type, plus a boolean
   tree (a Difference of a Union of two primitives and a Mesh) with non-identity rotations and
   positions. Checks:
   - `from_dict(json.loads(json.dumps(to_dict(g))))` has the same type;
   - every constructor-derived field is equal;
   - SDF values on 1000 random points agree to 1e-6;
   - `to_mesh()` vertices and faces are equal.
2. `test_every_constructor_parameter_is_serialized`: for each type, the dict keys equal the
   constructor parameter names minus {"device"} plus {"type"}. A parameter added to a constructor
   later must make this test fail.
3. `test_large_arrays_by_reference`: a mesh with more than 4096 vertex coordinates goes into
   `arrays` and round-trips.

## Rules

- No defensive programming: no speculative validation, fallbacks, try/except, or compat shims.
  Fail loudly.
- Minimal tests: only the three above. Do not edit other tests.
- Do not bump the package version and do not touch packaging. Release timing is coordinated
  separately.
- One command per shell call, no `&&`. Python via `/home/xingyu/miniconda3/envs/maxwell/bin/python`.
  For tests, prepend this worktree so it shadows the installed core:
  `env PYTHONPATH=/home/xingyu/code/witwin/wt/core-scene-dict /home/xingyu/miniconda3/envs/maxwell/bin/python -m pytest tests/test_geometry_serialize.py`.
  CPU is enough. If you need a GPU: `env CUDA_VISIBLE_DEVICES=1 ...`, never device 0.
- Commit on branch `scene-dict/geometry` with
  `git -c user.name="Xingyu Chen" -c user.email="xic063@ucsd.edu" commit ...`, no AI attribution.
  Do not push or merge.
- Report: files changed, test output, anything ambiguous.
