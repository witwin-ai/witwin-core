# Nested tensor geometry constructor repair

This is a private CPU-qualified candidate. Neither repository checkout, the installed dependency, public mapped AD admission, nor earlier numerical archives were modified. No GPU was used. Only the root agent may integrate and commit.

## Cause and repair

Installed `witwin` 0.4.0, source SHA `b063986f4cbcc270b80c1d963cc983696f5b1c755826058220a6be1f0213a784`, preserves a direct float64 Tensor in Box position/size/rotation. Its sequence path instead calls `float(component)` before constructing a float32 tensor. Consequently `Box(position=(theta, 0., 0.))` silently loses ancestry while `Box(position=torch.stack((theta, theta*0, theta*0)))` retains it. This can hide a prohibited geometry dependency sharing a registered epsilon leaf from a later semantic admission census.

The installed-contract patch modifies only `witwin/core/geometry/base.py`:

- Recursively inspect tuple/list and NumPy object-array Tensor leaves for device resolution. This reads metadata only and performs no value extraction or device synchronization.
- Stack scalar Tensor components with PyTorch operations. Promote among authored Tensor dtypes; Python constants adopt that promoted dtype. Reject nonscalar Tensor components rather than converting them with `float`.
- Apply the same rule to position, vector dimensions, Euler tuples and quaternion tuples. Tensor-authored Euler triples remain triples, matching the existing direct-Tensor representation and its deferred quaternion conversion.
- Keep direct Tensor identity, float64, shape checks, explicit device conflict rejection and `device=None` inference. Python-only inputs retain their existing float32 conversions, including scalar-size expansion and Euler-to-quaternion behavior.

This is a constructor ancestry fix, not geometry differentiation support for mapped grids. Existing mapped admission remains closed. Arbitrary user Python `.item()`/`.detach()` remains the ordinary user-code boundary and is outside this repair.

## Installed source versus checkout

The clean `witwin-core` checkout is `78aad439865448b9f213e78cae4baa807b57d3be`, version 0.3.0. Installed code is version 0.4.0 and has no installed `direct_url.json`; its build source commit cannot be inferred from version metadata. The detailed per-file comparison is `installed-vs-checkout.json`.

The differences include logical-world/material APIs and mesh runtime changes, as well as existing geometry dtype/device semantics. Replacing the installed dependency with the current 0.3 checkout would regress current Maxwell behavior. In particular the checkout casts direct geometry Tensors to float32, lacks cross-field device resolution and eagerly converts Euler Tensors. This repair must land in the shared **core** implementation; a second helper in Maxwell would leave other users exposed.

Preferred landing order:

1. Reconcile the core source/release corresponding to the actual 0.4.0 dependency. Apply `installed-fix-and-contracts.patch` there (one production file and one CPU test file). Publish/install a reviewed core build through the normal dependency workflow, not by editing site-packages.
2. Add `maxwell-integration-contract.patch` to Maxwell and rerun its Scene/material/geometry gradient regressions against that exact core artifact. Maxwell needs no production conversion shim or blanket prohibition of existing Cartesian Tensor geometry AD.
3. If the available 0.3 checkout must be the landing base, `checkout-reconciliation-and-contracts.patch` is a separately qualified, narrower alternative: preserve the installed base-helper contracts, propagate device resolution across the nine primitive constructors, and defer Euler conversion at the old Mesh rotation-matrix call. It does not import the unrelated 0.4 logical-world APIs or claim that the entire old Mesh runtime has been upgraded to 0.4 semantics. Full dependency/release reconciliation remains necessary before using that checkout as a replacement for installed 0.4 in Maxwell.

The candidate installed package SHA is `373afab7ff14ad516f534fbcea05829ca9f24d0cb20452e3e8c3b5336b9ad8a4`. The separately reconciled 0.3 candidate SHA is `e3adba72865ac6d7eca7947972c9557aa39e1ac8586d8d99e0844c267af5ccc2`.

## Bound CPU evidence

All runs hid CUDA (`CUDA_VISIBLE_DEVICES=''`) and finished with `torch.cuda.is_initialized()==False`. JSON records actual imported source paths, all package `.py/.cu/.cpp/.h/.cuh` file hashes, tests/controllers, original installed and checkout identities before/after, Python and thread settings. Source identities are unchanged across each run. JUnit time and measured pytest-call elapsed time are reported separately.

| Run | Result | JUnit seconds | Purpose |
| --- | --- | --- | --- |
| `installed-baseline-focused` | 10 pass, **39 fail**, no skips | 1.963 | Same 49 tests, real unmodified installed source; retained failure evidence |
| `installed-candidate-focused` | **49 pass**, no skips | 1.201 | Position/size/Euler/quaternion tuple/list/object-array value and gradient equality; shared epsilon leaf; dtype promotion; direct Tensor identity; float-only compatibility; CPU/meta device contracts; malformed components; serialization; independent SDF finite-difference gradcheck |
| `checkout-reconciled-candidate-focused` | **49 pass**, no skips | 1.118 | Same substantive contracts on the separately reconciled 0.3 source |
| `installed-baseline-existing` | 39 pass, 1 existing API mismatch, 6 CUDA-only deferred | 1.361 | Unchanged old checkout tests against installed 0.4 |
| `installed-candidate-existing` | 39 pass, same 1 API mismatch, 6 CUDA-only deferred | 1.507 | Confirms no added failure in those tests |
| `checkout-reconciled-candidate-existing` | **40 pass**, 6 CUDA-only deferred | 1.511 | Existing core contracts, primitive SDF and mesh smoke tests against reconciled checkout |
| `maxwell-integration` | **101 pass**, no skips | 0.799 | Seven actual Maxwell Scene/CPU material compiler ancestry and stack-equivalence tests plus the 94 previously merged constructor contracts |

Each prefix has `.json`, `.log` and `.xml` files in this directory. The single old-test mismatch is `test_mesh_dynamic_views_and_bounds_follow_tensor_updates`: the 0.3 test calls `world_vertices_tensor()`, an API absent in installed 0.4. It occurs identically before and after this patch. No test was skipped or changed to hide it.

The Maxwell integration source is a fresh committed archive `b27570b9293090d077060b18a7c4a656af247ee4`, with only the new seven-case integration test overlay. It imports the private candidate core. The seven cases compile actual CPU material arrays and compare nested versus stacked VJPs, with separate and shared epsilon ancestry; the zero-numerical-derivative case proves the structural dependency survives for later admission policy. No field solve or solver AD acceptance is implied.

Serialization uses the existing PyTorch object contract (`torch.save` / `torch.load(weights_only=False)` on controlled local data): values, dtype, device and trainability survive. As with ordinary stacked tensors, the original Python autograd graph is not serialized; this stage does not invent such a guarantee.

A first integration setup attempt requested nonexistent `pytest.ini` in `git archive`; extraction failed before creating the archive, and the runner import-path assertion then rejected the unfrozen installed Maxwell path before any tests. The corrected archive uses the tracked `pyproject.toml`. This was a controller setup error, not a numerical test result; all qualified evidence above binds the corrected archive before execution.

## Remaining validation and capability accounting

- Run existing genuine CUDA geometry/material/SceneModule adjoint controls after a reviewed core dependency release. CPU metadata-device controls do not substitute for GPU placement or CUDA kernels.
- Rerun actual fixed-kappa epsilon/sigma/Cu ordinary-forward FD and Graph/resume controls only under root's durable queue and immutable complete source identities.
- No `NotImplementedError` site was added or retired. New malformed-component failures are `ValueError`. The mapped guard and its census entry are untouched.
- Suggested core feature entry: "Tensor components inside primitive position, size and rotation sequences preserve autograd, promoted Tensor dtype and inferred device; mismatched devices and nonscalar components reject explicitly."
- Suggested Maxwell feature note: "Shared core geometry sequence ancestry is preserved through CPU Scene placement and material compilation; mapped geometry gradients remain unsupported."
