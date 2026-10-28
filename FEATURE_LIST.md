# Core feature inventory

- Shared logical scenes, snapshots, material, antenna and motion contracts remain solver neutral.
- Analytic primitives and mesh/SDF utilities retain native PyTorch tensor workflows.
- Primitive position, size, Euler and quaternion sequences preserve nested Tensor ancestry, promoted dtype and inferred device; direct Tensor inputs keep their existing behavior.
- Scalar Python inputs retain their established geometry representation. Nonscalar coordinate components and conflicting tensor devices reject explicitly.
- CPU coverage includes nested-versus-stacked gradients, float64, metadata-only device contracts, serialization and SDF finite differences. Genuine CUDA geometry and solver tests remain separate checks.
