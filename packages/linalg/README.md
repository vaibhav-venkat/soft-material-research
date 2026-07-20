# linalg

Backend-neutral Zig 0.17 nightly wrappers for dense BLAS and LAPACK operations.

The initial implementation targets Apple's Accelerate framework. Select the backend explicitly:

```bash
zig build -Dbackend=accelerate check
zig build -Dbackend=accelerate test
```

Matrices use column-major storage. The CUDA backend contract is reserved for a separate CUDA 12.9+ implementation.
