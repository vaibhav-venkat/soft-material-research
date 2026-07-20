# linalg

`linalg` is a backend-neutral Zig 0.17 nightly wrapper around vendor BLAS and
LAPACK implementations. It owns API semantics, validation, storage layout, and
tests; a backend driver owns memory and vendor calls.

The current implementation uses Apple's Accelerate framework. A CUDA driver is
reserved behind the same public API for cuBLAS and cuSOLVER.

The source layout follows the numerical boundaries without splitting every
routine into its own file:

- `root.zig`: public facade and scientific guidance for choosing operations.
- `types.zig`: contexts, events, buffers, vectors, matrices, and views.
- `blas.zig`: backend-neutral vector and matrix BLAS validation/dispatch.
- `lapack.zig`: backend-neutral SVD and least-squares ownership/dispatch.
- `tests.zig`: the shared backend-independent numerical contract.
- `backend/accelerate_{core,blas,lapack}.zig`: Accelerate driver layers.

## Build on macOS

Backend selection is intentionally explicit:

```bash
zig build -Dbackend=accelerate check
zig build -Dbackend=accelerate test
```

The default SDK is the Command Line Tools SDK. Override it when Xcode or another
SDK should supply the headers:

```bash
zig build -Dbackend=accelerate \
  -Dmacos-sdk=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
  test
```

The Accelerate LAPACK interface selected here requires macOS 13.3 or newer.
Zig 0.17 no longer uses `@cImport`: `build.zig` translates the stable backend
header into the `accelerate_c` module. A small C source shim keeps Apple's C
complex types and `$NEWLAPACK` symbol aliases out of the Zig-facing ABI.

## Public contract

- Scalars: `f32`, `f64`, `Complex32`, and `Complex64`.
- Complex storage: an extern `{ re, im }` pair with explicit unconjugated and
  conjugated operations (`dotu`/`dotc`, `geruInto`/`gercInto`).
- Storage: dense column-major matrices, plus non-owning row, column, and matrix
  views with explicit strides and leading dimensions.
- Ownership: `Context` owns backend state; `Vector(T)` and `Matrix(T)` own
  backend buffers. Deinitialization synchronizes before releasing memory.
- Execution: calls are ordered through a context. Accelerate completes them
  synchronously; a CUDA context will enqueue them on one owned stream. `Event`
  is the explicit synchronization boundary for future asynchronous consumers.
- Errors: shape, context, stride, alias, size, unsupported wide-matrix, backend,
  and non-convergence failures are reported through Zig error returns.

Implemented wrappers:

- Level 1: copy, swap, scale, AXPY, real/complex dot products, 2-norm, absolute
  sum, and absolute-maximum index.
- Matrix BLAS: GEMV, real/complex GER, and GEMM, with allocating conveniences
  for matrix-vector and matrix-matrix products.
- LAPACK: thin or values-only SVD, and overdetermined/square least squares with
  multiple right-hand sides, numerical rank, singular values, and `rcond`.

SVD and least squares currently accept tall or square coefficient matrices.
LAPACK-destructive inputs are copied, so caller-owned matrices are preserved.

## CUDA implementation contract

The CUDA implementation should replace `src/backend/cuda.zig` without changing
`src/root.zig` or its tests. Its build branch should require an explicit CUDA
root (for example `-Dcuda-root=/usr/local/cuda`), translate a narrow shim header
from `build.zig`, compile its shim source, and link `cudart`, `cublas`, and
`cusolver`. Do not restore `@cImport`.

Each CUDA `Context` should own one `cudaStream_t`, one `cublasHandle_t`, and one
`cusolverDnHandle_t`, with both handles bound to that stream. `Buffer(T)` should
own device memory; host transfers, BLAS calls, LAPACK-equivalent calls, events,
and synchronization should preserve the public ordering and error semantics.
Validate `Complex32`/`Complex64` layout against `cuComplex`/`cuDoubleComplex` in
the shim before using pointer casts.

Map the operations as follows:

| Public operation | Accelerate | CUDA |
| --- | --- | --- |
| Level 1, GEMV, GER, GEMM | CBLAS | cuBLAS |
| thin/values-only SVD | `xGESDD` | cuSOLVER `gesvd`/`gesvdj` adapter |
| least squares | `xGELSS` | cuSOLVER SVD plus rank-filtered solve |
| synchronize/event | synchronous/no-op event | stream/event |

The package-local tests intentionally contain no Accelerate symbols or macOS
conditionals. Run the same tests with `-Dbackend=cuda`; backend-specific tests
should be limited to driver initialization and error translation.
