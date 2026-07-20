# linalg

`linalg` is a backend-neutral Zig 0.17 nightly wrapper around vendor BLAS and
LAPACK implementations. It owns API semantics, validation, storage layout, and
tests; a backend driver owns memory and vendor calls.

The package implements Apple's Accelerate framework and NVIDIA CUDA 12.9+
behind the same public API. Backend selection is explicit and never falls back.

The source layout follows the numerical boundaries without splitting every
routine into its own file:

- `root.zig`: public facade and scientific guidance for choosing operations.
- `types.zig`: contexts, events, buffers, vectors, matrices, and views.
- `blas.zig`: backend-neutral vector and matrix BLAS validation/dispatch.
- `lapack.zig`: backend-neutral SVD and least-squares ownership/dispatch.
- `tests.zig`: the shared backend-independent numerical contract.
- `backend/accelerate_{core,blas,lapack}.zig`: Accelerate driver layers.
- `backend/cuda_{core,blas,lapack}.zig`: CUDA memory/event, cuBLAS, and
  cuSOLVER driver layers.

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

## Build on Linux with CUDA

CUDA 12.9 or newer, a working NVIDIA driver, and an explicit toolkit path are
required. The path must contain `include/`; libraries are discovered in `lib`,
`lib64`, or `targets/x86_64-linux/lib` beneath it.

```bash
zig build -Dbackend=cuda -Dcuda-path=/usr/local/cuda-12.9 check
zig build -Dbackend=cuda -Dcuda-path=/usr/local/cuda-12.9 test
```

The repository Pixi environment exposes the toolkit target directory and has
matching tasks:

```bash
pixi run linalg-cuda-check
pixi run linalg-cuda-test
```

Each CUDA `Context` owns one nonblocking `cudaStream_t`, one `cublasHandle_t`,
and one `cusolverDnHandle_t`, with both handles bound to that stream. Buffers
live in device memory. Host-returning transfers and reductions synchronize;
BLAS calls enqueue in context order. Events provide opt-in completion checks.
The translated CUDA header rejects older toolkits and asserts complex size and
field layout before the driver uses pointer casts.

Map the operations as follows:

| Public operation | Accelerate | CUDA |
| --- | --- | --- |
| Level 1, GEMV, GER, GEMM | CBLAS | cuBLAS |
| thin/values-only SVD | `xGESDD` | cuSOLVER `gesvd`/`gesvdj` adapter |
| least squares | `xGELSS` | cuSOLVER SVD plus rank-filtered cuBLAS solve |
| synchronize/event | synchronous/no-op event | stream/event |

CUDA least squares performs the one intentional singular-value synchronization
needed to determine rank, then uses cuBLAS scaling and GEMM for the pseudoinverse
application and residual norms. It does not use a custom CUDA kernel.

The package-local tests intentionally contain no Accelerate symbols or platform
conditionals. The same numerical contract runs for either selected backend.
