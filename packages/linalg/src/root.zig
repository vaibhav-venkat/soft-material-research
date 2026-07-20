//! Backend-neutral dense linear algebra for scientific computing.
//!
//! Matrices are column-major. All operations accept real and complex scalars and
//! dispatch to the backend selected in build.zig.

const types = @import("types.zig");
const blas = @import("blas.zig");
const lapack = @import("lapack.zig");

pub const Backend = types.Backend;
pub const compiled_backend = types.compiled_backend;
pub const Transpose = types.Transpose;
pub const Complex = types.Complex;
pub const Complex32 = types.Complex32;
pub const Complex64 = types.Complex64;
pub const Real = types.Real;
pub const ContextOptions = types.ContextOptions;
pub const BackendInfo = types.BackendInfo;
pub const Context = types.Context;
pub const Event = types.Event;
pub const VectorView = types.VectorView;
pub const ConstVectorView = types.ConstVectorView;
pub const MatrixView = types.MatrixView;
pub const ConstMatrixView = types.ConstMatrixView;
pub const Vector = types.Vector;
pub const Matrix = types.Matrix;

/// Copies a vector without changing it. Use this to preserve a state, residual,
/// forcing term, or experimental signal before an in-place numerical update.
pub const copy = blas.copy;
/// Exchanges two vectors. This is useful for buffer rotation in time integrators,
/// Krylov iterations, eigensolvers, and other algorithms that alternate states.
pub const swap = blas.swap;
/// Applies x <- alpha*x. Use it for nondimensionalization, unit conversion,
/// normalization, damping, or multiplying a field by a physical coefficient.
pub const scale = blas.scale;
/// Scales a complex vector by a real coefficient, preserving every phase. This is
/// appropriate for real attenuation, normalization, and timestep factors.
pub const scaleReal = blas.scaleReal;
/// Applies y <- alpha*x + y. AXPY is the basic update in explicit integrators,
/// residual correction, gradient methods, and linear combinations of fields.
pub const axpy = blas.axpy;
/// Computes the real bilinear dot product. Use it for projections, correlations,
/// work/energy-like contractions, orthogonality checks, and residual tests.
pub const dot = blas.dot;
/// Computes the unconjugated complex product sum(x_i*y_i). Use it for symmetric
/// complex bilinear forms; it is not the usual Hilbert-space inner product.
pub const dotu = blas.dotu;
/// Computes the Hermitian product sum(conj(x_i)*y_i). Use it for complex norms,
/// quantum/wave overlaps, modal projection, and complex orthogonality.
pub const dotc = blas.dotc;
/// Computes the Euclidean/Hermitian 2-norm. Use it to normalize states, measure
/// residual magnitude, enforce tolerances, or monitor conservation errors.
pub const norm2 = blas.norm2;
/// Computes the BLAS 1-norm sum(abs(x)) for real data and sum(abs(re)+abs(im))
/// for complex data. Use it for robust magnitude diagnostics and scale estimates.
pub const absSum = blas.absSum;
/// Finds the component with largest BLAS absolute magnitude. Use it for pivot
/// selection, locating a dominant mode/component, and peak-error diagnostics.
pub const indexAbsMax = blas.indexAbsMax;

/// Applies y <- alpha*op(A)*x + beta*y without allocating. Use GEMV when a dense
/// linear operator acts on one scientific state vector or parameter vector.
pub const gemvInto = blas.gemvInto;
/// Allocating GEMV convenience for direct operator application, such as advancing
/// a linear model, evaluating a Jacobian action, or changing vector bases.
pub const matvec = blas.matvec;
/// Applies the real rank-one update A <- alpha*x*y^T + A. Use it to accumulate
/// covariance-like matrices, outer products, Jacobians, or quasi-Newton updates.
pub const gerInto = blas.gerInto;
/// Applies the unconjugated complex rank-one update A <- alpha*x*y^T + A. Use it
/// for complex-symmetric models where conjugating the second vector is incorrect.
pub const geruInto = blas.geruInto;
/// Applies the Hermitian rank-one update A <- alpha*x*conj(y)^T + A. Use it for
/// density matrices, cross-spectral matrices, and complex covariance accumulation.
pub const gercInto = blas.gercInto;
/// Applies C <- alpha*op(A)*op(B) + beta*C without allocating. Use GEMM for
/// batched states, basis changes, covariance products, and dense operator chains.
pub const gemmInto = blas.gemmInto;
/// Allocating GEMM convenience for composing dense operators or transforming a
/// collection of state/observation vectors stored as matrix columns.
pub const matmul = blas.matmul;

pub const SvdMode = lapack.SvdMode;
pub const SvdOptions = lapack.SvdOptions;
pub const SvdResult = lapack.SvdResult;
/// Factors A = U*S*V^H (thin mode) or returns only singular values. Use SVD for
/// principal modes, conditioning, numerical rank, pseudoinverses, compression,
/// denoising, and identifying poorly constrained directions in an inverse problem.
pub const svd = lapack.svd;
pub const LeastSquaresOptions = lapack.LeastSquaresOptions;
pub const LeastSquaresResult = lapack.LeastSquaresResult;
/// Solves min_X ||A*X-B||_2 using an SVD-based rank decision. Use least squares
/// for regression, parameter estimation, calibration, data fitting, and noisy or
/// overdetermined inverse problems; inspect rank and singular values for identifiability.
pub const leastSquares = lapack.leastSquares;
