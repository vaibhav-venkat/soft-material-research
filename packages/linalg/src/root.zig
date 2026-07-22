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

/// Copies a vector without changing it.
pub const copy = blas.copy;
/// Exchanges two vectors.
pub const swap = blas.swap;
/// Applies x <- alpha*x.
pub const scale = blas.scale;
/// Scales a complex vector by a real coefficient, preserving every phase.
pub const scaleReal = blas.scaleReal;
/// Applies y <- alpha*x + y.
pub const axpy = blas.axpy;
/// Computes the real bilinear dot product.
pub const dot = blas.dot;
/// Computes the unconjugated complex product sum(x_i*y_i).
pub const dotu = blas.dotu;
/// Computes the Hermitian product sum(conj(x_i)*y_i).
pub const dotc = blas.dotc;
/// Computes the Euclidean/Hermitian 2-norm
pub const norm2 = blas.norm2;
/// Computes the BLAS 1-norm sum(abs(x)) for real data and sum(abs(re)+abs(im))
/// for complex data. Manhattan distance
pub const absSum = blas.absSum;
/// Finds the component with largest BLAS absolute magnitude.
pub const indexAbsMax = blas.indexAbsMax;
/// Applies y <- alpha*op(A)*x + beta*y without allocating.
pub const gemvInto = blas.gemvInto;
/// Allocating GEMV convenience for direct operator application.
pub const matvec = blas.matvec;
/// Applies the real rank-one update A <- alpha*x*y^T + A.
pub const gerInto = blas.gerInto;
/// Applies the unconjugated complex rank-one update A <- alpha*x*y^T + A.
pub const geruInto = blas.geruInto;
/// Applies the Hermitian rank-one update A <- alpha*x*conj(y)^T + A.
pub const gercInto = blas.gercInto;
/// Applies C <- alpha*op(A)*op(B) + beta*C without allocating.
pub const gemmInto = blas.gemmInto;
/// Allocating GEMM convenience
pub const matmul = blas.matmul;
// Find mean of a vector
pub const vecMean = blas.vecMean;
pub const SvdMode = lapack.SvdMode;
pub const SvdOptions = lapack.SvdOptions;
pub const SvdResult = lapack.SvdResult;
/// Factors A = U*S*V^H (thin mode) or returns only singular values.
pub const svd = lapack.svd;
pub const LeastSquaresOptions = lapack.LeastSquaresOptions;
pub const LeastSquaresResult = lapack.LeastSquaresResult;
/// Solves min_X ||A*X-B||_2 using an SVD-based rank decision.
pub const leastSquares = lapack.leastSquares;
