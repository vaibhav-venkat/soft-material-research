const core = @import("accelerate_core.zig");
const blas = @import("accelerate_blas.zig");
const lapack = @import("accelerate_lapack.zig");

pub const c = core.c;
pub const Context = core.Context;
pub const Event = core.Event;
pub const Buffer = core.Buffer;
pub const allocate = core.allocate;
pub const release = core.release;
pub const copyFromHost = core.copyFromHost;
pub const copyToHost = core.copyToHost;

pub const blasCopy = blas.blasCopy;
pub const blasSwap = blas.blasSwap;
pub const blasScale = blas.blasScale;
pub const blasScaleReal = blas.blasScaleReal;
pub const blasAxpy = blas.blasAxpy;
pub const blasDot = blas.blasDot;
pub const blasNorm2 = blas.blasNorm2;
pub const blasAbsSum = blas.blasAbsSum;
pub const blasIndexAbsMax = blas.blasIndexAbsMax;
pub const blasGemv = blas.blasGemv;
pub const blasGer = blas.blasGer;
pub const blasGemm = blas.blasGemm;

pub const lapackSvd = lapack.lapackSvd;
pub const lapackLeastSquares = lapack.lapackLeastSquares;
