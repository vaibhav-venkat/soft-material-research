const std = @import("std");
const core = @import("cuda_core.zig");
const scalar = @import("../scalar.zig");
const c = core.c;
const Buffer = core.Buffer;
const asCInt = core.asCInt;
const check = core.checkBlas;

fn cp(comptime T: type, p: anytype) *const T {
    return @ptrCast(p);
}
fn mp(comptime T: type, p: anytype) *T {
    return @ptrCast(p);
}

pub fn blasCopy(comptime T: type, x: *const Buffer(T), xo: usize, xs: usize, y: *Buffer(T), yo: usize, ys: usize, len: usize) !void {
    if (len == 0) return;
    const n = try asCInt(len);
    const ix = try asCInt(xs);
    const iy = try asCInt(ys);
    const xp = core.constPointer(T, x, xo);
    const yp = core.mutablePointer(T, y, yo);
    try check(switch (T) {
        f32 => c.cublasScopy_v2(x.context.blas, n, xp, ix, yp, iy),
        f64 => c.cublasDcopy_v2(x.context.blas, n, xp, ix, yp, iy),
        scalar.Complex32 => c.cublasCcopy_v2(x.context.blas, n, cp(c.cuComplex, xp), ix, mp(c.cuComplex, yp), iy),
        else => c.cublasZcopy_v2(x.context.blas, n, cp(c.cuDoubleComplex, xp), ix, mp(c.cuDoubleComplex, yp), iy),
    });
}

pub fn blasSwap(comptime T: type, x: *Buffer(T), xo: usize, xs: usize, y: *Buffer(T), yo: usize, ys: usize, len: usize) !void {
    if (len == 0) return;
    const n = try asCInt(len);
    const ix = try asCInt(xs);
    const iy = try asCInt(ys);
    const xp = core.mutablePointer(T, x, xo);
    const yp = core.mutablePointer(T, y, yo);
    try check(switch (T) {
        f32 => c.cublasSswap_v2(x.context.blas, n, xp, ix, yp, iy),
        f64 => c.cublasDswap_v2(x.context.blas, n, xp, ix, yp, iy),
        scalar.Complex32 => c.cublasCswap_v2(x.context.blas, n, mp(c.cuComplex, xp), ix, mp(c.cuComplex, yp), iy),
        else => c.cublasZswap_v2(x.context.blas, n, mp(c.cuDoubleComplex, xp), ix, mp(c.cuDoubleComplex, yp), iy),
    });
}

pub fn blasScale(comptime T: type, alpha: T, x: *Buffer(T), o: usize, stride: usize, len: usize) !void {
    if (len == 0) return;
    const n = try asCInt(len);
    const inc = try asCInt(stride);
    const xp = core.mutablePointer(T, x, o);
    try check(switch (T) {
        f32 => c.cublasSscal_v2(x.context.blas, n, &alpha, xp, inc),
        f64 => c.cublasDscal_v2(x.context.blas, n, &alpha, xp, inc),
        scalar.Complex32 => c.cublasCscal_v2(x.context.blas, n, cp(c.cuComplex, &alpha), mp(c.cuComplex, xp), inc),
        else => c.cublasZscal_v2(x.context.blas, n, cp(c.cuDoubleComplex, &alpha), mp(c.cuDoubleComplex, xp), inc),
    });
}

pub fn blasScaleReal(comptime T: type, alpha: scalar.Real(T), x: *Buffer(T), o: usize, stride: usize, len: usize) !void {
    if (comptime !scalar.isComplex(T)) @compileError("scaleReal requires complex values");
    if (len == 0) return;
    const n = try asCInt(len);
    const inc = try asCInt(stride);
    const xp = core.mutablePointer(T, x, o);
    try check(switch (T) {
        scalar.Complex32 => c.cublasCsscal_v2(x.context.blas, n, &alpha, mp(c.cuComplex, xp), inc),
        else => c.cublasZdscal_v2(x.context.blas, n, &alpha, mp(c.cuDoubleComplex, xp), inc),
    });
}

pub fn blasAxpy(comptime T: type, alpha: T, x: *const Buffer(T), xo: usize, xs: usize, y: *Buffer(T), yo: usize, ys: usize, len: usize) !void {
    if (len == 0) return;
    const n = try asCInt(len);
    const ix = try asCInt(xs);
    const iy = try asCInt(ys);
    const xp = core.constPointer(T, x, xo);
    const yp = core.mutablePointer(T, y, yo);
    try check(switch (T) {
        f32 => c.cublasSaxpy_v2(x.context.blas, n, &alpha, xp, ix, yp, iy),
        f64 => c.cublasDaxpy_v2(x.context.blas, n, &alpha, xp, ix, yp, iy),
        scalar.Complex32 => c.cublasCaxpy_v2(x.context.blas, n, cp(c.cuComplex, &alpha), cp(c.cuComplex, xp), ix, mp(c.cuComplex, yp), iy),
        else => c.cublasZaxpy_v2(x.context.blas, n, cp(c.cuDoubleComplex, &alpha), cp(c.cuDoubleComplex, xp), ix, mp(c.cuDoubleComplex, yp), iy),
    });
}

pub fn blasDot(comptime T: type, conjugate: bool, x: *const Buffer(T), xo: usize, xs: usize, y: *const Buffer(T), yo: usize, ys: usize, len: usize) !T {
    var result: T = if (comptime scalar.isComplex(T)) T.init(0, 0) else 0;
    if (len == 0) return result;
    const n = try asCInt(len);
    const ix = try asCInt(xs);
    const iy = try asCInt(ys);
    const xp = core.constPointer(T, x, xo);
    const yp = core.constPointer(T, y, yo);
    try check(switch (T) {
        f32 => c.cublasSdot_v2(x.context.blas, n, xp, ix, yp, iy, &result),
        f64 => c.cublasDdot_v2(x.context.blas, n, xp, ix, yp, iy, &result),
        scalar.Complex32 => if (conjugate) c.cublasCdotc_v2(x.context.blas, n, cp(c.cuComplex, xp), ix, cp(c.cuComplex, yp), iy, mp(c.cuComplex, &result)) else c.cublasCdotu_v2(x.context.blas, n, cp(c.cuComplex, xp), ix, cp(c.cuComplex, yp), iy, mp(c.cuComplex, &result)),
        else => if (conjugate) c.cublasZdotc_v2(x.context.blas, n, cp(c.cuDoubleComplex, xp), ix, cp(c.cuDoubleComplex, yp), iy, mp(c.cuDoubleComplex, &result)) else c.cublasZdotu_v2(x.context.blas, n, cp(c.cuDoubleComplex, xp), ix, cp(c.cuDoubleComplex, yp), iy, mp(c.cuDoubleComplex, &result)),
    });
    return result;
}

pub fn blasNorm2(comptime T: type, x: *const Buffer(T), o: usize, stride: usize, len: usize) !scalar.Real(T) {
    var result: scalar.Real(T) = 0;
    if (len == 0) return result;
    const n = try asCInt(len);
    const inc = try asCInt(stride);
    const xp = core.constPointer(T, x, o);
    try check(switch (T) {
        f32 => c.cublasSnrm2_v2(x.context.blas, n, xp, inc, &result),
        f64 => c.cublasDnrm2_v2(x.context.blas, n, xp, inc, &result),
        scalar.Complex32 => c.cublasScnrm2_v2(x.context.blas, n, cp(c.cuComplex, xp), inc, &result),
        else => c.cublasDznrm2_v2(x.context.blas, n, cp(c.cuDoubleComplex, xp), inc, &result),
    });
    return result;
}

pub fn blasAbsSum(comptime T: type, x: *const Buffer(T), o: usize, stride: usize, len: usize) !scalar.Real(T) {
    var result: scalar.Real(T) = 0;
    if (len == 0) return result;
    const n = try asCInt(len);
    const inc = try asCInt(stride);
    const xp = core.constPointer(T, x, o);
    try check(switch (T) {
        f32 => c.cublasSasum_v2(x.context.blas, n, xp, inc, &result),
        f64 => c.cublasDasum_v2(x.context.blas, n, xp, inc, &result),
        scalar.Complex32 => c.cublasScasum_v2(x.context.blas, n, cp(c.cuComplex, xp), inc, &result),
        else => c.cublasDzasum_v2(x.context.blas, n, cp(c.cuDoubleComplex, xp), inc, &result),
    });
    return result;
}

pub fn blasIndexAbsMax(comptime T: type, x: *const Buffer(T), o: usize, stride: usize, len: usize) !usize {
    if (len == 0) return error.EmptyInput;
    var result: c_int = 0;
    const n = try asCInt(len);
    const inc = try asCInt(stride);
    const xp = core.constPointer(T, x, o);
    try check(switch (T) {
        f32 => c.cublasIsamax_v2(x.context.blas, n, xp, inc, &result),
        f64 => c.cublasIdamax_v2(x.context.blas, n, xp, inc, &result),
        scalar.Complex32 => c.cublasIcamax_v2(x.context.blas, n, cp(c.cuComplex, xp), inc, &result),
        else => c.cublasIzamax_v2(x.context.blas, n, cp(c.cuDoubleComplex, xp), inc, &result),
    });
    if (result <= 0) return error.BackendFailure;
    return @intCast(result - 1);
}

fn op(comptime T: type, operation: u8) c.cublasOperation_t {
    return switch (operation) {
        0 => c.CUBLAS_OP_N,
        1 => c.CUBLAS_OP_T,
        2 => if (scalar.isComplex(T)) c.CUBLAS_OP_C else c.CUBLAS_OP_T,
        else => unreachable,
    };
}

pub fn blasGemv(comptime T: type, operation: u8, alpha: T, a: *const Buffer(T), ao: usize, rows: usize, cols: usize, lda0: usize, x: *const Buffer(T), xo: usize, xs: usize, beta: T, y: *Buffer(T), yo: usize, ys: usize) !void {
    const m = try asCInt(rows);
    const n = try asCInt(cols);
    const lda = try asCInt(lda0);
    const ix = try asCInt(xs);
    const iy = try asCInt(ys);
    const ap = core.constPointer(T, a, ao);
    const xp = core.constPointer(T, x, xo);
    const yp = core.mutablePointer(T, y, yo);
    const trans = op(T, operation);
    try check(switch (T) {
        f32 => c.cublasSgemv_v2(a.context.blas, trans, m, n, &alpha, ap, lda, xp, ix, &beta, yp, iy),
        f64 => c.cublasDgemv_v2(a.context.blas, trans, m, n, &alpha, ap, lda, xp, ix, &beta, yp, iy),
        scalar.Complex32 => c.cublasCgemv_v2(a.context.blas, trans, m, n, cp(c.cuComplex, &alpha), cp(c.cuComplex, ap), lda, cp(c.cuComplex, xp), ix, cp(c.cuComplex, &beta), mp(c.cuComplex, yp), iy),
        else => c.cublasZgemv_v2(a.context.blas, trans, m, n, cp(c.cuDoubleComplex, &alpha), cp(c.cuDoubleComplex, ap), lda, cp(c.cuDoubleComplex, xp), ix, cp(c.cuDoubleComplex, &beta), mp(c.cuDoubleComplex, yp), iy),
    });
}

pub fn blasGer(comptime T: type, conj: bool, alpha: T, x: *const Buffer(T), xo: usize, xs: usize, y: *const Buffer(T), yo: usize, ys: usize, a: *Buffer(T), ao: usize, rows: usize, cols: usize, lda0: usize) !void {
    const m = try asCInt(rows);
    const n = try asCInt(cols);
    const ix = try asCInt(xs);
    const iy = try asCInt(ys);
    const lda = try asCInt(lda0);
    const xp = core.constPointer(T, x, xo);
    const yp = core.constPointer(T, y, yo);
    const ap = core.mutablePointer(T, a, ao);
    try check(switch (T) {
        f32 => c.cublasSger_v2(a.context.blas, m, n, &alpha, xp, ix, yp, iy, ap, lda),
        f64 => c.cublasDger_v2(a.context.blas, m, n, &alpha, xp, ix, yp, iy, ap, lda),
        scalar.Complex32 => if (conj) c.cublasCgerc_v2(a.context.blas, m, n, cp(c.cuComplex, &alpha), cp(c.cuComplex, xp), ix, cp(c.cuComplex, yp), iy, mp(c.cuComplex, ap), lda) else c.cublasCgeru_v2(a.context.blas, m, n, cp(c.cuComplex, &alpha), cp(c.cuComplex, xp), ix, cp(c.cuComplex, yp), iy, mp(c.cuComplex, ap), lda),
        else => if (conj) c.cublasZgerc_v2(a.context.blas, m, n, cp(c.cuDoubleComplex, &alpha), cp(c.cuDoubleComplex, xp), ix, cp(c.cuDoubleComplex, yp), iy, mp(c.cuDoubleComplex, ap), lda) else c.cublasZgeru_v2(a.context.blas, m, n, cp(c.cuDoubleComplex, &alpha), cp(c.cuDoubleComplex, xp), ix, cp(c.cuDoubleComplex, yp), iy, mp(c.cuDoubleComplex, ap), lda),
    });
}

pub fn blasGemm(comptime T: type, oa: u8, ob: u8, m0: usize, n0: usize, k0: usize, alpha: T, a: *const Buffer(T), ao: usize, lda0: usize, b: *const Buffer(T), bo: usize, ldb0: usize, beta: T, out: *Buffer(T), oo: usize, ldc0: usize) !void {
    const m = try asCInt(m0);
    const n = try asCInt(n0);
    const k = try asCInt(k0);
    const lda = try asCInt(lda0);
    const ldb = try asCInt(ldb0);
    const ldc = try asCInt(ldc0);
    const ap = core.constPointer(T, a, ao);
    const bp = core.constPointer(T, b, bo);
    const cp0 = core.mutablePointer(T, out, oo);
    try check(switch (T) {
        f32 => c.cublasSgemm_v2(a.context.blas, op(T, oa), op(T, ob), m, n, k, &alpha, ap, lda, bp, ldb, &beta, cp0, ldc),
        f64 => c.cublasDgemm_v2(a.context.blas, op(T, oa), op(T, ob), m, n, k, &alpha, ap, lda, bp, ldb, &beta, cp0, ldc),
        scalar.Complex32 => c.cublasCgemm_v2(a.context.blas, op(T, oa), op(T, ob), m, n, k, cp(c.cuComplex, &alpha), cp(c.cuComplex, ap), lda, cp(c.cuComplex, bp), ldb, cp(c.cuComplex, &beta), mp(c.cuComplex, cp0), ldc),
        else => c.cublasZgemm_v2(a.context.blas, op(T, oa), op(T, ob), m, n, k, cp(c.cuDoubleComplex, &alpha), cp(c.cuDoubleComplex, ap), lda, cp(c.cuDoubleComplex, bp), ldb, cp(c.cuDoubleComplex, &beta), mp(c.cuDoubleComplex, cp0), ldc),
    });
}
