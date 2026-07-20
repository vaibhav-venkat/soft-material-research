const std = @import("std");
const scalar = @import("../scalar.zig");

pub const c = @import("accelerate_c");

pub const Context = struct {
    pub fn init() Context {
        return .{};
    }

    pub fn deinit(_: *Context) void {}
    pub fn synchronize(_: *Context) void {}
};

pub const Event = struct {
    pub fn query(_: *const Event) bool {
        return true;
    }

    pub fn wait(_: *Event) void {}
    pub fn deinit(_: *Event) void {}
};

pub fn Buffer(comptime T: type) type {
    return struct {
        values: []T,
    };
}

pub fn allocate(comptime T: type, allocator: std.mem.Allocator, len: usize) !Buffer(T) {
    return .{ .values = try allocator.alloc(T, len) };
}

pub fn release(comptime T: type, allocator: std.mem.Allocator, buffer: *Buffer(T)) void {
    allocator.free(buffer.values);
    buffer.values = &.{};
}

pub fn copyFromHost(
    comptime T: type,
    buffer: *Buffer(T),
    offset: usize,
    stride: usize,
    source: []const T,
) void {
    for (source, 0..) |value, index| buffer.values[offset + index * stride] = value;
}

pub fn copyToHost(
    comptime T: type,
    buffer: *const Buffer(T),
    offset: usize,
    stride: usize,
    destination: []T,
) void {
    for (destination, 0..) |*value, index| value.* = buffer.values[offset + index * stride];
}

fn asCInt(value: usize) !c_int {
    return std.math.cast(c_int, value) orelse error.DimensionTooLarge;
}

fn constPointer(comptime T: type, buffer: *const Buffer(T), offset: usize) *const T {
    return &buffer.values[offset];
}

fn mutablePointer(comptime T: type, buffer: *Buffer(T), offset: usize) *T {
    return &buffer.values[offset];
}

pub fn blasCopy(
    comptime T: type,
    x: *const Buffer(T),
    x_offset: usize,
    x_stride: usize,
    y: *Buffer(T),
    y_offset: usize,
    y_stride: usize,
    len: usize,
) !void {
    if (len == 0) return;
    const n = try asCInt(len);
    const incx = try asCInt(x_stride);
    const incy = try asCInt(y_stride);
    const xp = constPointer(T, x, x_offset);
    const yp = mutablePointer(T, y, y_offset);
    if (T == f32) c.cblas_scopy(n, @ptrCast(xp), incx, @ptrCast(yp), incy) else if (T == f64) c.cblas_dcopy(n, @ptrCast(xp), incx, @ptrCast(yp), incy) else if (T == scalar.Complex32) c.cblas_ccopy(n, xp, incx, yp, incy) else c.cblas_zcopy(n, xp, incx, yp, incy);
}

pub fn blasSwap(
    comptime T: type,
    x: *Buffer(T),
    x_offset: usize,
    x_stride: usize,
    y: *Buffer(T),
    y_offset: usize,
    y_stride: usize,
    len: usize,
) !void {
    if (len == 0) return;
    const n = try asCInt(len);
    const incx = try asCInt(x_stride);
    const incy = try asCInt(y_stride);
    const xp = mutablePointer(T, x, x_offset);
    const yp = mutablePointer(T, y, y_offset);
    if (T == f32) c.cblas_sswap(n, @ptrCast(xp), incx, @ptrCast(yp), incy) else if (T == f64) c.cblas_dswap(n, @ptrCast(xp), incx, @ptrCast(yp), incy) else if (T == scalar.Complex32) c.cblas_cswap(n, xp, incx, yp, incy) else c.cblas_zswap(n, xp, incx, yp, incy);
}

pub fn blasScale(
    comptime T: type,
    alpha: T,
    x: *Buffer(T),
    offset: usize,
    stride: usize,
    len: usize,
) !void {
    if (len == 0) return;
    const n = try asCInt(len);
    const incx = try asCInt(stride);
    const xp = mutablePointer(T, x, offset);
    if (T == f32) c.cblas_sscal(n, alpha, @ptrCast(xp), incx) else if (T == f64) c.cblas_dscal(n, alpha, @ptrCast(xp), incx) else if (T == scalar.Complex32) c.cblas_cscal(n, &alpha, xp, incx) else c.cblas_zscal(n, &alpha, xp, incx);
}

pub fn blasScaleReal(
    comptime T: type,
    alpha: scalar.Real(T),
    x: *Buffer(T),
    offset: usize,
    stride: usize,
    len: usize,
) !void {
    if (comptime !scalar.isComplex(T)) @compileError("scaleReal requires a complex vector");
    if (len == 0) return;
    const n = try asCInt(len);
    const incx = try asCInt(stride);
    const xp = mutablePointer(T, x, offset);
    if (T == scalar.Complex32) c.cblas_csscal(n, alpha, xp, incx) else c.cblas_zdscal(n, alpha, xp, incx);
}

pub fn blasAxpy(
    comptime T: type,
    alpha: T,
    x: *const Buffer(T),
    x_offset: usize,
    x_stride: usize,
    y: *Buffer(T),
    y_offset: usize,
    y_stride: usize,
    len: usize,
) !void {
    if (len == 0) return;
    const n = try asCInt(len);
    const incx = try asCInt(x_stride);
    const incy = try asCInt(y_stride);
    const xp = constPointer(T, x, x_offset);
    const yp = mutablePointer(T, y, y_offset);
    if (T == f32) c.cblas_saxpy(n, alpha, @ptrCast(xp), incx, @ptrCast(yp), incy) else if (T == f64) c.cblas_daxpy(n, alpha, @ptrCast(xp), incx, @ptrCast(yp), incy) else if (T == scalar.Complex32) c.cblas_caxpy(n, &alpha, xp, incx, yp, incy) else c.cblas_zaxpy(n, &alpha, xp, incx, yp, incy);
}

pub fn blasDot(
    comptime T: type,
    conjugate: bool,
    x: *const Buffer(T),
    x_offset: usize,
    x_stride: usize,
    y: *const Buffer(T),
    y_offset: usize,
    y_stride: usize,
    len: usize,
) !T {
    if (len == 0) return if (comptime scalar.isComplex(T)) T.init(0, 0) else 0;
    const n = try asCInt(len);
    const incx = try asCInt(x_stride);
    const incy = try asCInt(y_stride);
    const xp = constPointer(T, x, x_offset);
    const yp = constPointer(T, y, y_offset);
    if (T == f32) return c.cblas_sdot(n, @ptrCast(xp), incx, @ptrCast(yp), incy);
    if (T == f64) return c.cblas_ddot(n, @ptrCast(xp), incx, @ptrCast(yp), incy);
    var result = T.init(0, 0);
    if (T == scalar.Complex32) {
        if (conjugate) c.cblas_cdotc_sub(n, xp, incx, yp, incy, &result) else c.cblas_cdotu_sub(n, xp, incx, yp, incy, &result);
    } else {
        if (conjugate) c.cblas_zdotc_sub(n, xp, incx, yp, incy, &result) else c.cblas_zdotu_sub(n, xp, incx, yp, incy, &result);
    }
    return result;
}

pub fn blasNorm2(
    comptime T: type,
    x: *const Buffer(T),
    offset: usize,
    stride: usize,
    len: usize,
) !scalar.Real(T) {
    if (len == 0) return 0;
    const n = try asCInt(len);
    const incx = try asCInt(stride);
    const xp = constPointer(T, x, offset);
    if (T == f32) return c.cblas_snrm2(n, @ptrCast(xp), incx);
    if (T == f64) return c.cblas_dnrm2(n, @ptrCast(xp), incx);
    if (T == scalar.Complex32) return c.cblas_scnrm2(n, xp, incx);
    return c.cblas_dznrm2(n, xp, incx);
}

pub fn blasAbsSum(
    comptime T: type,
    x: *const Buffer(T),
    offset: usize,
    stride: usize,
    len: usize,
) !scalar.Real(T) {
    if (len == 0) return 0;
    const n = try asCInt(len);
    const incx = try asCInt(stride);
    const xp = constPointer(T, x, offset);
    if (T == f32) return c.cblas_sasum(n, @ptrCast(xp), incx);
    if (T == f64) return c.cblas_dasum(n, @ptrCast(xp), incx);
    if (T == scalar.Complex32) return c.cblas_scasum(n, xp, incx);
    return c.cblas_dzasum(n, xp, incx);
}

pub fn blasIndexAbsMax(
    comptime T: type,
    x: *const Buffer(T),
    offset: usize,
    stride: usize,
    len: usize,
) !usize {
    if (len == 0) return error.EmptyInput;
    const n = try asCInt(len);
    const incx = try asCInt(stride);
    const xp = constPointer(T, x, offset);
    const index = if (T == f32)
        c.cblas_isamax(n, @ptrCast(xp), incx)
    else if (T == f64)
        c.cblas_idamax(n, @ptrCast(xp), incx)
    else if (T == scalar.Complex32)
        c.cblas_icamax(n, xp, incx)
    else
        c.cblas_izamax(n, xp, incx);
    return std.math.cast(usize, index) orelse error.BackendFailure;
}

fn cOrder() c.enum_CBLAS_ORDER {
    return @intCast(c.CblasColMajor);
}

fn cTranspose(comptime T: type, operation: u8) c.enum_CBLAS_TRANSPOSE {
    return @intCast(switch (operation) {
        0 => c.CblasNoTrans,
        1 => c.CblasTrans,
        2 => if (scalar.isComplex(T)) c.CblasConjTrans else c.CblasTrans,
        else => unreachable,
    });
}

pub fn blasGemv(
    comptime T: type,
    operation: u8,
    alpha: T,
    a: *const Buffer(T),
    a_offset: usize,
    rows: usize,
    cols: usize,
    leading_dimension: usize,
    x: *const Buffer(T),
    x_offset: usize,
    x_stride: usize,
    beta: T,
    y: *Buffer(T),
    y_offset: usize,
    y_stride: usize,
) !void {
    const m = try asCInt(rows);
    const n = try asCInt(cols);
    const lda = try asCInt(leading_dimension);
    const incx = try asCInt(x_stride);
    const incy = try asCInt(y_stride);
    const ap = constPointer(T, a, a_offset);
    const xp = constPointer(T, x, x_offset);
    const yp = mutablePointer(T, y, y_offset);
    const trans = cTranspose(T, operation);
    if (T == f32) c.cblas_sgemv(cOrder(), trans, m, n, alpha, @ptrCast(ap), lda, @ptrCast(xp), incx, beta, @ptrCast(yp), incy) else if (T == f64) c.cblas_dgemv(cOrder(), trans, m, n, alpha, @ptrCast(ap), lda, @ptrCast(xp), incx, beta, @ptrCast(yp), incy) else if (T == scalar.Complex32) c.cblas_cgemv(cOrder(), trans, m, n, &alpha, ap, lda, xp, incx, &beta, yp, incy) else c.cblas_zgemv(cOrder(), trans, m, n, &alpha, ap, lda, xp, incx, &beta, yp, incy);
}

pub fn blasGer(
    comptime T: type,
    conjugate_y: bool,
    alpha: T,
    x: *const Buffer(T),
    x_offset: usize,
    x_stride: usize,
    y: *const Buffer(T),
    y_offset: usize,
    y_stride: usize,
    a: *Buffer(T),
    a_offset: usize,
    rows: usize,
    cols: usize,
    leading_dimension: usize,
) !void {
    const m = try asCInt(rows);
    const n = try asCInt(cols);
    const incx = try asCInt(x_stride);
    const incy = try asCInt(y_stride);
    const lda = try asCInt(leading_dimension);
    const xp = constPointer(T, x, x_offset);
    const yp = constPointer(T, y, y_offset);
    const ap = mutablePointer(T, a, a_offset);
    if (T == f32) c.cblas_sger(cOrder(), m, n, alpha, @ptrCast(xp), incx, @ptrCast(yp), incy, @ptrCast(ap), lda) else if (T == f64) c.cblas_dger(cOrder(), m, n, alpha, @ptrCast(xp), incx, @ptrCast(yp), incy, @ptrCast(ap), lda) else if (T == scalar.Complex32) {
        if (conjugate_y) c.cblas_cgerc(cOrder(), m, n, &alpha, xp, incx, yp, incy, ap, lda) else c.cblas_cgeru(cOrder(), m, n, &alpha, xp, incx, yp, incy, ap, lda);
    } else {
        if (conjugate_y) c.cblas_zgerc(cOrder(), m, n, &alpha, xp, incx, yp, incy, ap, lda) else c.cblas_zgeru(cOrder(), m, n, &alpha, xp, incx, yp, incy, ap, lda);
    }
}

pub fn blasGemm(
    comptime T: type,
    operation_a: u8,
    operation_b: u8,
    m_value: usize,
    n_value: usize,
    k_value: usize,
    alpha: T,
    a: *const Buffer(T),
    a_offset: usize,
    lda_value: usize,
    b: *const Buffer(T),
    b_offset: usize,
    ldb_value: usize,
    beta: T,
    output: *Buffer(T),
    output_offset: usize,
    ldc_value: usize,
) !void {
    const m = try asCInt(m_value);
    const n = try asCInt(n_value);
    const k = try asCInt(k_value);
    const lda = try asCInt(lda_value);
    const ldb = try asCInt(ldb_value);
    const ldc = try asCInt(ldc_value);
    const ap = constPointer(T, a, a_offset);
    const bp = constPointer(T, b, b_offset);
    const cp = mutablePointer(T, output, output_offset);
    const trans_a = cTranspose(T, operation_a);
    const trans_b = cTranspose(T, operation_b);
    if (T == f32) c.cblas_sgemm(cOrder(), trans_a, trans_b, m, n, k, alpha, @ptrCast(ap), lda, @ptrCast(bp), ldb, beta, @ptrCast(cp), ldc) else if (T == f64) c.cblas_dgemm(cOrder(), trans_a, trans_b, m, n, k, alpha, @ptrCast(ap), lda, @ptrCast(bp), ldb, beta, @ptrCast(cp), ldc) else if (T == scalar.Complex32) c.cblas_cgemm(cOrder(), trans_a, trans_b, m, n, k, &alpha, ap, lda, bp, ldb, &beta, cp, ldc) else c.cblas_zgemm(cOrder(), trans_a, trans_b, m, n, k, &alpha, ap, lda, bp, ldb, &beta, cp, ldc);
}

fn callGesdd(
    comptime T: type,
    jobz: u8,
    m: c_int,
    n: c_int,
    a: [*]T,
    singular_values: [*]scalar.Real(T),
    u: [*]T,
    vt: [*]T,
    work: [*]T,
    lwork: c_int,
    rwork: ?[*]scalar.Real(T),
    iwork: [*]c_int,
    info: *c_int,
) void {
    const lda = m;
    const ldu: c_int = if (jobz == 'S') m else 1;
    const ldvt: c_int = if (jobz == 'S') n else 1;
    if (T == f32) {
        c.linalg_sgesdd(jobz, m, n, a, lda, singular_values, u, ldu, vt, ldvt, work, lwork, iwork, info);
    } else if (T == f64) {
        c.linalg_dgesdd(jobz, m, n, a, lda, singular_values, u, ldu, vt, ldvt, work, lwork, iwork, info);
    } else if (T == scalar.Complex32) {
        c.linalg_cgesdd(jobz, m, n, a, lda, singular_values, u, ldu, vt, ldvt, work, lwork, rwork.?, iwork, info);
    } else {
        c.linalg_zgesdd(jobz, m, n, a, lda, singular_values, u, ldu, vt, ldvt, work, lwork, rwork.?, iwork, info);
    }
}

pub fn lapackSvd(
    comptime T: type,
    allocator: std.mem.Allocator,
    vectors: bool,
    input: *const Buffer(T),
    input_offset: usize,
    rows: usize,
    cols: usize,
    input_leading_dimension: usize,
    output_u: ?*Buffer(T),
    singular_values: *Buffer(scalar.Real(T)),
    output_vh: ?*Buffer(T),
) !void {
    const m = try asCInt(rows);
    const n = try asCInt(cols);
    const matrix_len = try std.math.mul(usize, rows, cols);
    var a = try allocator.alloc(T, matrix_len);
    defer allocator.free(a);
    for (0..cols) |column| {
        @memcpy(a[column * rows ..][0..rows], input.values[input_offset + column * input_leading_dimension ..][0..rows]);
    }

    const thin_len = try std.math.mul(usize, rows, cols);
    const u = try allocator.alloc(T, if (vectors) thin_len else 1);
    defer allocator.free(u);
    const vh = try allocator.alloc(T, if (vectors) try std.math.mul(usize, cols, cols) else 1);
    defer allocator.free(vh);
    const s = try allocator.alloc(scalar.Real(T), cols);
    defer allocator.free(s);
    const iwork = try allocator.alloc(c_int, try std.math.mul(usize, 8, cols));
    defer allocator.free(iwork);

    var rwork: ?[]scalar.Real(T) = null;
    if (comptime scalar.isComplex(T)) {
        const n_squared = try std.math.mul(usize, cols, cols);
        const mn = try std.math.mul(usize, rows, cols);
        const rwork_len = if (vectors)
            try std.math.add(usize, try std.math.add(usize, try std.math.mul(usize, 7, n_squared), try std.math.mul(usize, 6, cols)), try std.math.mul(usize, 2, mn))
        else
            try std.math.mul(usize, 5, cols);
        rwork = try allocator.alloc(scalar.Real(T), @max(rwork_len, 1));
    }
    defer if (rwork) |values| allocator.free(values);

    var query: [1]T = undefined;
    var info: c_int = 0;
    const jobz: u8 = if (vectors) 'S' else 'N';
    callGesdd(T, jobz, m, n, a.ptr, s.ptr, u.ptr, vh.ptr, &query, -1, if (rwork) |values| values.ptr else null, iwork.ptr, &info);
    if (info != 0) return error.BackendFailure;
    const suggested = if (comptime scalar.isComplex(T)) query[0].re else query[0];
    if (!std.math.isFinite(suggested) or suggested < 1) return error.BackendFailure;
    const work_len: usize = @intFromFloat(@ceil(suggested));
    const lwork = try asCInt(work_len);
    const work = try allocator.alloc(T, work_len);
    defer allocator.free(work);

    callGesdd(T, jobz, m, n, a.ptr, s.ptr, u.ptr, vh.ptr, work.ptr, lwork, if (rwork) |values| values.ptr else null, iwork.ptr, &info);
    if (info < 0) return error.BackendFailure;
    if (info > 0) return error.NonConvergent;

    @memcpy(singular_values.values[0..cols], s);
    if (vectors) {
        @memcpy(output_u.?.values[0..thin_len], u);
        @memcpy(output_vh.?.values[0 .. cols * cols], vh);
    }
}
