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
