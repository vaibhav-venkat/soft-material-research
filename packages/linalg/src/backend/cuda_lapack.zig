const std = @import("std");
const core = @import("cuda_core.zig");
const blas = @import("cuda_blas.zig");
const scalar = @import("../scalar.zig");
const c = core.c;
const Buffer = core.Buffer;

fn cp(comptime T: type, p: anytype) *const T {
    return @ptrCast(p);
}
fn mp(comptime T: type, p: anytype) *T {
    return @ptrCast(p);
}

fn cloneMatrix(comptime T: type, source: *const Buffer(T), offset: usize, rows: usize, cols: usize, ld: usize) !Buffer(T) {
    var result = try core.allocate(T, undefined, source.context, try std.math.mul(usize, rows, cols));
    errdefer core.release(T, undefined, &result);
    try core.checkCuda(c.cudaMemcpy2DAsync(
        result.values,
        rows * @sizeOf(T),
        source.values + offset,
        ld * @sizeOf(T),
        rows * @sizeOf(T),
        cols,
        c.cudaMemcpyDeviceToDevice,
        source.context.stream,
    ));
    return result;
}

fn gesvdBufferSize(comptime T: type, context: *core.Context, m: c_int, n: c_int) !c_int {
    var size: c_int = 0;
    try core.checkSolver(switch (T) {
        f32 => c.cusolverDnSgesvd_bufferSize(context.solver, m, n, &size),
        f64 => c.cusolverDnDgesvd_bufferSize(context.solver, m, n, &size),
        scalar.Complex32 => c.cusolverDnCgesvd_bufferSize(context.solver, m, n, &size),
        else => c.cusolverDnZgesvd_bufferSize(context.solver, m, n, &size),
    });
    if (size < 1) return error.BackendFailure;
    return size;
}

fn runGesvd(
    comptime T: type,
    context: *core.Context,
    vectors: bool,
    m: c_int,
    n: c_int,
    a: *Buffer(T),
    s: *Buffer(scalar.Real(T)),
    u: ?*Buffer(T),
    vh: ?*Buffer(T),
    work: *Buffer(T),
    rwork: ?*Buffer(scalar.Real(T)),
    info: *Buffer(c_int),
) !void {
    const job: i8 = if (vectors) 'S' else 'N';
    const dummy_t: [*]T = a.values;
    const up = if (u) |value| value.values else dummy_t;
    const vhp = if (vh) |value| value.values else dummy_t;
    const ldu: c_int = if (vectors) m else 1;
    const ldv: c_int = if (vectors) n else 1;
    const rw = if (rwork) |value| value.values else null;
    const lwork = try core.asCInt(work.len);
    try core.checkSolver(switch (T) {
        f32 => c.cusolverDnSgesvd(context.solver, job, job, m, n, a.values, m, s.values, up, ldu, vhp, ldv, work.values, lwork, rw, info.values),
        f64 => c.cusolverDnDgesvd(context.solver, job, job, m, n, a.values, m, s.values, up, ldu, vhp, ldv, work.values, lwork, rw, info.values),
        scalar.Complex32 => c.cusolverDnCgesvd(context.solver, job, job, m, n, mp(c.cuComplex, a.values), m, s.values, mp(c.cuComplex, up), ldu, mp(c.cuComplex, vhp), ldv, mp(c.cuComplex, work.values), lwork, rw, info.values),
        else => c.cusolverDnZgesvd(context.solver, job, job, m, n, mp(c.cuDoubleComplex, a.values), m, s.values, mp(c.cuDoubleComplex, up), ldu, mp(c.cuDoubleComplex, vhp), ldv, mp(c.cuDoubleComplex, work.values), lwork, rw, info.values),
    });
}

pub fn lapackSvd(
    comptime T: type,
    _: std.mem.Allocator,
    vectors: bool,
    input: *const Buffer(T),
    input_offset: usize,
    rows: usize,
    cols: usize,
    input_ld: usize,
    output_u: ?*Buffer(T),
    singular_values: *Buffer(scalar.Real(T)),
    output_vh: ?*Buffer(T),
) !void {
    const context = input.context;
    const m = try core.asCInt(rows);
    const n = try core.asCInt(cols);
    var a = try cloneMatrix(T, input, input_offset, rows, cols, input_ld);
    defer core.release(T, undefined, &a);
    const lwork: usize = @intCast(try gesvdBufferSize(T, context, m, n));
    var work = try core.allocate(T, undefined, context, lwork);
    defer core.release(T, undefined, &work);
    var info = try core.allocate(c_int, undefined, context, 1);
    defer core.release(c_int, undefined, &info);
    var rwork: ?Buffer(scalar.Real(T)) = null;
    if (comptime scalar.isComplex(T)) rwork = try core.allocate(scalar.Real(T), undefined, context, @max(cols -| 1, 1));
    defer if (rwork) |*value| core.release(scalar.Real(T), undefined, value);
    try runGesvd(T, context, vectors, m, n, &a, singular_values, output_u, output_vh, &work, if (rwork) |*value| value else null, &info);
    var host_info: c_int = 0;
    try core.checkCuda(c.cudaMemcpyAsync(&host_info, info.values, @sizeOf(c_int), c.cudaMemcpyDeviceToHost, context.stream));
    try context.synchronize();
    if (host_info < 0) return error.BackendFailure;
    if (host_info > 0) return error.NonConvergent;
}

fn zero(comptime T: type) T {
    return if (comptime scalar.isComplex(T)) T.init(0, 0) else 0;
}
fn one(comptime T: type) T {
    return if (comptime scalar.isComplex(T)) T.init(1, 0) else 1;
}
fn negativeOne(comptime T: type) T {
    return if (comptime scalar.isComplex(T)) T.init(-1, 0) else -1;
}

pub fn lapackLeastSquares(
    comptime T: type,
    allocator: std.mem.Allocator,
    coefficients: *const Buffer(T),
    coefficients_offset: usize,
    rows: usize,
    cols: usize,
    coefficients_ld: usize,
    rhs: *const Buffer(T),
    rhs_offset: usize,
    rhs_cols: usize,
    rhs_ld: usize,
    requested_rcond: scalar.Real(T),
    solution: *Buffer(T),
    singular_values: *Buffer(scalar.Real(T)),
    residual_norms: *Buffer(scalar.Real(T)),
) !usize {
    const context = coefficients.context;
    var u = try core.allocate(T, allocator, context, try std.math.mul(usize, rows, cols));
    defer core.release(T, allocator, &u);
    var vh = try core.allocate(T, allocator, context, try std.math.mul(usize, cols, cols));
    defer core.release(T, allocator, &vh);
    try lapackSvd(T, allocator, true, coefficients, coefficients_offset, rows, cols, coefficients_ld, &u, singular_values, &vh);

    var y = try core.allocate(T, allocator, context, try std.math.mul(usize, cols, rhs_cols));
    defer core.release(T, allocator, &y);
    var rhs_copy = try cloneMatrix(T, rhs, rhs_offset, rows, rhs_cols, rhs_ld);
    defer core.release(T, allocator, &rhs_copy);
    try blas.blasGemm(T, 2, 0, cols, rhs_cols, rows, one(T), &u, 0, rows, &rhs_copy, 0, rows, zero(T), &y, 0, cols);

    const host_s = try allocator.alloc(scalar.Real(T), cols);
    defer allocator.free(host_s);
    try core.copyToHost(scalar.Real(T), singular_values, 0, 1, host_s);
    const default_rcond: scalar.Real(T) = std.math.floatEps(scalar.Real(T)) * @as(scalar.Real(T), @floatFromInt(@max(rows, cols)));
    const rcond = if (requested_rcond < 0) default_rcond else requested_rcond;
    var rank: usize = 0;
    if (cols != 0 and host_s[0] > 0) {
        while (rank < cols and host_s[rank] > rcond * host_s[0]) : (rank += 1) {}
    }
    for (0..cols) |row| {
        const factor: T = if (row < rank)
            (if (comptime scalar.isComplex(T)) T.init(1 / host_s[row], 0) else 1 / host_s[row])
        else
            zero(T);
        try blas.blasScale(T, factor, &y, row, cols, rhs_cols);
    }
    try blas.blasGemm(T, 2, 0, cols, rhs_cols, cols, one(T), &vh, 0, cols, &y, 0, cols, zero(T), solution, 0, cols);
    var residual = try cloneMatrix(T, rhs, rhs_offset, rows, rhs_cols, rhs_ld);
    defer core.release(T, allocator, &residual);
    try blas.blasGemm(T, 0, 0, rows, rhs_cols, cols, one(T), coefficients, coefficients_offset, coefficients_ld, solution, 0, cols, negativeOne(T), &residual, 0, rows);
    try core.checkBlas(c.cublasSetPointerMode_v2(context.blas, c.CUBLAS_POINTER_MODE_DEVICE));
    errdefer _ = c.cublasSetPointerMode_v2(context.blas, c.CUBLAS_POINTER_MODE_HOST);
    const m = try core.asCInt(rows);
    for (0..rhs_cols) |column| {
        const source = residual.values + column * rows;
        const destination = residual_norms.values + column;
        try core.checkBlas(switch (T) {
            f32 => c.cublasSnrm2_v2(context.blas, m, source, 1, destination),
            f64 => c.cublasDnrm2_v2(context.blas, m, source, 1, destination),
            scalar.Complex32 => c.cublasScnrm2_v2(context.blas, m, cp(c.cuComplex, source), 1, destination),
            else => c.cublasDznrm2_v2(context.blas, m, cp(c.cuDoubleComplex, source), 1, destination),
        });
    }
    try core.checkBlas(c.cublasSetPointerMode_v2(context.blas, c.CUBLAS_POINTER_MODE_HOST));
    return rank;
}
