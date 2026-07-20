const std = @import("std");
const core = @import("accelerate_core.zig");
const scalar = @import("../scalar.zig");

const c = core.c;
const Buffer = core.Buffer;
const asCInt = core.asCInt;

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

fn callGelss(
    comptime T: type,
    m: c_int,
    n: c_int,
    nrhs: c_int,
    a: [*]T,
    b: [*]T,
    singular_values: [*]scalar.Real(T),
    rcond: scalar.Real(T),
    rank: *c_int,
    work: [*]T,
    lwork: c_int,
    rwork: ?[*]scalar.Real(T),
    info: *c_int,
) void {
    if (T == f32) {
        c.linalg_sgelss(m, n, nrhs, a, m, b, m, singular_values, rcond, rank, work, lwork, info);
    } else if (T == f64) {
        c.linalg_dgelss(m, n, nrhs, a, m, b, m, singular_values, rcond, rank, work, lwork, info);
    } else if (T == scalar.Complex32) {
        c.linalg_cgelss(m, n, nrhs, a, m, b, m, singular_values, rcond, rank, work, lwork, rwork.?, info);
    } else {
        c.linalg_zgelss(m, n, nrhs, a, m, b, m, singular_values, rcond, rank, work, lwork, rwork.?, info);
    }
}

pub fn lapackLeastSquares(
    comptime T: type,
    allocator: std.mem.Allocator,
    coefficients: *const Buffer(T),
    coefficients_offset: usize,
    rows: usize,
    cols: usize,
    coefficients_leading_dimension: usize,
    right_hand_sides: *const Buffer(T),
    rhs_offset: usize,
    rhs_cols: usize,
    rhs_leading_dimension: usize,
    rcond: scalar.Real(T),
    solution: *Buffer(T),
    singular_values: *Buffer(scalar.Real(T)),
) !usize {
    const m = try asCInt(rows);
    const n = try asCInt(cols);
    const nrhs = try asCInt(rhs_cols);
    const a_len = try std.math.mul(usize, rows, cols);
    const b_len = try std.math.mul(usize, rows, rhs_cols);
    const a = try allocator.alloc(T, a_len);
    defer allocator.free(a);
    const b = try allocator.alloc(T, b_len);
    defer allocator.free(b);
    for (0..cols) |column| {
        @memcpy(a[column * rows ..][0..rows], coefficients.values[coefficients_offset + column * coefficients_leading_dimension ..][0..rows]);
    }
    for (0..rhs_cols) |column| {
        @memcpy(b[column * rows ..][0..rows], right_hand_sides.values[rhs_offset + column * rhs_leading_dimension ..][0..rows]);
    }
    const s = try allocator.alloc(scalar.Real(T), cols);
    defer allocator.free(s);
    var rwork: ?[]scalar.Real(T) = null;
    if (comptime scalar.isComplex(T)) {
        rwork = try allocator.alloc(scalar.Real(T), try std.math.mul(usize, 5, cols));
    }
    defer if (rwork) |values| allocator.free(values);

    var query: [1]T = undefined;
    var rank: c_int = 0;
    var info: c_int = 0;
    callGelss(T, m, n, nrhs, a.ptr, b.ptr, s.ptr, rcond, &rank, &query, -1, if (rwork) |values| values.ptr else null, &info);
    if (info != 0) return error.BackendFailure;
    const suggested = if (comptime scalar.isComplex(T)) query[0].re else query[0];
    if (!std.math.isFinite(suggested) or suggested < 1) return error.BackendFailure;
    const work_len: usize = @intFromFloat(@ceil(suggested));
    const lwork = try asCInt(work_len);
    const work = try allocator.alloc(T, work_len);
    defer allocator.free(work);

    callGelss(T, m, n, nrhs, a.ptr, b.ptr, s.ptr, rcond, &rank, work.ptr, lwork, if (rwork) |values| values.ptr else null, &info);
    if (info < 0) return error.BackendFailure;
    if (info > 0) return error.NonConvergent;
    if (rank < 0) return error.BackendFailure;

    for (0..rhs_cols) |column| {
        @memcpy(solution.values[column * cols ..][0..cols], b[column * rows ..][0..cols]);
    }
    @memcpy(singular_values.values[0..cols], s);
    return std.math.cast(usize, rank) orelse error.BackendFailure;
}
