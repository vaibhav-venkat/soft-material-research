const std = @import("std");
const linalg = @import("linalg");

const Complex32 = linalg.Complex32;
const Complex64 = linalg.Complex64;
const Real = linalg.Real;
const Context = linalg.Context;
const Vector = linalg.Vector;
const Matrix = linalg.Matrix;
const copy = linalg.copy;
const swap = linalg.swap;
const scale = linalg.scale;
const scaleReal = linalg.scaleReal;
const axpy = linalg.axpy;
const dot = linalg.dot;
const dotu = linalg.dotu;
const dotc = linalg.dotc;
const norm2 = linalg.norm2;
const absSum = linalg.absSum;
const indexAbsMax = linalg.indexAbsMax;
const gemvInto = linalg.gemvInto;
const matvec = linalg.matvec;
const gerInto = linalg.gerInto;
const geruInto = linalg.geruInto;
const gercInto = linalg.gercInto;
const matmul = linalg.matmul;
const svd = linalg.svd;
const leastSquares = linalg.leastSquares;

fn isComplex(comptime T: type) bool {
    return T == Complex32 or T == Complex64;
}

fn scalarZero(comptime T: type) T {
    return if (comptime isComplex(T)) T.init(0, 0) else 0;
}

fn expectComplexApprox(comptime T: type, expected: T, actual: T, tolerance: Real(T)) !void {
    try std.testing.expectApproxEqAbs(expected.re, actual.re, tolerance);
    try std.testing.expectApproxEqAbs(expected.im, actual.im, tolerance);
}

fn testRealLevelOne(comptime T: type, tolerance: T) !void {
    var context = try Context.init(std.testing.allocator, .{});
    defer context.deinit();
    const x_values = [_]T{ 1, -2, 3 };
    const y_values = [_]T{ 4, 5, 6 };
    var x = try Vector(T).fromHost(&context, &x_values);
    defer x.deinit();
    var y = try Vector(T).fromHost(&context, &y_values);
    defer y.deinit();

    try std.testing.expectApproxEqAbs(@as(T, 12), try dot(T, x.constView(), y.constView()), tolerance);
    try std.testing.expectApproxEqAbs(@sqrt(@as(T, 14)), try norm2(T, x.constView()), tolerance);
    try std.testing.expectApproxEqAbs(@as(T, 6), try absSum(T, x.constView()), tolerance);
    try std.testing.expectEqual(@as(usize, 2), try indexAbsMax(T, x.constView()));

    try scale(T, 2, y.view());
    try axpy(T, -1, x.constView(), y.view());
    var output: [3]T = undefined;
    try y.copyToHost(&output);
    try std.testing.expectEqualSlices(T, &[_]T{ 7, 12, 9 }, &output);

    try copy(T, x.constView(), y.view());
    try swap(T, x.view(), y.view());
    try x.copyToHost(&output);
    try std.testing.expectEqualSlices(T, &x_values, &output);
}

fn testComplexLevelOne(comptime T: type, tolerance: Real(T)) !void {
    var context = try Context.init(std.testing.allocator, .{});
    defer context.deinit();
    const x_values = [_]T{ .init(1, 2), .init(-3, 1) };
    const y_values = [_]T{ .init(2, -1), .init(1, 4) };
    var x = try Vector(T).fromHost(&context, &x_values);
    defer x.deinit();
    var y = try Vector(T).fromHost(&context, &y_values);
    defer y.deinit();

    try expectComplexApprox(T, T.init(-3, -8), try dotu(T, x.constView(), y.constView()), tolerance);
    try expectComplexApprox(T, T.init(1, -18), try dotc(T, x.constView(), y.constView()), tolerance);
    try std.testing.expectApproxEqAbs(@sqrt(@as(Real(T), 15)), try norm2(T, x.constView()), tolerance);
    try std.testing.expectApproxEqAbs(@as(Real(T), 7), try absSum(T, x.constView()), tolerance);
    try std.testing.expectEqual(@as(usize, 1), try indexAbsMax(T, x.constView()));

    try scaleReal(T, 2, x.view());
    try scale(T, T.init(0, 0.5), x.view());
    try axpy(T, T.init(1, 0), y.constView(), x.view());
    var output: [2]T = undefined;
    try x.copyToHost(&output);
    try expectComplexApprox(T, T.init(0, 0), output[0], tolerance);
    try expectComplexApprox(T, T.init(0, 1), output[1], tolerance);
}

fn testRealMatrixBlas(comptime T: type, tolerance: T) !void {
    var context = try Context.init(std.testing.allocator, .{});
    defer context.deinit();
    const a_values = [_]T{ 1, 4, 2, 5, 3, 6 };
    var a = try Matrix(T).fromHost(&context, 2, 3, &a_values);
    defer a.deinit();
    var x = try Vector(T).fromHost(&context, &[_]T{ 1, 2, 1 });
    defer x.deinit();
    var y = try Vector(T).fromHost(&context, &[_]T{ 1, 1 });
    defer y.deinit();
    try gemvInto(T, .none, 2, a.constView(), x.constView(), 0.5, y.view());
    var y_output: [2]T = undefined;
    try y.copyToHost(&y_output);
    try std.testing.expectApproxEqAbs(@as(T, 16.5), y_output[0], tolerance);
    try std.testing.expectApproxEqAbs(@as(T, 40.5), y_output[1], tolerance);

    var trans_x = try Vector(T).fromHost(&context, &[_]T{ 1, 2 });
    defer trans_x.deinit();
    var trans_y = try matvec(T, .transpose, a.constView(), trans_x.constView());
    defer trans_y.deinit();
    var trans_output: [3]T = undefined;
    try trans_y.copyToHost(&trans_output);
    try std.testing.expectEqualSlices(T, &[_]T{ 9, 12, 15 }, &trans_output);

    const b_values = [_]T{ 1, 0, 1, 2, 1, 0 };
    var b = try Matrix(T).fromHost(&context, 3, 2, &b_values);
    defer b.deinit();
    var product = try matmul(T, .none, .none, a.constView(), b.constView());
    defer product.deinit();
    var product_output: [4]T = undefined;
    try product.copyToHost(&product_output);
    try std.testing.expectEqualSlices(T, &[_]T{ 4, 10, 4, 13 }, &product_output);

    var outer_x = try Vector(T).fromHost(&context, &[_]T{ 1, 2 });
    defer outer_x.deinit();
    var outer_y = try Vector(T).fromHost(&context, &[_]T{ 3, 4, 5 });
    defer outer_y.deinit();
    var outer = try Matrix(T).fromHost(&context, 2, 3, &[_]T{ 0, 0, 0, 0, 0, 0 });
    defer outer.deinit();
    try gerInto(T, 1, outer_x.constView(), outer_y.constView(), outer.view());
    var outer_output: [6]T = undefined;
    try outer.copyToHost(&outer_output);
    try std.testing.expectEqualSlices(T, &[_]T{ 3, 6, 4, 8, 5, 10 }, &outer_output);
}

fn testComplexMatrixBlas(comptime T: type, tolerance: Real(T)) !void {
    var context = try Context.init(std.testing.allocator, .{});
    defer context.deinit();
    const a_values = [_]T{ .init(1, 1), .init(2, 0), .init(0, 1), .init(3, -1) };
    var a = try Matrix(T).fromHost(&context, 2, 2, &a_values);
    defer a.deinit();
    var x = try Vector(T).fromHost(&context, &[_]T{ .init(1, 0), .init(0, 1) });
    defer x.deinit();
    var y = try matvec(T, .conjugate_transpose, a.constView(), x.constView());
    defer y.deinit();
    var y_output: [2]T = undefined;
    try y.copyToHost(&y_output);
    try expectComplexApprox(T, T.init(1, 1), y_output[0], tolerance);
    try expectComplexApprox(T, T.init(-1, 2), y_output[1], tolerance);

    var identity = try Matrix(T).fromHost(&context, 2, 2, &[_]T{ .init(1, 0), .init(0, 0), .init(0, 0), .init(1, 0) });
    defer identity.deinit();
    var product = try matmul(T, .none, .none, a.constView(), identity.constView());
    defer product.deinit();
    var product_output: [4]T = undefined;
    try product.copyToHost(&product_output);
    for (a_values, product_output) |expected, actual| try expectComplexApprox(T, expected, actual, tolerance);

    var outer_x = try Vector(T).fromHost(&context, &[_]T{.init(1, 1)});
    defer outer_x.deinit();
    var outer_y = try Vector(T).fromHost(&context, &[_]T{.init(2, 1)});
    defer outer_y.deinit();
    var outer_u = try Matrix(T).fromHost(&context, 1, 1, &[_]T{.init(0, 0)});
    defer outer_u.deinit();
    var outer_c = try Matrix(T).fromHost(&context, 1, 1, &[_]T{.init(0, 0)});
    defer outer_c.deinit();
    try geruInto(T, T.init(1, 0), outer_x.constView(), outer_y.constView(), outer_u.view());
    try gercInto(T, T.init(1, 0), outer_x.constView(), outer_y.constView(), outer_c.view());
    var value: [1]T = undefined;
    try outer_u.copyToHost(&value);
    try expectComplexApprox(T, T.init(1, 3), value[0], tolerance);
    try outer_c.copyToHost(&value);
    try expectComplexApprox(T, T.init(3, 1), value[0], tolerance);
}

fn testSvd(comptime T: type, tolerance: Real(T)) !void {
    var context = try Context.init(std.testing.allocator, .{});
    defer context.deinit();
    const values = if (comptime isComplex(T))
        [_]T{ .init(1, 1), .init(2, 0), .init(0, -1), .init(0, 0), .init(1, -1), .init(3, 0) }
    else
        [_]T{ 1, 2, -1, 0, 1, 3 };
    var input = try Matrix(T).fromHost(&context, 3, 2, &values);
    defer input.deinit();
    var result = try svd(T, input.constView(), .{ .mode = .thin });
    defer result.deinit();

    try std.testing.expect(result.u != null);
    try std.testing.expect(result.vh != null);
    var singular_host: [2]Real(T) = undefined;
    try result.singular_values.copyToHost(&singular_host);
    try std.testing.expect(singular_host[0] >= singular_host[1]);
    try std.testing.expect(singular_host[1] >= 0);

    var sigma_values = [_]T{ scalarZero(T), scalarZero(T), scalarZero(T), scalarZero(T) };
    sigma_values[0] = if (comptime isComplex(T)) T.init(singular_host[0], 0) else singular_host[0];
    sigma_values[3] = if (comptime isComplex(T)) T.init(singular_host[1], 0) else singular_host[1];
    var sigma = try Matrix(T).fromHost(&context, 2, 2, &sigma_values);
    defer sigma.deinit();
    var scaled_u = try matmul(T, .none, .none, result.u.?.constView(), sigma.constView());
    defer scaled_u.deinit();
    var reconstructed = try matmul(T, .none, .none, scaled_u.constView(), result.vh.?.constView());
    defer reconstructed.deinit();
    var reconstructed_host: [6]T = undefined;
    try reconstructed.copyToHost(&reconstructed_host);
    for (values, reconstructed_host) |expected, actual| {
        if (comptime isComplex(T)) {
            try expectComplexApprox(T, expected, actual, tolerance);
        } else {
            try std.testing.expectApproxEqAbs(expected, actual, tolerance);
        }
    }

    var values_only = try svd(T, input.constView(), .{ .mode = .values_only });
    defer values_only.deinit();
    try std.testing.expect(values_only.u == null);
    try std.testing.expect(values_only.vh == null);
    var values_only_host: [2]Real(T) = undefined;
    try values_only.singular_values.copyToHost(&values_only_host);
    for (singular_host, values_only_host) |expected, actual| try std.testing.expectApproxEqAbs(expected, actual, tolerance);
}

fn testLeastSquares(comptime T: type, tolerance: Real(T)) !void {
    var context = try Context.init(std.testing.allocator, .{});
    defer context.deinit();
    const coefficient_values = if (comptime isComplex(T))
        [_]T{ .init(1, 1), .init(0, 0), .init(1, 0), .init(0, 0), .init(1, -1), .init(1, 0) }
    else
        [_]T{ 1, 0, 1, 0, 1, 1 };
    const expected_values = if (comptime isComplex(T))
        [_]T{ .init(2, -1), .init(3, 2), .init(-1, 1), .init(4, -2) }
    else
        [_]T{ 2, 3, -1, 4 };
    var coefficients = try Matrix(T).fromHost(&context, 3, 2, &coefficient_values);
    defer coefficients.deinit();
    var expected = try Matrix(T).fromHost(&context, 2, 2, &expected_values);
    defer expected.deinit();
    var right_hand_sides = try matmul(T, .none, .none, coefficients.constView(), expected.constView());
    defer right_hand_sides.deinit();

    var result = try leastSquares(T, coefficients.constView(), right_hand_sides.constView(), .{});
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.rank);
    var solution_host: [4]T = undefined;
    try result.solution.copyToHost(&solution_host);
    for (expected_values, solution_host) |expected_value, actual| {
        if (comptime isComplex(T)) {
            try expectComplexApprox(T, expected_value, actual, tolerance);
        } else {
            try std.testing.expectApproxEqAbs(expected_value, actual, tolerance);
        }
    }
    var residual_host: [2]Real(T) = undefined;
    try result.residual_norms.copyToHost(&residual_host);
    for (residual_host) |residual| try std.testing.expectApproxEqAbs(@as(Real(T), 0), residual, tolerance * 10);
}

fn testRankDeficientLeastSquares(comptime T: type, tolerance: Real(T)) !void {
    var context = try Context.init(std.testing.allocator, .{});
    defer context.deinit();
    const coefficients_values = if (comptime isComplex(T))
        [_]T{ .init(1, 0), .init(2, 0), .init(3, 0), .init(2, 0), .init(4, 0), .init(6, 0) }
    else
        [_]T{ 1, 2, 3, 2, 4, 6 };
    const rhs_values = if (comptime isComplex(T))
        [_]T{ .init(3, 1), .init(6, 2), .init(9, 3), .init(-1, 2), .init(-2, 4), .init(-3, 6) }
    else
        [_]T{ 3, 6, 9, -1, -2, -3 };
    var coefficients = try Matrix(T).fromHost(&context, 3, 2, &coefficients_values);
    defer coefficients.deinit();
    var rhs = try Matrix(T).fromHost(&context, 3, 2, &rhs_values);
    defer rhs.deinit();
    var result = try leastSquares(T, coefficients.constView(), rhs.constView(), .{});
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 1), result.rank);
    var residuals: [2]Real(T) = undefined;
    try result.residual_norms.copyToHost(&residuals);
    for (residuals) |residual| try std.testing.expectApproxEqAbs(@as(Real(T), 0), residual, tolerance * 20);
}

test "complex representation and conversions" {
    const value = Complex32.init(2.5, -3.0);
    try std.testing.expectEqual(@sizeOf([2]f32), @sizeOf(Complex32));
    try std.testing.expectEqual(2 * @alignOf(f32), @alignOf(Complex32));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(Complex32, "re"));
    try std.testing.expectEqual(@sizeOf(f32), @offsetOf(Complex32, "im"));
    try std.testing.expectEqual(value, Complex32.fromStd(value.toStd()));
    try std.testing.expectEqual(Complex32.init(2.5, 3.0), value.conjugate());
}

test "backend-owned vector and matrix round trips" {
    var context = try Context.init(std.testing.allocator, .{});
    defer context.deinit();

    const vector_values = [_]Complex64{
        .init(1.0, 2.0),
        .init(-3.0, 4.0),
    };
    var vector = try Vector(Complex64).fromHost(&context, &vector_values);
    defer vector.deinit();
    var vector_output: [2]Complex64 = undefined;
    try vector.copyToHost(&vector_output);
    try std.testing.expectEqualSlices(Complex64, &vector_values, &vector_output);

    const matrix_values = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var matrix = try Matrix(f64).fromHost(&context, 3, 2, &matrix_values);
    defer matrix.deinit();
    const row = try matrix.row(1);
    const column = try matrix.column(1);
    const submatrix = try matrix.subview(1, 0, 2, 2);
    try std.testing.expectEqual(@as(usize, 3), row.stride);
    try std.testing.expectEqual(@as(usize, 3), column.len);
    try std.testing.expectEqual(@as(usize, 1), submatrix.offset);
    try std.testing.expectEqual(@as(usize, 3), submatrix.leading_dimension);

    var matrix_output: [6]f64 = undefined;
    try matrix.copyToHost(&matrix_output);
    try std.testing.expectEqualSlices(f64, &matrix_values, &matrix_output);
}

test "real Level 1 BLAS" {
    try testRealLevelOne(f32, 1.0e-5);
    try testRealLevelOne(f64, 1.0e-12);
}

test "complex Level 1 BLAS" {
    try testComplexLevelOne(Complex32, 1.0e-5);
    try testComplexLevelOne(Complex64, 1.0e-12);
}

test "real matrix BLAS" {
    try testRealMatrixBlas(f32, 1.0e-5);
    try testRealMatrixBlas(f64, 1.0e-12);
}

test "complex matrix BLAS" {
    try testComplexMatrixBlas(Complex32, 1.0e-5);
    try testComplexMatrixBlas(Complex64, 1.0e-12);
}

test "thin and values-only SVD" {
    try testSvd(f32, 1.0e-4);
    try testSvd(f64, 1.0e-11);
    try testSvd(Complex32, 1.0e-4);
    try testSvd(Complex64, 1.0e-11);
}

test "overdetermined least squares" {
    try testLeastSquares(f32, 1.0e-4);
    try testLeastSquares(f64, 1.0e-11);
    try testLeastSquares(Complex32, 1.0e-4);
    try testLeastSquares(Complex64, 1.0e-11);
}

test "rank-deficient least squares and residual norms" {
    try testRankDeficientLeastSquares(f32, 5.0e-4);
    try testRankDeficientLeastSquares(f64, 1.0e-10);
    try testRankDeficientLeastSquares(Complex32, 5.0e-4);
    try testRankDeficientLeastSquares(Complex64, 1.0e-10);
}

test "events and validation failures" {
    var first = try Context.init(std.testing.allocator, .{});
    defer first.deinit();
    var second = try Context.init(std.testing.allocator, .{});
    defer second.deinit();
    var x = try Vector(f64).fromHost(&first, &[_]f64{ 1, 2 });
    defer x.deinit();
    var y = try Vector(f64).fromHost(&second, &[_]f64{ 3, 4 });
    defer y.deinit();
    try std.testing.expectError(error.ContextMismatch, copy(f64, x.constView(), y.view()));

    var event = try first.recordEvent();
    defer event.deinit();
    try event.wait();
    try std.testing.expect(try event.query());

    var wide = try Matrix(f64).fromHost(&first, 1, 2, &[_]f64{ 1, 2 });
    defer wide.deinit();
    try std.testing.expectError(error.WideMatrixUnsupported, svd(f64, wide.constView(), .{}));
    var rhs = try Matrix(f64).fromHost(&first, 1, 1, &[_]f64{1});
    defer rhs.deinit();
    var square = try Matrix(f64).fromHost(&first, 1, 1, &[_]f64{1});
    defer square.deinit();
    try std.testing.expectError(error.InvalidTolerance, leastSquares(f64, square.constView(), rhs.constView(), .{ .rcond = -1 }));
}
