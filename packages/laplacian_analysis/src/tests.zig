const std = @import("std");
const laplacian = @import("laplacian_analysis");

fn correlation() laplacian.dynamics_analysis.CorrelationSeries {
    return .{
        .lag_indices = @constCast(&[_]usize{ 0, 1, 2 }),
        .lag_times = @constCast(&[_]f64{ 0.0, 1.0, 2.0 }),
        .pearson = @constCast(&[_]f64{ 1.0, 0.5, 0.0 }),
        .origin_counts = @constCast(&[_]usize{ 4, 3, 2 }),
    };
}

fn expectApproxSlice(expected: []const f64, actual: []const f64) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |expected_value, actual_value| {
        try std.testing.expectApproxEqAbs(expected_value, actual_value, 1.0e-9);
    }
}

test "public defaults mirror the Rust numerical controls" {
    const value = laplacian.Options{};
    try std.testing.expectEqual(@as(usize, 161), value.transform.r_points);
    try std.testing.expectEqual(@as(usize, 241), value.transform.omega_points);
    try std.testing.expectEqual(@as(f64, 0.05), value.fit.soft_l1_scale);
    try std.testing.expectEqual(@as(usize, 20_000), value.fit.maximum_evaluations);
}

test "schema accepts a finite uniformly spaced correlation" {
    try std.testing.expectApproxEqAbs(
        @as(f64, 1.0),
        try laplacian.schema.validateCorrelation(correlation()),
        1.0e-12,
    );
    try laplacian.schema.validateTransformOptions(.{});
    try laplacian.schema.validateFitOptions(.{});
}

test "transform axes use duration and Nyquist defaults" {
    const axes = try laplacian.transformAxes(
        std.testing.allocator,
        correlation(),
        .{ .r_points = 3, .omega_points = 3 },
    );
    defer axes.deinit(std.testing.allocator);
    try expectApproxSlice(&.{ -5.0, -2.5, 0.0 }, axes.r);
    try expectApproxSlice(&.{ -std.math.pi, 0.0, std.math.pi }, axes.omega);
}

test "preferred axes contain only negative rates and positive frequencies" {
    const axes = try laplacian.preferredAxes(
        std.testing.allocator,
        correlation(),
        .{ .r_points = 5, .omega_points = 5 },
    );
    defer axes.deinit(std.testing.allocator);
    try std.testing.expect(axes.r.len >= 2);
    try std.testing.expect(axes.omega.len >= 2);
    for (axes.r) |value| try std.testing.expect(value < 0.0);
    for (axes.omega) |value| try std.testing.expect(value > 0.0);
}

test "complex Laplace grid uses omega-major row order" {
    var context = try laplacian.dynamics_analysis.backend.Context.init(
        std.testing.allocator,
        .{},
    );
    defer context.deinit();
    const axes = laplacian.TransformAxes{
        .r = @constCast(&[_]f64{ 0.0, 0.5 }),
        .omega = @constCast(&[_]f64{ 0.0, 1.0 }),
    };
    const grid = try laplacian.analyzeLaplace(
        std.testing.allocator,
        &context,
        .{
            .lag_indices = @constCast(&[_]usize{ 0, 1, 2 }),
            .lag_times = @constCast(&[_]f64{ 0.0, 1.0, 2.0 }),
            .pearson = @constCast(&[_]f64{ 1.0, 1.0, 1.0 }),
            .origin_counts = @constCast(&[_]usize{ 3, 2, 1 }),
        },
        axes,
    );
    defer grid.deinit(std.testing.allocator);
    try std.testing.expectEqual([2]usize{ 2, 2 }, grid.shape);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), grid.values_real[0], 1.0e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), grid.values_imag[0], 1.0e-9);
}

test "preferred omega reports the maximum log-magnitude coordinate" {
    var context = try laplacian.dynamics_analysis.backend.Context.init(
        std.testing.allocator,
        .{},
    );
    defer context.deinit();
    const estimate = try laplacian.preferredCoordinate(
        std.testing.allocator,
        &context,
        correlation(),
        .omega,
        &.{ 0.25, 1.0 },
    );
    try std.testing.expectEqual(laplacian.PreferredAxis.omega, estimate.axis);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), estimate.coordinate, 1.0e-9);
    try std.testing.expectEqual(@as(usize, 1), estimate.replicate_count);
}

test "damped cosine enforces unit value at zero lag" {
    const values = try laplacian.dampedCosine(
        std.testing.allocator,
        &.{ 0.0, 1.0 },
        .{ 1.0, 0.0, std.math.pi, 0.0 },
    );
    defer std.testing.allocator.free(values);
    try expectApproxSlice(&.{ 1.0, -1.0 }, values);
}

test "analytic damped-cosine Jacobian has sample-major shape and values" {
    const jacobian = try laplacian.dampedCosineJacobian(
        std.testing.allocator,
        &.{ 0.0, 1.0 },
        .{ 1.0, 0.0, 0.0, 0.0 },
    );
    defer std.testing.allocator.free(jacobian);
    try expectApproxSlice(&.{ 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0 }, jacobian);
}

test "constrained fit recovers a noiseless damped cosine" {
    var context = try laplacian.dynamics_analysis.backend.Context.init(
        std.testing.allocator,
        .{},
    );
    defer context.deinit();
    const fitted = try laplacian.fitDampedCosine(
        std.testing.allocator,
        &context,
        .{
            .lag_indices = @constCast(&[_]usize{ 0, 1, 2, 3, 4, 5, 6, 7 }),
            .lag_times = @constCast(&[_]f64{ 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 }),
            .pearson = @constCast(&[_]f64{
                1.0,
                0.774463892631,
                0.420787858905,
                0.048616920318,
                -0.252405815308,
                -0.428821276107,
                -0.467639342859,
                -0.390373226005,
            }),
            .origin_counts = @constCast(&[_]usize{ 8, 7, 6, 5, 4, 3, 2, 1 }),
        },
        &.{ 0.5, 1.0, 1.5 },
        .{},
    );
    defer fitted.deinit(std.testing.allocator);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), fitted.amplitude, 1.0e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), fitted.rate, 1.0e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), fitted.omega, 1.0e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), fitted.phase, 1.0e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), fitted.r_squared, 1.0e-8);
    try std.testing.expect(fitted.converged);
}

test "top-level analysis routes empty datasets through dynamics input validation" {
    var context = try laplacian.dynamics_analysis.backend.Context.init(
        std.testing.allocator,
        .{},
    );
    defer context.deinit();
    try std.testing.expectError(
        error.NoInput,
        laplacian.analyze(
            std.testing.allocator,
            &context,
            .{ .static = .{ .path = "missing" }, .shard_paths = &.{} },
            .{},
        ),
    );
}
