const std = @import("std");
const cluster_analysis = @import("cluster_analysis");

fn psi6AtOrientationDegrees(degrees: f64) [2]f64 {
    const phase = 6.0 * degrees * std.math.pi / 180.0;
    return .{ @cos(phase), @sin(phase) };
}

fn expectClusterSizes(
    expected: []const usize,
    frame: cluster_analysis.StructuralFrame,
    options: cluster_analysis.Options,
) !void {
    const result = try cluster_analysis.analyzeStructuralFrame(
        std.testing.allocator,
        frame,
        options,
    );
    defer result.deinit(std.testing.allocator);
    try std.testing.expectEqual(expected.len + 1, result.offsets.len);
    const actual = try std.testing.allocator.alloc(usize, expected.len);
    defer std.testing.allocator.free(actual);
    for (actual, 0..) |*size, index| {
        size.* = result.offsets[index + 1] - result.offsets[index];
    }
    std.mem.sort(usize, actual, {}, comptime std.sort.asc(usize));
    try std.testing.expectEqualSlices(usize, expected, actual);
    try std.testing.expectEqual(result.points.len, result.offsets[result.offsets.len - 1]);
}

test "schema rejects invalid structural controls" {
    try std.testing.expectError(
        error.InvalidPsi6Threshold,
        cluster_analysis.schema.validateOptions(.{ .psi6_minimum = 1.1 }),
    );
    try std.testing.expectError(
        error.InvalidMisorientation,
        cluster_analysis.schema.validateOptions(.{ .misorientation_degrees = 31.0 }),
    );
    try std.testing.expectError(
        error.InvalidMinimumParticles,
        cluster_analysis.schema.validateOptions(.{ .minimum_particles = 1 }),
    );
}

test "FFI exposes point-cluster ABI version three and releases an empty result" {
    try std.testing.expectEqual(@as(u32, 3), cluster_analysis.ffi.cluster_analysis_api_version());
    var output = std.mem.zeroes(cluster_analysis.ffi.CResult);
    cluster_analysis.ffi.cluster_analysis_release(&output);
    try std.testing.expectEqual(@as(?*anyopaque, null), output.owner);
}

test "cluster result contains the actual center coordinates" {
    const points = [_][2]f64{ .{ 0.0, 0.0 }, .{ 1.0, 0.0 }, .{ 2.0, 0.0 } };
    const frame = cluster_analysis.StructuralFrame{
        .points = &points,
        .psi6 = &.{ .{ 1.0, 0.0 }, .{ 1.0, 0.0 }, .{ 1.0, 0.0 } },
        .eligible = &.{ true, true, true },
        .periods = .{ 100.0, 100.0 },
    };
    const result = try cluster_analysis.analyzeStructuralFrame(
        std.testing.allocator,
        frame,
        .{},
    );
    defer result.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(usize, &.{ 0, 3 }, result.offsets);
    try std.testing.expectEqualSlices([2]f64, &points, result.points);
}

test "two disconnected components return two ragged clusters" {
    const frame = cluster_analysis.StructuralFrame{
        .points = &.{
            .{ 0.0, 0.0 },  .{ 1.0, 0.0 },
            .{ 10.0, 0.0 }, .{ 11.0, 0.0 },
            .{ 12.0, 0.0 },
        },
        .psi6 = &.{
            .{ 1.0, 0.0 }, .{ 1.0, 0.0 },
            .{ 1.0, 0.0 }, .{ 1.0, 0.0 },
            .{ 1.0, 0.0 },
        },
        .eligible = &.{ true, true, true, true, true },
        .periods = .{ 100.0, 100.0 },
    };
    try expectClusterSizes(&.{ 2, 3 }, frame, .{});
}

test "periodic seams still connect center clusters" {
    const axial = cluster_analysis.StructuralFrame{
        .points = &.{ .{ 0.2, 5.0 }, .{ 9.8, 5.0 } },
        .psi6 = &.{ .{ 1.0, 0.0 }, .{ 1.0, 0.0 } },
        .eligible = &.{ true, true },
        .periods = .{ 10.0, 20.0 },
    };
    try expectClusterSizes(&.{2}, axial, .{});
    const circumferential = cluster_analysis.StructuralFrame{
        .points = &.{ .{ 5.0, 0.2 }, .{ 5.0, 19.8 } },
        .psi6 = &.{ .{ 1.0, 0.0 }, .{ 1.0, 0.0 } },
        .eligible = &.{ true, true },
        .periods = .{ 10.0, 20.0 },
    };
    try expectClusterSizes(&.{2}, circumferential, .{});
}

test "periodic unwrapping follows accepted orientation edges only" {
    const points = [_][2]f64{
        .{ 0.0, 0.2 },
        .{ 0.0, 2.0 },
        .{ 0.0, 3.8 },
        .{ 0.0, 5.6 },
        .{ 0.0, 7.4 },
        .{ 0.0, 9.2 },
    };
    const psi = [_][2]f64{
        psi6AtOrientationDegrees(0.0),
        psi6AtOrientationDegrees(4.0),
        psi6AtOrientationDegrees(8.0),
        psi6AtOrientationDegrees(12.0),
        psi6AtOrientationDegrees(16.0),
        psi6AtOrientationDegrees(20.0),
    };
    const frame = cluster_analysis.StructuralFrame{
        .points = &points,
        .psi6 = &psi,
        .eligible = &.{ true, true, true, true, true, true },
        .periods = .{ 20.0, 10.0 },
    };
    const result = try cluster_analysis.analyzeStructuralFrame(
        std.testing.allocator,
        frame,
        .{
            .particle_diameter = 1.0,
            .neighbor_radius_diameters = 1.9,
            .misorientation_degrees = 5.0,
        },
    );
    defer result.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(usize, &.{ 0, 6 }, result.offsets);
    var minimum = result.points[0][1];
    var maximum = minimum;
    for (result.points[1..]) |point| {
        minimum = @min(minimum, point[1]);
        maximum = @max(maximum, point[1]);
    }
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), maximum - minimum, 1.0e-12);
}

test "filters remain applied before returning cluster coordinates" {
    const psi = [_][2]f64{
        psi6AtOrientationDegrees(0.0),
        psi6AtOrientationDegrees(6.0),
    };
    const frame = cluster_analysis.StructuralFrame{
        .points = &.{ .{ 0.0, 0.0 }, .{ 1.0, 0.0 } },
        .psi6 = &psi,
        .eligible = &.{ true, true },
        .periods = .{ 100.0, 100.0 },
    };
    try expectClusterSizes(&.{}, frame, .{ .misorientation_degrees = 5.0 });
    try expectClusterSizes(
        &.{},
        .{
            .points = &.{ .{ 0.0, 0.0 }, .{ 1.0, 0.0 } },
            .psi6 = &.{ .{ 1.0, 0.0 }, .{ 1.0, 0.0 } },
            .eligible = &.{ true, true },
            .periods = .{ 100.0, 100.0 },
        },
        .{ .minimum_particles = 3 },
    );
}
