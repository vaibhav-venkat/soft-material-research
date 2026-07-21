const std = @import("std");
const cluster_analysis = @import("cluster_analysis");

test "public contracts default to the Rust-compatible square-root ratio" {
    const options = cluster_analysis.Options{};
    try std.testing.expectEqual(cluster_analysis.RatioMode.sqrt_area_fraction, options.ratio_mode);
    try std.testing.expectEqual(@as(f64, 0.7), options.psi6_minimum);
    try std.testing.expectEqual(@as(f64, 5.0), options.misorientation_degrees);
    try std.testing.expectEqual(@as(f64, 1.7272), options.neighbor_radius_diameters);
    try std.testing.expectEqual(@as(usize, 2), options.minimum_particles);
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

test "C k-d tree wrapper returns indices in a two-dimensional radius" {
    var tree = try cluster_analysis.backend.kdtree.Tree2.init();
    defer tree.deinit();
    try tree.insert(.{ 0.0, 0.0 }, 0);
    try tree.insert(.{ 0.5, 0.0 }, 1);
    try tree.insert(.{ 4.0, 0.0 }, 2);
    const found = try tree.within(std.testing.allocator, .{ 0.0, 0.0 }, 0.75);
    defer std.testing.allocator.free(found);
    std.mem.sort(usize, found, {}, comptime std.sort.asc(usize));
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, found);
}

test "FFI exposes version two and safely releases an empty result" {
    try std.testing.expectEqual(@as(u32, 2), cluster_analysis.ffi.cluster_analysis_api_version());
    var output = std.mem.zeroes(cluster_analysis.ffi.CResult);
    cluster_analysis.ffi.cluster_analysis_release(&output);
    try std.testing.expectEqual(@as(?*anyopaque, null), output.owner);
}

test "structural cluster returns the Rust-compatible sqrt(A over SA)" {
    const frame = cluster_analysis.StructuralFrame{
        .points = &.{ .{ 0.0, 0.0 }, .{ 1.0, 0.0 } },
        .psi6 = &.{ .{ 1.0, 0.0 }, .{ 1.0, 0.0 } },
        .eligible = &.{ true, true },
        .periods = .{ 10.0, 20.0 },
    };
    const result = try cluster_analysis.clusters.analyzeStructuralFrame(
        std.testing.allocator,
        frame,
        .{ .ratio_mode = .sqrt_area_fraction },
    );
    defer result.deinit(std.testing.allocator);
    const expected = @sqrt((2.0 * std.math.pi / 4.0) / 200.0);
    try std.testing.expectEqual(@as(usize, 1), result.ratios.len);
    try std.testing.expectApproxEqAbs(expected, result.ratios[0], 1.0e-12);
}

test "structural cluster can return the regular A over SA ratio" {
    const frame = cluster_analysis.StructuralFrame{
        .points = &.{ .{ 0.0, 0.0 }, .{ 1.0, 0.0 } },
        .psi6 = &.{ .{ 1.0, 0.0 }, .{ 1.0, 0.0 } },
        .eligible = &.{ true, true },
        .periods = .{ 10.0, 20.0 },
    };
    const result = try cluster_analysis.clusters.analyzeStructuralFrame(
        std.testing.allocator,
        frame,
        .{ .ratio_mode = .area_fraction },
    );
    defer result.deinit(std.testing.allocator);
    const expected = (2.0 * std.math.pi / 4.0) / 200.0;
    try std.testing.expectEqual(@as(usize, 1), result.ratios.len);
    try std.testing.expectApproxEqAbs(expected, result.ratios[0], 1.0e-12);
}
