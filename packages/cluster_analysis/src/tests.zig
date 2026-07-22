const std = @import("std");
const cluster_analysis = @import("cluster_analysis");

fn psi6AtOrientationDegrees(degrees: f64) [2]f64 {
    const phase = 6.0 * degrees * std.math.pi / 180.0;
    return .{ @cos(phase), @sin(phase) };
}

fn expectedRatio(
    particle_count: usize,
    particle_diameter: f64,
    periods: [2]f64,
    mode: cluster_analysis.RatioMode,
) f64 {
    const count: f64 = @floatFromInt(particle_count);
    const area_fraction = count * std.math.pi * particle_diameter * particle_diameter /
        (4.0 * periods[0] * periods[1]);
    return switch (mode) {
        .area_fraction => area_fraction,
        .sqrt_area_fraction => @sqrt(area_fraction),
    };
}

fn expectRatios(
    expected: []const f64,
    frame: cluster_analysis.StructuralFrame,
    options: cluster_analysis.Options,
) !void {
    const result = try cluster_analysis.clusters.analyzeStructuralFrame(
        std.testing.allocator,
        frame,
        options,
    );
    defer result.deinit(std.testing.allocator);
    std.mem.sort(f64, result.ratios, {}, comptime std.sort.asc(f64));
    try std.testing.expectEqual(expected.len, result.ratios.len);
    for (expected, result.ratios) |expected_value, actual_value| {
        try std.testing.expectApproxEqAbs(expected_value, actual_value, 1.0e-12);
    }
}

fn expectNoRatios(
    frame: cluster_analysis.StructuralFrame,
    options: cluster_analysis.Options,
) !void {
    try expectRatios(&.{}, frame, options);
}

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

test "three-particle neighbor chain forms one transitive component" {
    const periods = [2]f64{ 100.0, 100.0 };
    const frame = cluster_analysis.StructuralFrame{
        .points = &.{ .{ 0.0, 0.0 }, .{ 1.5, 0.0 }, .{ 3.0, 0.0 } },
        .psi6 = &.{ .{ 1.0, 0.0 }, .{ 1.0, 0.0 }, .{ 1.0, 0.0 } },
        .eligible = &.{ true, true, true },
        .periods = periods,
    };
    try expectRatios(
        &.{expectedRatio(3, 1.0, periods, .sqrt_area_fraction)},
        frame,
        .{},
    );
}

test "two disconnected components return two independently sized samples" {
    const periods = [2]f64{ 100.0, 100.0 };
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
        .periods = periods,
    };
    try expectRatios(
        &.{
            expectedRatio(2, 1.0, periods, .area_fraction),
            expectedRatio(3, 1.0, periods, .area_fraction),
        },
        frame,
        .{ .ratio_mode = .area_fraction },
    );
}

test "particles farther than the neighbor cutoff do not form a cluster" {
    const frame = cluster_analysis.StructuralFrame{
        .points = &.{ .{ 0.0, 0.0 }, .{ 2.0, 0.0 } },
        .psi6 = &.{ .{ 1.0, 0.0 }, .{ 1.0, 0.0 } },
        .eligible = &.{ true, true },
        .periods = .{ 100.0, 100.0 },
    };
    try expectNoRatios(frame, .{ .neighbor_radius_diameters = 1.5 });
}

test "an ineligible bridge cannot connect two ordered particles" {
    const frame = cluster_analysis.StructuralFrame{
        .points = &.{ .{ 0.0, 0.0 }, .{ 1.5, 0.0 }, .{ 3.0, 0.0 } },
        .psi6 = &.{ .{ 1.0, 0.0 }, .{ 1.0, 0.0 }, .{ 1.0, 0.0 } },
        .eligible = &.{ true, false, true },
        .periods = .{ 100.0, 100.0 },
    };
    try expectNoRatios(frame, .{});
}

test "nearby particles outside lattice-orientation tolerance remain separate" {
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
    try expectNoRatios(frame, .{ .misorientation_degrees = 5.0 });
}

test "sixfold orientation comparison wraps across plus and minus thirty degrees" {
    const periods = [2]f64{ 100.0, 100.0 };
    const psi = [_][2]f64{
        psi6AtOrientationDegrees(29.0),
        psi6AtOrientationDegrees(-29.0),
    };
    const frame = cluster_analysis.StructuralFrame{
        .points = &.{ .{ 0.0, 0.0 }, .{ 1.0, 0.0 } },
        .psi6 = &psi,
        .eligible = &.{ true, true },
        .periods = periods,
    };
    try expectRatios(
        &.{expectedRatio(2, 1.0, periods, .area_fraction)},
        frame,
        .{ .misorientation_degrees = 5.0, .ratio_mode = .area_fraction },
    );
}

test "neighbors connect across the axial periodic seam" {
    const periods = [2]f64{ 10.0, 20.0 };
    const frame = cluster_analysis.StructuralFrame{
        .points = &.{ .{ 0.2, 5.0 }, .{ 9.8, 5.0 } },
        .psi6 = &.{ .{ 1.0, 0.0 }, .{ 1.0, 0.0 } },
        .eligible = &.{ true, true },
        .periods = periods,
    };
    try expectRatios(
        &.{expectedRatio(2, 1.0, periods, .sqrt_area_fraction)},
        frame,
        .{},
    );
}

test "neighbors connect across the circumferential periodic seam" {
    const periods = [2]f64{ 10.0, 20.0 };
    const frame = cluster_analysis.StructuralFrame{
        .points = &.{ .{ 5.0, 0.2 }, .{ 5.0, 19.8 } },
        .psi6 = &.{ .{ 1.0, 0.0 }, .{ 1.0, 0.0 } },
        .eligible = &.{ true, true },
        .periods = periods,
    };
    try expectRatios(
        &.{expectedRatio(2, 1.0, periods, .sqrt_area_fraction)},
        frame,
        .{},
    );
}

test "periodic image duplicates do not inflate component particle count" {
    const periods = [2]f64{ 1.0, 1.0 };
    const frame = cluster_analysis.StructuralFrame{
        .points = &.{ .{ 0.1, 0.1 }, .{ 0.2, 0.1 } },
        .psi6 = &.{ .{ 1.0, 0.0 }, .{ 1.0, 0.0 } },
        .eligible = &.{ true, true },
        .periods = periods,
    };
    try expectRatios(
        &.{expectedRatio(2, 1.0, periods, .area_fraction)},
        frame,
        .{ .ratio_mode = .area_fraction },
    );
}

test "minimum particle count filters a smaller connected component" {
    const frame = cluster_analysis.StructuralFrame{
        .points = &.{ .{ 0.0, 0.0 }, .{ 1.0, 0.0 } },
        .psi6 = &.{ .{ 1.0, 0.0 }, .{ 1.0, 0.0 } },
        .eligible = &.{ true, true },
        .periods = .{ 100.0, 100.0 },
    };
    try expectNoRatios(frame, .{ .minimum_particles = 3 });
}

test "particle diameter scales both cutoff and reported cluster area" {
    const periods = [2]f64{ 20.0, 30.0 };
    const frame = cluster_analysis.StructuralFrame{
        .points = &.{ .{ 0.0, 0.0 }, .{ 1.5, 0.0 } },
        .psi6 = &.{ .{ 1.0, 0.0 }, .{ 1.0, 0.0 } },
        .eligible = &.{ true, true },
        .periods = periods,
    };
    try expectRatios(
        &.{expectedRatio(2, 2.0, periods, .area_fraction)},
        frame,
        .{
            .particle_diameter = 2.0,
            .neighbor_radius_diameters = 1.0,
            .ratio_mode = .area_fraction,
        },
    );
}
