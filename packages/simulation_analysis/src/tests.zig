const std = @import("std");
const gsd = @import("gsd");
const hdf5 = @import("hdf5");
const simulation_analysis = @import("simulation_analysis");

fn path(allocator: std.mem.Allocator, temporary: std.testing.TmpDir, name: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/{s}", .{ temporary.sub_path, name });
}

fn createTrajectory(allocator: std.mem.Allocator, trajectory_path: []const u8) !void {
    var writer = try gsd.File.create(
        allocator,
        trajectory_path,
        "simulation-analysis-test",
        "hoomd",
        gsd.makeVersion(1, 4),
        true,
    );
    defer writer.deinit();
    try writer.writeChunk(u64, "configuration/step", 1, 1, &.{100});
    try writer.writeChunk(u8, "configuration/dimensions", 1, 1, &.{3});
    try writer.writeChunk(f32, "configuration/box", 6, 1, &.{ 20, 30, 30, 0, 0, 0 });
    try writer.writeChunk(u32, "particles/N", 1, 1, &.{3});
    try writer.writeChunk(f32, "particles/position", 3, 3, &.{
        1, 0, 2,
        2, 2, 0,
        3, 0, -2,
    });
    try writer.endFrame();
    try writer.writeChunk(u64, "configuration/step", 1, 1, &.{200});
    try writer.writeChunk(f32, "particles/position", 3, 3, &.{
        4, -2, 0,
        5, 0,  0,
        6, 2,  2,
    });
    try writer.endFrame();
    try writer.flush();
}

test "public defaults and field registry are fixed before implementation" {
    const options = simulation_analysis.Options{ .input_path = "input.gsd", .output_dir = "output" };
    try std.testing.expectEqual(@as(usize, 256 * 1024 * 1024), options.target_shard_bytes);
    try std.testing.expectEqual(simulation_analysis.WriteMode.create, options.write_mode);
    try std.testing.expectEqualStrings("coords", simulation_analysis.schema.base_fields[3].name);
    try std.testing.expectEqual(@as(u8, 3), simulation_analysis.schema.base_fields[3].rank);
}

test "option validation catches empty paths and invalid resource controls" {
    try std.testing.expectError(
        error.EmptyInputPath,
        simulation_analysis.schema.validateOptions(.{ .input_path = "", .output_dir = "out" }),
    );
    try std.testing.expectError(
        error.InvalidWorkerCount,
        simulation_analysis.schema.validateOptions(.{
            .input_path = "in.gsd",
            .output_dir = "out",
            .worker_count = 0,
        }),
    );
}

test "shard planning covers every frame contiguously" {
    const ranges = try simulation_analysis.planShards(std.testing.allocator, 10, 100, 4_900);
    defer std.testing.allocator.free(ranges);
    try std.testing.expectEqualDeep(&[_]simulation_analysis.ShardRange{
        .{ .start = 0, .stop = 4 },
        .{ .start = 4, .stop = 8 },
        .{ .start = 8, .stop = 10 },
    }, ranges);
}

test "SIMD cylindrical transform handles quadrants, origin, and a tail" {
    var positions: simulation_analysis.CartesianPositions = .empty;
    defer positions.deinit(std.testing.allocator);
    try positions.resize(std.testing.allocator, 5);
    var position_slices = positions.slice();
    position_slices.set(0, .{ .x = 1, .y = 0, .z = 2 });
    position_slices.set(1, .{ .x = 2, .y = 2, .z = 0 });
    position_slices.set(2, .{ .x = 3, .y = 0, .z = -2 });
    position_slices.set(3, .{ .x = 4, .y = -2, .z = 0 });
    position_slices.set(4, .{ .x = 5, .y = 0, .z = 0 });
    var coordinates: simulation_analysis.CylindricalCoordinates = .empty;
    defer coordinates.deinit(std.testing.allocator);
    try coordinates.resize(std.testing.allocator, positions.len);
    try simulation_analysis.transformCylindrical(positions.slice(), coordinates.slice());
    const result = coordinates.slice();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5 }, result.items(.x));
    const expected_theta = [_]f32{ 0, std.math.pi / 2.0, std.math.pi, 3.0 * std.math.pi / 2.0, 0 };
    const expected_r = [_]f32{ 2, 2, 2, 2, 0 };
    for (expected_theta, result.items(.theta)) |wanted, actual| {
        try std.testing.expectApproxEqAbs(wanted, actual, 2.0e-5);
    }
    for (expected_r, result.items(.r)) |wanted, actual| {
        try std.testing.expectApproxEqAbs(wanted, actual, 1.0e-5);
    }
}

test "COM property headers accept SoA buffers and remain explicitly unimplemented" {
    var coordinates: simulation_analysis.CylindricalCoordinates = .empty;
    defer coordinates.deinit(std.testing.allocator);
    try coordinates.resize(std.testing.allocator, 2);
    var result: simulation_analysis.properties.CylindricalFrameValues = .empty;
    defer result.deinit(std.testing.allocator);
    try result.resize(std.testing.allocator, 1);
    try std.testing.expectError(
        error.NotImplemented,
        simulation_analysis.properties.com_unwrapped(coordinates.slice(), 2, result.slice()),
    );
    try std.testing.expectError(
        error.NotImplemented,
        simulation_analysis.properties.com_velocity_unwrapped(coordinates.slice(), &.{100}, 2, result.slice()),
    );
}

test "new conversion writes static metadata and frame shards" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const trajectory = try path(std.testing.allocator, temporary, "trajectory.gsd");
    defer std.testing.allocator.free(trajectory);
    const output = try path(std.testing.allocator, temporary, "analysis");
    defer std.testing.allocator.free(output);
    try createTrajectory(std.testing.allocator, trajectory);

    const result = try simulation_analysis.run(std.testing.allocator, .{
        .input_path = trajectory,
        .output_dir = output,
        .worker_count = 2,
        .target_shard_bytes = 64,
    });
    try std.testing.expectEqual(@as(u64, 2), result.frame_count);
    try std.testing.expectEqual(@as(usize, 2), result.shard_count);

    const static_path = try std.fs.path.join(std.testing.allocator, &.{ output, "static.h5" });
    defer std.testing.allocator.free(static_path);
    var static_file = try hdf5.File.openPath(std.testing.allocator, static_path, .read_only);
    defer static_file.deinit();
    try std.testing.expectEqual(@as(u64, 2), try static_file.readAttribute(u64, "frame_count"));
    try std.testing.expectEqual(@as(u64, 3), try static_file.readAttribute(u64, "particle_count"));

    const first_shard_path = try std.fs.path.join(std.testing.allocator, &.{ output, "frames_000000.h5" });
    defer std.testing.allocator.free(first_shard_path);
    var first_shard = try hdf5.File.openPath(std.testing.allocator, first_shard_path, .read_only);
    defer first_shard.deinit();
    try std.testing.expect(!(try first_shard.objectExists("com_unwrapped")));
    try std.testing.expect(!(try first_shard.objectExists("com_velocity_unwrapped")));
    var first_coords = try first_shard.openDataset("coords");
    defer first_coords.deinit();
    const first_shape = try first_coords.shapeAlloc(std.testing.allocator);
    defer std.testing.allocator.free(first_shape);
    try std.testing.expectEqualSlices(u64, &.{ 3, 1, 3 }, first_shape);
    var first_values: [9]f32 = undefined;
    try first_coords.readAll(f32, &first_values);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, first_values[0..3]);
    try std.testing.expectApproxEqAbs(@as(f32, 0), first_values[3], 1.0e-5);
    try std.testing.expectApproxEqAbs(@as(f32, std.math.pi / 2.0), first_values[4], 2.0e-5);
    try std.testing.expectApproxEqAbs(@as(f32, std.math.pi), first_values[5], 2.0e-5);
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2 }, first_values[6..9]);

    const second_shard_path = try std.fs.path.join(std.testing.allocator, &.{ output, "frames_000001.h5" });
    defer std.testing.allocator.free(second_shard_path);
    var second_shard = try hdf5.File.openPath(std.testing.allocator, second_shard_path, .read_only);
    defer second_shard.deinit();
    var second_coords = try second_shard.openDataset("coords");
    defer second_coords.deinit();
    var second_values: [9]f32 = undefined;
    try second_coords.readAll(f32, &second_values);
    try std.testing.expectEqualSlices(f32, &.{ 4, 5, 6 }, second_values[0..3]);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0 * std.math.pi / 2.0), second_values[3], 2.0e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), second_values[4], 1.0e-5);
    try std.testing.expectApproxEqAbs(@as(f32, std.math.pi / 4.0), second_values[5], 2.0e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2), second_values[6], 1.0e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), second_values[7], 1.0e-5);
    try std.testing.expectApproxEqAbs(@as(f32, @sqrt(8.0)), second_values[8], 1.0e-5);
    var inherited_box = try second_shard.openDataset("box");
    defer inherited_box.deinit();
    var box: [6]f32 = undefined;
    try inherited_box.readAll(f32, &box);
    try std.testing.expectEqualSlices(f32, &.{ 20, 30, 30, 0, 0, 0 }, &box);
}

test "update adds a missing field without replacing established data" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const trajectory = try path(std.testing.allocator, temporary, "trajectory.gsd");
    defer std.testing.allocator.free(trajectory);
    const output = try path(std.testing.allocator, temporary, "analysis");
    defer std.testing.allocator.free(output);
    try createTrajectory(std.testing.allocator, trajectory);

    const created = try simulation_analysis.run(std.testing.allocator, .{
        .input_path = trajectory,
        .output_dir = output,
    });
    try std.testing.expectEqual(@as(usize, 4), created.fields_written);
    const shard_path = try std.fs.path.join(std.testing.allocator, &.{ output, "frames_000000.h5" });
    defer std.testing.allocator.free(shard_path);
    {
        var shard = try hdf5.File.openPath(std.testing.allocator, shard_path, .read_write);
        defer shard.deinit();
        try shard.writeAttribute(u64, "preserved", 42);
    }
    const updated = try simulation_analysis.run(std.testing.allocator, .{
        .input_path = trajectory,
        .output_dir = output,
        .write_mode = .update,
    });
    try std.testing.expectEqual(@as(usize, 0), updated.fields_written);
    var shard = try hdf5.File.openPath(std.testing.allocator, shard_path, .read_only);
    defer shard.deinit();
    try std.testing.expectEqual(@as(u64, 42), try shard.readAttribute(u64, "preserved"));
}

test "existing output is refused without update or overwrite" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const trajectory = try path(std.testing.allocator, temporary, "trajectory.gsd");
    defer std.testing.allocator.free(trajectory);
    const output = try path(std.testing.allocator, temporary, "analysis");
    defer std.testing.allocator.free(output);
    try createTrajectory(std.testing.allocator, trajectory);
    _ = try simulation_analysis.run(std.testing.allocator, .{
        .input_path = trajectory,
        .output_dir = output,
    });
    try std.testing.expectError(error.OutputExists, simulation_analysis.run(std.testing.allocator, .{
        .input_path = trajectory,
        .output_dir = output,
    }));
}

test "CLI parses the agreed package-only conversion interface" {
    const options = try simulation_analysis.cli.parseArgs(std.testing.allocator, &.{
        "simulation-analysis",
        "--input",
        "trajectory.gsd",
        "--output-dir",
        "analysis",
        "--workers",
        "4",
        "--target-shard-mib",
        "128",
        "--update",
    });
    try std.testing.expectEqualStrings("trajectory.gsd", options.input_path);
    try std.testing.expectEqualStrings("analysis", options.output_dir);
    try std.testing.expectEqual(@as(?usize, 4), options.worker_count);
    try std.testing.expectEqual(@as(usize, 128 * 1024 * 1024), options.target_shard_bytes);
    try std.testing.expectEqual(simulation_analysis.WriteMode.update, options.write_mode);

    try std.testing.expectError(error.UnknownArgument, simulation_analysis.cli.parseArgs(std.testing.allocator, &.{
        "simulation-analysis",
        "--input",
        "trajectory.gsd",
        "--output-dir",
        "analysis",
        "--metadata",
        "metadata.json",
    }));
}
