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
    try writer.writeChunk(f32, "particles/orientation", 4, 3, &.{
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
    });
    try writer.endFrame();
    try writer.writeChunk(u64, "configuration/step", 1, 1, &.{200});
    try writer.writeChunk(f32, "particles/position", 3, 3, &.{
        4, -2, 0,
        5, 0,  0,
        6, 2,  2,
    });
    try writer.writeChunk(f32, "particles/orientation", 4, 3, &.{
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
    });
    try writer.endFrame();
    try writer.flush();
}

test "public defaults and field registry are fixed before implementation" {
    const options = simulation_analysis.Options{ .input_path = "input.gsd", .output_dir = "output", .cylinder_radius = 10 };
    try std.testing.expectEqual(@as(usize, 256 * 1024 * 1024), options.target_shard_bytes);
    try std.testing.expectEqual(@as(f64, 1.0), options.timestep);
    try std.testing.expectEqual(simulation_analysis.WriteMode.create, options.write_mode);
    try std.testing.expectEqualStrings("coords", simulation_analysis.schema.base_fields[3].name);
    try std.testing.expectEqual(@as(u8, 3), simulation_analysis.schema.base_fields[3].rank);
}

test "option validation catches empty paths and invalid resource controls" {
    try std.testing.expectError(
        error.EmptyInputPath,
        simulation_analysis.schema.validateOptions(.{ .input_path = "", .output_dir = "out", .cylinder_radius = 10 }),
    );
    try std.testing.expectError(
        error.InvalidWorkerCount,
        simulation_analysis.schema.validateOptions(.{
            .input_path = "in.gsd",
            .output_dir = "out",
            .cylinder_radius = 10,
            .worker_count = 0,
        }),
    );
    try std.testing.expectError(
        error.InvalidTimestep,
        simulation_analysis.schema.validateOptions(.{
            .input_path = "in.gsd",
            .output_dir = "out",
            .cylinder_radius = 10,
            .timestep = 0,
        }),
    );
}

test "shard planning covers every frame contiguously" {
    const ranges = try simulation_analysis.planShards(std.testing.allocator, 10, 100, 16_900);
    defer std.testing.allocator.free(ranges);
    try std.testing.expectEqualDeep(&[_]simulation_analysis.ShardRange{
        .{ .start = 0, .stop = 4 },
        .{ .start = 4, .stop = 8 },
        .{ .start = 8, .stop = 10 },
    }, ranges);
}

test "SIMD cylindrical transform handles quadrants, origin, and a tail" {
    var positions: simulation_analysis.data_structures.coordinates.CartesianPositions = .empty;
    defer positions.deinit(std.testing.allocator);
    try positions.resize(std.testing.allocator, 5);
    var position_slices = positions.slice();
    position_slices.set(0, .{ .x = 1, .y = 0, .z = 2 });
    position_slices.set(1, .{ .x = 2, .y = 2, .z = 0 });
    position_slices.set(2, .{ .x = 3, .y = 0, .z = -2 });
    position_slices.set(3, .{ .x = 4, .y = -2, .z = 0 });
    position_slices.set(4, .{ .x = 5, .y = 0, .z = 0 });
    var coordinates: simulation_analysis.data_structures.coordinates.CylindricalCoordinates = .empty;
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

test "generic cell list finds periodic radius neighbors without duplicates" {
    const CellList2 = simulation_analysis.data_structures.CellList(2);
    const x = [_]f32{ -4.9, 4.9, 0.0, 0.4 };
    const s = [_]f32{ 0.0, 0.0, 3.0, 3.0 };
    var cells = try CellList2.init(
        std.testing.allocator,
        .{ &x, &s },
        .{ .{ .lower = -5, .upper = 5 }, .{ .lower = 0, .upper = 6 } },
        .{ true, true },
        1.0,
    );
    defer cells.deinit();
    var neighbors: std.ArrayList(CellList2.Neighbor) = .empty;
    defer neighbors.deinit(std.testing.allocator);
    try cells.queryRadius(0, 0.3, true, null, &neighbors);
    try std.testing.expectEqual(@as(usize, 1), neighbors.items.len);
    try std.testing.expectEqual(@as(usize, 1), neighbors.items[0].index);
    try std.testing.expectApproxEqAbs(@as(f32, -0.2), neighbors.items[0].displacement[0], 1.0e-5);
}

test "generic cell list rebuilds and returns deterministic nearest neighbors" {
    const CellList3 = simulation_analysis.data_structures.CellList(3);
    const x = [_]f32{ 0, 1, 2, 3, 4, 5, 6 };
    const y = [_]f32{ 0, 0, 0, 0, 0, 0, 0 };
    const z = [_]f32{ 0, 0, 0, 0, 0, 0, 0 };
    var cells = try CellList3.init(
        std.testing.allocator,
        .{ &x, &y, &z },
        .{
            .{ .lower = -0.5, .upper = 6.5 },
            .{ .lower = -1, .upper = 1 },
            .{ .lower = -1, .upper = 1 },
        },
        .{ false, false, false },
        0.75,
    );
    defer cells.deinit();
    var neighbors: std.ArrayList(CellList3.Neighbor) = .empty;
    defer neighbors.deinit(std.testing.allocator);
    try cells.nearest(0, 6, null, &neighbors);
    try std.testing.expectEqual(@as(usize, 6), neighbors.items.len);
    for (neighbors.items, 1..) |neighbor, expected| try std.testing.expectEqual(expected, neighbor.index);

    const reversed = [_]f32{ 6, 5, 4, 3, 2, 1, 0 };
    try cells.rebuild(.{ &reversed, &y, &z });
    try cells.queryRadius(6, 1.1, true, null, &neighbors);
    try std.testing.expectEqual(@as(usize, 1), neighbors.items.len);
    try std.testing.expectEqual(@as(usize, 5), neighbors.items[0].index);
}

test "math helpers match active-direction and cylindrical conventions" {
    const coordinate_types = simulation_analysis.data_structures.coordinates;
    const direction = try simulation_analysis.properties.math.activeDirection(.{
        .w = @sqrt(@as(f32, 0.5)),
        .x = 0,
        .y = 0,
        .z = @sqrt(@as(f32, 0.5)),
    });
    try std.testing.expectApproxEqAbs(@as(f32, 0), direction.x, 1.0e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), direction.y, 1.0e-6);
    const cylindrical = simulation_analysis.properties.math.cylindricalVector(
        coordinate_types.CartesianVector{ .x = 0, .y = 1, .z = 0 },
        0,
    );
    try std.testing.expectEqual(@as(f32, 0), cylindrical.r);
    try std.testing.expectEqual(@as(f32, 1), cylindrical.theta);
}

test "hexatic kernel gives unit order and zero charge on a triangular shell" {
    const coordinate_types = simulation_analysis.data_structures.coordinates;
    const CellList3 = simulation_analysis.data_structures.CellList(3);
    const radius: f32 = 20;
    var positions: coordinate_types.CartesianPositions = .empty;
    defer positions.deinit(std.testing.allocator);
    try positions.resize(std.testing.allocator, 7);
    var position_slices = positions.slice();
    position_slices.set(0, .{ .x = 0, .y = 0, .z = radius });
    for (0..6) |neighbor| {
        const angle = @as(f32, @floatFromInt(neighbor)) * std.math.pi / 3.0;
        const axial = @cos(angle);
        const surface = @sin(angle);
        position_slices.set(neighbor + 1, .{
            .x = axial,
            .y = radius * @sin(surface / radius),
            .z = radius * @cos(surface / radius),
        });
    }
    var cells = try CellList3.init(
        std.testing.allocator,
        .{ position_slices.items(.x), position_slices.items(.y), position_slices.items(.z) },
        .{
            .{ .lower = -5, .upper = 5 },
            .{ .lower = -radius - 1, .upper = radius + 1 },
            .{ .lower = -radius - 1, .upper = radius + 1 },
        },
        .{ true, false, false },
        1.2,
    );
    defer cells.deinit();
    var order: [7]f32 = undefined;
    var real: [7]f32 = undefined;
    var imaginary: [7]f32 = undefined;
    var charge: [7]i8 = undefined;
    var mask: [7]u8 = undefined;
    try simulation_analysis.properties.hexatic.calculate(
        std.testing.allocator,
        &cells,
        positions.slice(),
        .{ .cylinder_radius = radius, .shell_delta = 1, .coordination_radius = 1.2 },
        .{ .order = &order, .real = &real, .imaginary = &imaginary, .disclination = &charge, .shell_mask = &mask },
    );
    try std.testing.expectApproxEqAbs(@as(f32, 1), order[0], 2.0e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1), real[0], 2.0e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), imaginary[0], 2.0e-5);
    try std.testing.expectEqual(@as(i8, 0), charge[0]);
    try std.testing.expectEqual(@as(u8, 1), mask[0]);
}

test "polar kernel includes self and writes x radial azimuthal order" {
    const coordinate_types = simulation_analysis.data_structures.coordinates;
    const CellList3 = simulation_analysis.data_structures.CellList(3);
    const x = [_]f32{ 0, 0.25 };
    const y = [_]f32{ 0, 0 };
    const z = [_]f32{ 2, 2 };
    var cells = try CellList3.init(
        std.testing.allocator,
        .{ &x, &y, &z },
        .{ .{ .lower = -2, .upper = 2 }, .{ .lower = -1, .upper = 1 }, .{ .lower = 1, .upper = 3 } },
        .{ true, false, false },
        1,
    );
    defer cells.deinit();
    var orientations: coordinate_types.Quaternions = .empty;
    defer orientations.deinit(std.testing.allocator);
    try orientations.resize(std.testing.allocator, 2);
    var orientation_slices = orientations.slice();
    orientation_slices.set(0, .{ .w = 1, .x = 0, .y = 0, .z = 0 });
    orientation_slices.set(1, .{ .w = 1, .x = 0, .y = 0, .z = 0 });
    var polar: coordinate_types.CylindricalVectors = .empty;
    defer polar.deinit(std.testing.allocator);
    try polar.resize(std.testing.allocator, 2);
    var rho: [2]f32 = undefined;
    try simulation_analysis.properties.polar.calculate(
        std.testing.allocator,
        &cells,
        orientations.slice(),
        &.{ 0, 0 },
        .{ .pocket_radius = 1 },
        &rho,
        polar.slice(),
    );
    const expected = try simulation_analysis.properties.math.gaussianWeight(0, 1) +
        try simulation_analysis.properties.math.gaussianWeight(0.25 * 0.25, 1);
    try std.testing.expectApproxEqAbs(expected, rho[0], 1.0e-6);
    try std.testing.expectApproxEqAbs(expected, polar.slice().items(.x)[0], 1.0e-6);
    try std.testing.expectEqual(@as(f32, 0), polar.slice().items(.r)[0]);
    try std.testing.expectEqual(@as(f32, 0), polar.slice().items(.theta)[0]);
}

test "cluster kernel returns raw component particle counts across a seam" {
    var sizes = try simulation_analysis.properties.clusters.calculate(
        std.testing.allocator,
        &.{ -4.9, 4.9, 0 },
        &.{ 0, 0, 2 },
        &.{ 1, 1, 1 },
        &.{ 0, 0, 0 },
        &.{ 1, 1, 1 },
        .{
            .axial_period = 10,
            .cylinder_radius = 2,
            .neighbor_radius = 0.5,
        },
    );
    defer sizes.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u32, &.{2}, sizes.items);
}

test "COM unwraps axial and angular crossings and velocity uses physical timestep" {
    var coordinates: simulation_analysis.data_structures.coordinates.CylindricalCoordinates = .empty;
    defer coordinates.deinit(std.testing.allocator);
    try coordinates.resize(std.testing.allocator, 6);
    var coordinate_slices = coordinates.slice();
    std.mem.copyForwards(f32, coordinate_slices.items(.x), &.{ 4.5, -4.0, -4.5, -3.0, -3.5, -2.0 });
    std.mem.copyForwards(f32, coordinate_slices.items(.theta), &.{ 6.0, 0.1, 0.2, 0.3, 0.4, 0.5 });
    std.mem.copyForwards(f32, coordinate_slices.items(.r), &.{ 1, 1, 2, 2, 3, 3 });
    var centers: simulation_analysis.properties.CylindricalFrameValues = .empty;
    defer centers.deinit(std.testing.allocator);
    try centers.resize(std.testing.allocator, 3);
    var workspace = try simulation_analysis.properties.com.Workspace.init(std.testing.allocator, 2);
    defer workspace.deinit();
    try simulation_analysis.properties.com_unwrapped(&workspace, coordinates.slice(), 2, 10, centers.slice());
    const center_slices = centers.slice();
    try std.testing.expectEqualSlices(f32, &.{ 0.25, 1.25, 2.25 }, center_slices.items(.x));
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, center_slices.items(.r));
    try std.testing.expectApproxEqAbs(@as(f32, 3.05), center_slices.items(.theta)[0], 1.0e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.3915927), center_slices.items(.theta)[1], 1.0e-5);
    var velocities: simulation_analysis.properties.CylindricalFrameValues = .empty;
    defer velocities.deinit(std.testing.allocator);
    try velocities.resize(std.testing.allocator, 3);
    try simulation_analysis.properties.com_velocity_unwrapped(centers.slice(), &.{ 0, 2, 4 }, 0.5, velocities.slice());
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1 }, velocities.slice().items(.x));
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1 }, velocities.slice().items(.r));
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
        .cylinder_radius = 3.5,
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
    try std.testing.expect(try first_shard.objectExists("com_unwrapped"));
    try std.testing.expect(try first_shard.objectExists("com_velocity_unwrapped"));
    var first_com = try first_shard.openDataset("com_unwrapped");
    defer first_com.deinit();
    const first_com_shape = try first_com.shapeAlloc(std.testing.allocator);
    defer std.testing.allocator.free(first_com_shape);
    try std.testing.expectEqualSlices(u64, &.{ 3, 1 }, first_com_shape);
    var first_velocity = try first_shard.openDataset("com_velocity_unwrapped");
    defer first_velocity.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), try first_velocity.readAttribute(f64, "timestep"));
    var first_velocity_values: [3]f32 = undefined;
    try first_velocity.readAll(f32, &first_velocity_values);
    try std.testing.expectApproxEqAbs(@as(f32, 0.03), first_velocity_values[0], 1.0e-6);
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
    var second_velocity = try second_shard.openDataset("com_velocity_unwrapped");
    defer second_velocity.deinit();
    var second_velocity_values: [3]f32 = undefined;
    try second_velocity.readAll(f32, &second_velocity_values);
    try std.testing.expectApproxEqAbs(first_velocity_values[0], second_velocity_values[0], 1.0e-6);
    try std.testing.expectApproxEqAbs(first_velocity_values[1], second_velocity_values[1], 1.0e-6);
    try std.testing.expectApproxEqAbs(first_velocity_values[2], second_velocity_values[2], 1.0e-6);
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

    var order_dataset = try first_shard.openDataset("hexatic_order");
    defer order_dataset.deinit();
    const order_shape = try order_dataset.shapeAlloc(std.testing.allocator);
    defer std.testing.allocator.free(order_shape);
    try std.testing.expectEqualSlices(u64, &.{ 1, 3 }, order_shape);
    var mask_dataset = try first_shard.openDataset("hexatic_shell_mask");
    defer mask_dataset.deinit();
    var mask_values: [3]u8 = undefined;
    try mask_dataset.readAll(u8, &mask_values);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0 }, &mask_values);
    var rho_dataset = try first_shard.openDataset("rho");
    defer rho_dataset.deinit();
    var rho_values: [3]f32 = undefined;
    try rho_dataset.readAll(f32, &rho_values);
    for (rho_values) |value| try std.testing.expect(value > 0);
    var polar_dataset = try first_shard.openDataset("polar_cylindrical");
    defer polar_dataset.deinit();
    const polar_shape = try polar_dataset.shapeAlloc(std.testing.allocator);
    defer std.testing.allocator.free(polar_shape);
    try std.testing.expectEqualSlices(u64, &.{ 3, 1, 3 }, polar_shape);
    var cluster_dataset = try first_shard.openDataset("cluster_sizes");
    defer cluster_dataset.deinit();
    const cluster_shape = try cluster_dataset.shapeAlloc(std.testing.allocator);
    defer std.testing.allocator.free(cluster_shape);
    try std.testing.expectEqualSlices(u64, &.{0}, cluster_shape);
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
        .cylinder_radius = 3.5,
    });
    try std.testing.expectEqual(@as(usize, 14), created.fields_written);
    const shard_path = try std.fs.path.join(std.testing.allocator, &.{ output, "frames_000000.h5" });
    defer std.testing.allocator.free(shard_path);
    {
        var shard = try hdf5.File.openPath(std.testing.allocator, shard_path, .read_write);
        defer shard.deinit();
        try shard.writeAttribute(u64, "preserved", 42);
        try shard.deleteLink("rho");
    }
    const updated = try simulation_analysis.run(std.testing.allocator, .{
        .input_path = trajectory,
        .output_dir = output,
        .cylinder_radius = 3.5,
        .write_mode = .update,
    });
    try std.testing.expectEqual(@as(usize, 1), updated.fields_written);
    var shard = try hdf5.File.openPath(std.testing.allocator, shard_path, .read_only);
    defer shard.deinit();
    try std.testing.expectEqual(@as(u64, 42), try shard.readAttribute(u64, "preserved"));
    try std.testing.expect(try shard.objectExists("rho"));
    try std.testing.expectError(error.AnalysisConfigurationMismatch, simulation_analysis.run(std.testing.allocator, .{
        .input_path = trajectory,
        .output_dir = output,
        .cylinder_radius = 4.0,
        .write_mode = .update,
    }));
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
        .cylinder_radius = 3.5,
    });
    try std.testing.expectError(error.OutputExists, simulation_analysis.run(std.testing.allocator, .{
        .input_path = trajectory,
        .output_dir = output,
        .cylinder_radius = 3.5,
    }));
}

test "CLI parses the agreed package-only conversion interface" {
    const options = try simulation_analysis.cli.parseArgs(std.testing.allocator, &.{
        "simulation-analysis",
        "--input",
        "trajectory.gsd",
        "--output-dir",
        "analysis",
        "--cylinder-radius",
        "10",
        "--workers",
        "4",
        "--target-shard-mib",
        "128",
        "--timestep",
        "0.000001",
        "--update",
    });
    try std.testing.expectEqualStrings("trajectory.gsd", options.input_path);
    try std.testing.expectEqualStrings("analysis", options.output_dir);
    try std.testing.expectEqual(@as(?usize, 4), options.worker_count);
    try std.testing.expectEqual(@as(usize, 128 * 1024 * 1024), options.target_shard_bytes);
    try std.testing.expectEqual(@as(f64, 0.000001), options.timestep);
    try std.testing.expectEqual(simulation_analysis.WriteMode.update, options.write_mode);
    try std.testing.expectEqual(@as(f64, 10), options.cylinder_radius);

    try std.testing.expectError(error.MissingCylinderRadius, simulation_analysis.cli.parseArgs(std.testing.allocator, &.{
        "simulation-analysis",
        "--input",
        "trajectory.gsd",
        "--output-dir",
        "analysis",
    }));

    try std.testing.expectError(error.UnknownArgument, simulation_analysis.cli.parseArgs(std.testing.allocator, &.{
        "simulation-analysis",
        "--input",
        "trajectory.gsd",
        "--output-dir",
        "analysis",
        "--cylinder-radius",
        "10",
        "--metadata",
        "metadata.json",
    }));
}
