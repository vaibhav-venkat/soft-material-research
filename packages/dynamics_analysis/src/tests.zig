const std = @import("std");
const dynamics_analysis = @import("dynamics_analysis");

const input = dynamics_analysis.input;
const ffi = dynamics_analysis.ffi;

const valid_header =
    \\{"__metadata__":{"schema":"hexatic.big_lx.frames.v1"},"coords":{"dtype":"F32","shape":[2,3,3],"data_offsets":[0,72]},"step":{"dtype":"I64","shape":[2],"data_offsets":[72,88]}}
;

fn makeFixture(allocator: std.mem.Allocator, header: []const u8, data_len: usize) ![]u8 {
    const bytes = try allocator.alloc(u8, 8 + header.len + data_len);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8 .. 8 + header.len], header);
    @memset(bytes[8 + header.len ..], 0);
    return bytes;
}

fn writeFixture(
    temporary: *std.testing.TmpDir,
    name: []const u8,
    header: []const u8,
    data_len: usize,
) ![]u8 {
    const bytes = try makeFixture(std.testing.allocator, header, data_len);
    defer std.testing.allocator.free(bytes);
    const io = std.testing.io;
    const file = try temporary.dir.createFile(io, name, .{});
    defer file.close(io);
    try file.writePositionalAll(io, bytes, 0);
    return std.fmt.allocPrint(
        std.testing.allocator,
        ".zig-cache/tmp/{s}/{s}",
        .{ temporary.sub_path, name },
    );
}

fn writeDynamicsFixtures(temporary: *std.testing.TmpDir) !struct { []u8, []u8 } {
    const static_header =
        \\{"__metadata__":{"schema":"hexatic.big_lx.static.v1"},"lx":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}
    ;
    const static_bytes = try makeFixture(std.testing.allocator, static_header, 4);
    defer std.testing.allocator.free(static_bytes);
    const static_data_start = 8 + static_header.len;
    std.mem.writeInt(u32, static_bytes[static_data_start..][0..4], @bitCast(@as(f32, 10.0)), .little);
    const io = std.testing.io;
    const static_file = try temporary.dir.createFile(io, "static.safetensors", .{});
    defer static_file.close(io);
    try static_file.writePositionalAll(io, static_bytes, 0);
    const static_path = try std.fmt.allocPrint(
        std.testing.allocator,
        ".zig-cache/tmp/{s}/static.safetensors",
        .{temporary.sub_path},
    );
    errdefer std.testing.allocator.free(static_path);

    const frame_header =
        \\{"__metadata__":{"schema":"hexatic.big_lx.frames.v1"},"coords":{"dtype":"F32","shape":[3,2,3],"data_offsets":[0,72]},"step":{"dtype":"I64","shape":[3],"data_offsets":[72,96]}}
    ;
    const bytes = try makeFixture(std.testing.allocator, frame_header, 96);
    defer std.testing.allocator.free(bytes);
    const data_start = 8 + frame_header.len;
    const coordinates = [_]f32{
        4.8,  0, 0, -2.8, 0, 0,
        -4.2, 0, 0, -1.8, 0, 0,
        -3.2, 0, 0, -0.8, 0, 0,
    };
    for (coordinates, 0..) |value, index| {
        const start = data_start + index * @sizeOf(f32);
        std.mem.writeInt(u32, bytes[start..][0..4], @bitCast(value), .little);
    }
    const steps = [_]i64{ 10, 20, 30 };
    for (steps, 0..) |value, index| {
        const start = data_start + 72 + index * @sizeOf(i64);
        std.mem.writeInt(i64, bytes[start..][0..8], value, .little);
    }
    const file = try temporary.dir.createFile(io, "frames.safetensors", .{});
    defer file.close(io);
    try file.writePositionalAll(io, bytes, 0);
    const frame_path = try std.fmt.allocPrint(
        std.testing.allocator,
        ".zig-cache/tmp/{s}/frames.safetensors",
        .{temporary.sub_path},
    );
    return .{ static_path, frame_path };
}

fn expectApproxSlice(expected: []const f64, actual: []const f64) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |expected_value, actual_value| {
        try std.testing.expectApproxEqAbs(expected_value, actual_value, 1.0e-6);
    }
}

fn freeCom(com: dynamics_analysis.ComSeries) void {
    std.testing.allocator.free(com.elapsed_time);
    std.testing.allocator.free(com.center);
    std.testing.allocator.free(com.velocity);
}

fn freeCorrelation(correlation: dynamics_analysis.CorrelationSeries) void {
    std.testing.allocator.free(correlation.lag_indices);
    std.testing.allocator.free(correlation.lag_times);
    std.testing.allocator.free(correlation.pearson);
    std.testing.allocator.free(correlation.origin_counts);
}

test "public option and result contracts have stable defaults and fields" {
    const options = dynamics_analysis.Options{};
    try std.testing.expectEqual(@as(usize, 0), options.frame_start);
    try std.testing.expectEqual(@as(?usize, null), options.frame_stop);
    try std.testing.expectEqual(@as(f64, 1.0), options.timestep);
    try std.testing.expectEqual(@as(?usize, null), options.max_lag);
    try std.testing.expectEqual(@as(u32, 0), options.device_ordinal);

    const dataset = input.DatasetInput{
        .static = .{ .path = "static.safetensors" },
        .shard_paths = &.{ "first.safetensors", "second.safetensors" },
    };
    try std.testing.expectEqualStrings("static.safetensors", dataset.static.path);
    try std.testing.expectEqual(@as(usize, 2), dataset.shard_paths.len);

    const result = dynamics_analysis.Result{
        .com = .{ .elapsed_time = &.{}, .center = &.{}, .velocity = &.{} },
        .correlation = .{
            .lag_indices = &.{},
            .lag_times = &.{},
            .pearson = &.{},
            .origin_counts = &.{},
        },
    };
    try std.testing.expectEqual(@as(usize, 0), result.com.velocity.len);
    try std.testing.expectEqual(@as(usize, 0), result.correlation.pearson.len);
}

test "mapped shard exposes zero-copy tensors and the frame schema" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try writeFixture(&temporary, "valid.safetensors", valid_header, 88);
    defer std.testing.allocator.free(path);

    var shard = try input.inspectShard(std.testing.allocator, path);
    defer shard.deinit();
    try std.testing.expect(shard.contains("coords"));
    try std.testing.expect(shard.contains("step"));
    try std.testing.expect(!shard.contains("missing"));
    try std.testing.expectEqual(@as(usize, 2), shard.keys().len);

    const schema = try shard.frameSchema();
    try std.testing.expectEqual(@as(usize, 2), schema.frame_count);
    try std.testing.expectEqual(@as(usize, 3), schema.particle_count);
    try std.testing.expectEqual(@as(usize, 3), schema.component_count);
    try std.testing.expectEqual(input.safetensors.Dtype.f32, schema.coordinate_dtype);
    try std.testing.expectEqual(input.safetensors.Dtype.i64, schema.step_dtype);

    const coordinates = try shard.tensor("coords");
    const first_frame = try coordinates.frame(0);
    try std.testing.expectEqual(@as(usize, 9), first_frame.elementCount());
    try std.testing.expectEqual(@intFromPtr(coordinates.bytes.ptr), @intFromPtr(first_frame.bytes.ptr));
    try std.testing.expectError(error.TensorNotFound, shard.tensor("missing"));
}

test "frame schema rejects bad ranks, lengths, and dtypes" {
    const cases = [_]struct { name: []const u8, header: []const u8, data_len: usize, expected: anyerror }{
        .{
            .name = "coords-rank.safetensors",
            .header =
            \\{"coords":{"dtype":"F32","shape":[2,9],"data_offsets":[0,72]},"step":{"dtype":"I64","shape":[2],"data_offsets":[72,88]}}
            ,
            .data_len = 88,
            .expected = error.InvalidCoordinateShape,
        },
        .{
            .name = "step-shape.safetensors",
            .header =
            \\{"coords":{"dtype":"F32","shape":[2,3,3],"data_offsets":[0,72]},"step":{"dtype":"I64","shape":[1],"data_offsets":[72,80]}}
            ,
            .data_len = 80,
            .expected = error.InvalidStepShape,
        },
        .{
            .name = "coords-dtype.safetensors",
            .header =
            \\{"coords":{"dtype":"I32","shape":[2,3,3],"data_offsets":[0,72]},"step":{"dtype":"I64","shape":[2],"data_offsets":[72,88]}}
            ,
            .data_len = 88,
            .expected = error.InvalidCoordinateDtype,
        },
        .{
            .name = "step-dtype.safetensors",
            .header =
            \\{"coords":{"dtype":"F32","shape":[2,3,3],"data_offsets":[0,72]},"step":{"dtype":"F32","shape":[2],"data_offsets":[72,80]}}
            ,
            .data_len = 80,
            .expected = error.InvalidStepDtype,
        },
    };

    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    for (cases) |case| {
        const path = try writeFixture(&temporary, case.name, case.header, case.data_len);
        defer std.testing.allocator.free(path);
        var shard = try input.Shard.open(std.testing.allocator, path);
        defer shard.deinit();
        try std.testing.expectError(case.expected, shard.frameSchema());
    }
}

test "shard set owns multiple mappings and cleans up partial failure" {
    try std.testing.expectError(
        error.NoInput,
        input.ShardSet.open(std.testing.allocator, &.{}),
    );

    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const first = try writeFixture(&temporary, "first.safetensors", valid_header, 88);
    defer std.testing.allocator.free(first);
    const second = try writeFixture(&temporary, "second.safetensors", valid_header, 88);
    defer std.testing.allocator.free(second);

    var set = try input.ShardSet.open(std.testing.allocator, &.{ first, second });
    try std.testing.expectEqual(@as(usize, 2), set.shards.len);
    try std.testing.expect(set.shards[0].contains("coords"));
    try std.testing.expect(set.shards[1].contains("step"));
    set.deinit();

    const failed = input.ShardSet.open(std.testing.allocator, &.{ first, "missing.safetensors" });
    if (failed) |opened| {
        var unexpected = opened;
        unexpected.deinit();
        return error.ExpectedOpenFailure;
    } else |_| {}
}

test "finite difference differentiates a quadratic on a uniform grid" {
    const derivative = try dynamics_analysis.dynamics.finiteDifference(
        std.testing.allocator,
        &.{ 0.0, 1.0, 4.0 },
        &.{ 0.0, 1.0, 2.0 },
    );
    defer std.testing.allocator.free(derivative);
    try expectApproxSlice(&.{ 0.0, 2.0, 4.0 }, derivative);
}

test "center of mass streams shards and returns elapsed time center and velocity" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const paths = try writeDynamicsFixtures(&temporary);
    defer std.testing.allocator.free(paths[0]);
    defer std.testing.allocator.free(paths[1]);
    const dataset = input.DatasetInput{
        .static = .{ .path = paths[0] },
        .shard_paths = &.{paths[1]},
    };
    const com = try dynamics_analysis.dynamics.analyzeCenterOfMass(
        std.testing.allocator,
        dataset,
        .{ .timestep = 0.1 },
    );
    defer freeCom(com);
    try expectApproxSlice(&.{ 0.0, 1.0, 2.0 }, com.elapsed_time);
    try expectApproxSlice(&.{ 1.0, 2.0, 3.0 }, com.center);
    try expectApproxSlice(&.{ 1.0, 1.0, 1.0 }, com.velocity);
}

test "linalg backend computes lagged Pearson coefficients" {
    var context = try dynamics_analysis.backend.Context.init(std.testing.allocator, .{});
    defer context.deinit();
    const pearson = try dynamics_analysis.backend.laggedPearson(
        std.testing.allocator,
        &context,
        &.{ 1.0, 2.0, 3.0, 4.0 },
        2,
    );
    defer std.testing.allocator.free(pearson);
    try expectApproxSlice(&.{ 1.0, 1.0, 1.0 }, pearson);
}

test "correlation builds lag coordinates coefficients and origin counts" {
    var context = try dynamics_analysis.backend.Context.init(std.testing.allocator, .{});
    defer context.deinit();
    const com = dynamics_analysis.ComSeries{
        .elapsed_time = @constCast(&[_]f64{ 0.0, 1.0, 2.0, 3.0 }),
        .center = @constCast(&[_]f64{ 0.0, 0.0, 0.0, 0.0 }),
        .velocity = @constCast(&[_]f64{ 1.0, 2.0, 3.0, 4.0 }),
    };
    const correlation = try dynamics_analysis.dynamics.analyzeCorrelation(
        std.testing.allocator,
        &context,
        com,
        .{ .max_lag = 2 },
    );
    defer freeCorrelation(correlation);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, correlation.lag_indices);
    try expectApproxSlice(&.{ 0.0, 1.0, 2.0 }, correlation.lag_times);
    try expectApproxSlice(&.{ 1.0, 1.0, 1.0 }, correlation.pearson);
    try std.testing.expectEqualSlices(usize, &.{ 4, 3, 2 }, correlation.origin_counts);
}

test "top-level analysis returns COM and correlation series" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const paths = try writeDynamicsFixtures(&temporary);
    defer std.testing.allocator.free(paths[0]);
    defer std.testing.allocator.free(paths[1]);
    const dataset = input.DatasetInput{
        .static = .{ .path = paths[0] },
        .shard_paths = &.{paths[1]},
    };
    var context = try dynamics_analysis.backend.Context.init(std.testing.allocator, .{});
    defer context.deinit();
    const result = try dynamics_analysis.analyze(
        std.testing.allocator,
        &context,
        dataset,
        .{ .timestep = 0.1, .max_lag = 1 },
    );
    defer freeCom(result.com);
    defer freeCorrelation(result.correlation);
    try expectApproxSlice(&.{ 0.0, 1.0, 2.0 }, result.com.elapsed_time);
    try expectApproxSlice(&.{ 1.0, 2.0, 3.0 }, result.com.center);
    try expectApproxSlice(&.{ 1.0, 1.0, 1.0 }, result.com.velocity);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, result.correlation.lag_indices);
    try expectApproxSlice(&.{ 1.0, 1.0 }, result.correlation.pearson);
}

test "C ABI reports version, argument errors, valid mappings, and file errors" {
    try std.testing.expectEqual(@as(u32, 1), ffi.dynamics_analysis_api_version());
    const options = ffi.COptions{
        .frame_start = 0,
        .frame_stop = 0,
        .has_frame_stop = false,
        .timestep = 1.0,
        .max_lag = 0,
        .has_max_lag = false,
    };
    var output: ffi.CResult = undefined;
    const no_paths = [_][*:0]const u8{""};
    try std.testing.expectEqual(
        @as(c_int, 1),
        ffi.dynamics_analysis_run(&no_paths, 0, &options, &output),
    );

    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try writeFixture(&temporary, "ffi.safetensors", valid_header, 88);
    defer std.testing.allocator.free(path);
    const path_z = try std.testing.allocator.dupeSentinel(u8, path, 0);
    defer std.testing.allocator.free(path_z);
    const valid_paths = [_][*:0]const u8{path_z.ptr};
    try std.testing.expectEqual(
        @as(c_int, 0),
        ffi.dynamics_analysis_run(&valid_paths, valid_paths.len, &options, &output),
    );
    try std.testing.expectEqual(@as(usize, 0), output.elapsed_time.len);
    try std.testing.expectEqual(@as(usize, 0), output.pearson.len);
    ffi.dynamics_analysis_release(&output);

    const missing_paths = [_][*:0]const u8{"missing.safetensors"};
    try std.testing.expectEqual(
        @as(c_int, 3),
        ffi.dynamics_analysis_run(&missing_paths, missing_paths.len, &options, &output),
    );
}
