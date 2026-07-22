//! Dataset-level structural cluster orchestration.

const std = @import("std");
const safetensors = @import("safetensors");
const clusters = @import("clusters.zig");
const input = @import("input/root.zig");
const Options = @import("options.zig").Options;
const Result = @import("result.zig").Result;
const schema = @import("schema.zig");

pub fn analyze(
    allocator: std.mem.Allocator,
    dataset_input: input.DatasetInput,
    options: Options,
) !Result {
    try schema.validateOptions(options);
    var dataset = try input.Dataset.open(allocator, dataset_input);
    defer dataset.deinit();
    const geometry = try schema.readGeometry(&dataset.static);

    var reference: ?schema.FrameSchema = null;
    var total_frames: usize = 0;
    for (dataset.shards) |*shard| {
        const candidate = try schema.inspectFrameSchema(&shard.reader);
        if (reference) |value| try schema.requireCompatibleFrames(value, candidate) else reference = candidate;
        total_frames = try std.math.add(usize, total_frames, candidate.frame_count);
    }
    const frame_stop = options.frame_stop orelse total_frames;
    if (options.frame_start >= frame_stop or frame_stop > total_frames) {
        return error.InvalidFrameRange;
    }

    const particle_count = reference.?.particle_count;
    const points = try allocator.alloc([2]f64, particle_count);
    defer allocator.free(points);
    const psi6 = try allocator.alloc([2]f64, particle_count);
    defer allocator.free(psi6);
    const eligible = try allocator.alloc(bool, particle_count);
    defer allocator.free(eligible);

    var ratios: std.ArrayList(f64) = .empty;
    errdefer ratios.deinit(allocator);
    var shard_frame_start: usize = 0;
    for (dataset.shards) |*shard| {
        const coordinates = try shard.reader.tensor("coords");
        const psi_real = try shard.reader.tensor("psi_real");
        const psi_imaginary = try shard.reader.tensor("psi_imag");
        const mask = try shard.reader.tensor("hexatic_shell_mask");
        const shard_frame_stop = try std.math.add(usize, shard_frame_start, coordinates.shape[0]);
        const selected_start = @max(options.frame_start, shard_frame_start);
        const selected_stop = @min(frame_stop, shard_frame_stop);
        if (selected_start < selected_stop) {
            const local_start = selected_start - shard_frame_start;
            const local_stop = selected_stop - shard_frame_start;
            for (local_start..local_stop) |local_frame| {
                try decodeFrame(
                    points,
                    psi6,
                    eligible,
                    coordinates,
                    psi_real,
                    psi_imaginary,
                    mask,
                    local_frame,
                    geometry.circumference,
                );
                const frame_result = try clusters.analyzeStructuralFrame(
                    allocator,
                    .{
                        .points = points,
                        .psi6 = psi6,
                        .eligible = eligible,
                        .periods = .{ geometry.axial_period, geometry.circumference },
                    },
                    options,
                );
                defer frame_result.deinit(allocator);
                try ratios.appendSlice(allocator, frame_result.ratios);
            }
        }
        shard_frame_start = shard_frame_stop;
    }

    return .{ .ratios = try ratios.toOwnedSlice(allocator) };
}

fn decodeFrame(
    points: [][2]f64,
    psi6: [][2]f64,
    eligible: []bool,
    coordinates: safetensors.TensorView,
    psi_real: safetensors.TensorView,
    psi_imaginary: safetensors.TensorView,
    mask: safetensors.TensorView,
    frame_index: usize,
    circumference: f64,
) !void {
    const coordinate_frame = try coordinates.frame(frame_index);
    switch (coordinate_frame.dtype) {
        .f32 => fillPoints(f32, points, try coordinate_frame.values(f32), circumference),
        .f64 => fillPoints(f64, points, try coordinate_frame.values(f64), circumference),
        else => return error.InvalidFloatDtype,
    }

    const real_frame = try psi_real.frame(frame_index);
    switch (real_frame.dtype) {
        .f32 => fillPsi6Component(f32, psi6, try real_frame.values(f32), 0),
        .f64 => fillPsi6Component(f64, psi6, try real_frame.values(f64), 0),
        else => return error.InvalidFloatDtype,
    }
    const imaginary_frame = try psi_imaginary.frame(frame_index);
    switch (imaginary_frame.dtype) {
        .f32 => fillPsi6Component(f32, psi6, try imaginary_frame.values(f32), 1),
        .f64 => fillPsi6Component(f64, psi6, try imaginary_frame.values(f64), 1),
        else => return error.InvalidFloatDtype,
    }

    const mask_frame = try mask.frame(frame_index);
    if (mask_frame.dtype != .bool or mask_frame.bytes.len != eligible.len) {
        return error.InvalidMaskDtype;
    }
    for (mask_frame.bytes, eligible) |value, *destination| {
        destination.* = value != 0;
    }
}

fn fillPoints(
    comptime T: type,
    points: [][2]f64,
    coordinates: []align(1) const T,
    circumference: f64,
) void {
    const azimuth_scale = circumference / std.math.tau;
    for (points, 0..) |*point, particle| {
        const start = particle * 3;
        point.* = .{
            @as(f64, @floatCast(coordinates[start])),
            @as(f64, @floatCast(coordinates[start + 1])) * azimuth_scale,
        };
    }
}

fn fillPsi6Component(
    comptime T: type,
    psi6: [][2]f64,
    source: []align(1) const T,
    comptime component: usize,
) void {
    for (psi6, source) |*destination, value| {
        destination[component] = @as(f64, @floatCast(value));
    }
}
