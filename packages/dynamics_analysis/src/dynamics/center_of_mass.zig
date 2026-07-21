//! Streaming center-of-mass unwrapping and finite differences.

const std = @import("std");
const input = @import("../input/root.zig");
const schema = @import("../schema.zig");
const Options = @import("../options.zig").Options;
const ComSeries = @import("../result.zig").ComSeries;

pub const ComWorkspace = struct {
    previous_wrapped: []f64,
    unwrapped: []f64,
};

pub fn analyzeCenterOfMass(
    allocator: std.mem.Allocator,
    dataset: input.DatasetInput,
    options: Options,
) !ComSeries {
    if (dataset.shard_paths.len == 0) return error.NoInput;
    try schema.validateTimestep(options.timestep);

    var static_shard = try input.Shard.open(allocator, dataset.static.path);
    defer static_shard.deinit();
    const lx = try schema.readLx(try static_shard.tensor("lx"));

    var shard_set = try input.ShardSet.open(allocator, dataset.shard_paths);
    defer shard_set.deinit();

    const first_schema = try shard_set.shards[0].frameSchema();
    var total_frames: usize = 0;
    for (shard_set.shards) |*shard| {
        const candidate = try shard.frameSchema();
        try schema.requireCompatibleFrames(first_schema, candidate);
        total_frames = try std.math.add(usize, total_frames, candidate.frame_count);
    }

    const frame_range = try schema.resolveFrameRange(
        total_frames,
        options.frame_start,
        options.frame_stop,
        2,
    );
    const output_frames = frame_range.len;

    const centers = try allocator.alloc(f64, output_frames);
    errdefer allocator.free(centers);
    const steps = try allocator.alloc(i64, output_frames);
    defer allocator.free(steps);
    const previous_wrapped = try allocator.alloc(f64, first_schema.particle_count);
    defer allocator.free(previous_wrapped);
    const unwrapped = try allocator.alloc(f64, first_schema.particle_count);
    defer allocator.free(unwrapped);
    var workspace = ComWorkspace{
        .previous_wrapped = previous_wrapped,
        .unwrapped = unwrapped,
    };

    var initialized = false;
    var global_frame: usize = 0;
    var output_frame: usize = 0;
    outer: for (shard_set.shards) |*shard| {
        const frame_schema = try shard.frameSchema();
        const coordinates = try shard.tensor("coords");
        const shard_steps = try shard.tensor("step");
        for (0..frame_schema.frame_count) |local_frame| {
            defer global_frame += 1;
            if (global_frame < frame_range.start) continue;
            if (global_frame >= frame_range.stop) break :outer;

            const frame = try coordinates.frame(local_frame);
            centers[output_frame] = switch (frame.dtype) {
                .f32 => try unwrapFrame(
                    f32,
                    try frame.values(f32),
                    frame_schema.component_count,
                    lx,
                    &workspace,
                    initialized,
                ),
                .f64 => try unwrapFrame(
                    f64,
                    try frame.values(f64),
                    frame_schema.component_count,
                    lx,
                    &workspace,
                    initialized,
                ),
                else => return error.InvalidCoordinateDtype,
            };
            steps[output_frame] = try stepAt(shard_steps, local_frame);
            try schema.validateNextStep(
                if (output_frame > 0) steps[output_frame - 1] else null,
                steps[output_frame],
            );
            initialized = true;
            output_frame += 1;
        }
    }
    if (output_frame != output_frames) return error.FrameCountMismatch;

    const elapsed_time = try allocator.alloc(f64, output_frames);
    errdefer allocator.free(elapsed_time);
    const initial_step = steps[0];
    for (steps, elapsed_time) |step, *elapsed| {
        elapsed.* = @as(f64, @floatFromInt(step - initial_step)) * options.timestep;
    }
    const velocity = try finiteDifference(allocator, centers, elapsed_time);
    errdefer allocator.free(velocity);
    return .{
        .elapsed_time = elapsed_time,
        .center = centers,
        .velocity = velocity,
    };
}

pub fn finiteDifference(
    allocator: std.mem.Allocator,
    values: []const f64,
    coords: []const f64,
) ![]f64 {
    try schema.validateDerivativeInput(values, coords);

    const derivative = try allocator.alloc(f64, values.len);
    errdefer allocator.free(derivative);
    if (values.len == 2) {
        const slope = (values[1] - values[0]) / (coords[1] - coords[0]);
        derivative[0] = slope;
        derivative[1] = slope;
        return derivative;
    }

    for (1..values.len - 1) |index| {
        const before = coords[index] - coords[index - 1];
        const after = coords[index + 1] - coords[index];
        const a = -after / (before * (before + after));
        const b = (after - before) / (before * after);
        const c = before / (after * (before + after));
        derivative[index] = a * values[index - 1] + b * values[index] + c * values[index + 1];
    }

    const first = coords[1] - coords[0];
    const second = coords[2] - coords[1];
    derivative[0] = -(2.0 * first + second) / (first * (first + second)) * values[0] +
        (first + second) / (first * second) * values[1] -
        first / (second * (first + second)) * values[2];

    const count = values.len;
    const before = coords[count - 2] - coords[count - 3];
    const last = coords[count - 1] - coords[count - 2];
    derivative[count - 1] = last / (before * (before + last)) * values[count - 3] -
        (last + before) / (before * last) * values[count - 2] +
        (2.0 * last + before) / (last * (before + last)) * values[count - 1];
    return derivative;
}

fn stepAt(tensor: input.safetensors.TensorView, index: usize) !i64 {
    return switch (tensor.dtype) {
        .i32 => @as(i64, (try tensor.values(i32))[index]),
        .i64 => (try tensor.values(i64))[index],
        else => error.InvalidStepDtype,
    };
}

fn unwrapFrame(
    comptime T: type,
    values: []align(1) const T,
    component_count: usize,
    lx: f64,
    workspace: *ComWorkspace,
    initialized: bool,
) !f64 {
    try schema.validateCoordinateFrameLength(
        values.len,
        workspace.unwrapped.len,
        component_count,
    );
    var sum: f64 = 0.0;
    var compensation: f64 = 0.0;
    for (0..workspace.unwrapped.len) |particle| {
        const wrapped = @as(f64, @floatCast(values[particle * component_count]));
        if (!std.math.isFinite(wrapped)) return error.NonFiniteCoordinate;
        if (initialized) {
            const displacement = wrapped - workspace.previous_wrapped[particle];
            workspace.unwrapped[particle] += displacement - lx * @round(displacement / lx);
        } else {
            workspace.unwrapped[particle] = wrapped;
        }
        workspace.previous_wrapped[particle] = wrapped;

        const corrected = workspace.unwrapped[particle] - compensation;
        const updated = sum + corrected;
        compensation = (updated - sum) - corrected;
        sum = updated;
    }
    return sum / @as(f64, @floatFromInt(workspace.unwrapped.len));
}
