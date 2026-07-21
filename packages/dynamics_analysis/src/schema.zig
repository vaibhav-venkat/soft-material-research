//! Central schema and numerical-series validation.

const std = @import("std");
const safetensors = @import("safetensors");

pub const FrameSchema = struct {
    frame_count: usize,
    particle_count: usize,
    component_count: usize,
    coordinate_dtype: safetensors.Dtype,
    step_dtype: safetensors.Dtype,
};

pub const FrameRange = struct {
    start: usize,
    stop: usize,
    len: usize,
};

pub fn inspectFrameSchema(
    coordinates: safetensors.TensorView,
    steps: safetensors.TensorView,
) !FrameSchema {
    if (coordinates.shape.len != 3 or coordinates.shape[1] == 0 or coordinates.shape[2] == 0) {
        return error.InvalidCoordinateShape;
    }
    if (steps.shape.len != 1 or steps.shape[0] != coordinates.shape[0]) {
        return error.InvalidStepShape;
    }
    if (coordinates.dtype != .f32 and coordinates.dtype != .f64) {
        return error.InvalidCoordinateDtype;
    }
    if (steps.dtype != .i32 and steps.dtype != .i64) return error.InvalidStepDtype;
    return .{
        .frame_count = coordinates.shape[0],
        .particle_count = coordinates.shape[1],
        .component_count = coordinates.shape[2],
        .coordinate_dtype = coordinates.dtype,
        .step_dtype = steps.dtype,
    };
}

pub fn requireCompatibleFrames(reference: FrameSchema, candidate: FrameSchema) !void {
    if (candidate.particle_count != reference.particle_count or
        candidate.component_count != reference.component_count)
    {
        return error.InconsistentShardShape;
    }
}

pub fn resolveFrameRange(
    total_frames: usize,
    frame_start: usize,
    requested_stop: ?usize,
    minimum_frames: usize,
) !FrameRange {
    const stop = requested_stop orelse total_frames;
    if (frame_start >= stop or stop > total_frames) return error.InvalidFrameRange;
    const len = stop - frame_start;
    if (len < minimum_frames) return error.TooFewFrames;
    return .{ .start = frame_start, .stop = stop, .len = len };
}

pub fn validateTimestep(timestep: f64) !void {
    if (!std.math.isFinite(timestep) or timestep <= 0.0) return error.InvalidTimestep;
}

pub fn readLx(tensor: safetensors.TensorView) !f64 {
    if (tensor.elementCount() != 1) return error.InvalidLxShape;
    const lx = switch (tensor.dtype) {
        .f32 => @as(f64, @floatCast((try tensor.values(f32))[0])),
        .f64 => (try tensor.values(f64))[0],
        else => return error.InvalidLxDtype,
    };
    if (!std.math.isFinite(lx) or lx <= 0.0) return error.InvalidLx;
    return lx;
}

pub fn validateNextStep(previous: ?i64, current: i64) !void {
    if (previous) |value| {
        if (current <= value) return error.NonIncreasingSteps;
    }
}

pub fn validateDerivativeInput(values: []const f64, coordinates: []const f64) !void {
    if (values.len != coordinates.len) return error.DimensionMismatch;
    if (values.len < 2) return error.TooFewSamples;
    for (values) |value| {
        if (!std.math.isFinite(value)) return error.NonFiniteValue;
    }
    for (coordinates, 0..) |coordinate, index| {
        if (!std.math.isFinite(coordinate)) return error.NonFiniteCoordinate;
        if (index > 0 and coordinate <= coordinates[index - 1]) {
            return error.NonIncreasingCoordinates;
        }
    }
}

pub fn uniformComSpacing(elapsed_time: []const f64, velocity_len: usize) !f64 {
    if (velocity_len != elapsed_time.len) return error.DimensionMismatch;
    if (elapsed_time.len < 3) return error.TooFewSamples;
    const spacing = elapsed_time[1] - elapsed_time[0];
    if (!std.math.isFinite(spacing) or spacing <= 0.0) return error.InvalidTimeSpacing;
    for (elapsed_time, 0..) |time, index| {
        if (!std.math.isFinite(time)) return error.NonFiniteTime;
        if (index == 0) continue;
        const actual = time - elapsed_time[index - 1];
        const tolerance = @max(1.0e-12, 1.0e-10 * @max(@abs(spacing), @abs(actual)));
        if (@abs(actual - spacing) > tolerance) return error.NonUniformTime;
    }
    return spacing;
}

pub fn validatePearsonInput(velocity: []const f64, max_lag: usize) !void {
    if (velocity.len < 3) return error.TooFewSamples;
    if (max_lag > velocity.len - 2) return error.InvalidMaxLag;
    for (velocity) |value| {
        if (!std.math.isFinite(value)) return error.NonFiniteVelocity;
    }
}

pub fn validateCoordinateFrameLength(
    element_count: usize,
    particle_count: usize,
    component_count: usize,
) !void {
    const expected = std.math.mul(usize, particle_count, component_count) catch
        return error.InvalidCoordinateShape;
    if (element_count != expected) return error.InvalidCoordinateShape;
}
