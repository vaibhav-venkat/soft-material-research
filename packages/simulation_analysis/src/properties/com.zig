//! Streaming unwrapped center of mass and physical-time finite differences.

const std = @import("std");
const coordinates_module = @import("../data_structures/coordinates.zig");

/// One frame-level cylindrical vector.
pub const CylindricalFrameValue = struct {
    x: f32,
    theta: f32,
    r: f32,
};

/// SoA storage for frame-level cylindrical values.
pub const CylindricalFrameValues = std.MultiArrayList(CylindricalFrameValue);

pub const Error = error{
    BufferSizeMismatch,
    InvalidParticleCount,
    InvalidPeriod,
    InvalidTimestep,
    InvalidStepSequence,
    NotEnoughFrames,
    NonFiniteCoordinate,
    OutOfMemory,
};

/// Persistent per-particle state required to continue unwrapping across shards.
pub const Workspace = struct {
    allocator: std.mem.Allocator,
    previous_x: []f64,
    previous_theta: []f64,
    unwrapped_x: []f64,
    unwrapped_theta: []f64,
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator, particle_count: usize) Error!Workspace {
        if (particle_count == 0) return error.InvalidParticleCount;
        const previous_x = allocator.alloc(f64, particle_count) catch return error.OutOfMemory;
        errdefer allocator.free(previous_x);
        const previous_theta = allocator.alloc(f64, particle_count) catch return error.OutOfMemory;
        errdefer allocator.free(previous_theta);
        const unwrapped_x = allocator.alloc(f64, particle_count) catch return error.OutOfMemory;
        errdefer allocator.free(unwrapped_x);
        const unwrapped_theta = allocator.alloc(f64, particle_count) catch return error.OutOfMemory;
        return .{
            .allocator = allocator,
            .previous_x = previous_x,
            .previous_theta = previous_theta,
            .unwrapped_x = unwrapped_x,
            .unwrapped_theta = unwrapped_theta,
        };
    }

    pub fn deinit(self: *Workspace) void {
        self.allocator.free(self.previous_x);
        self.allocator.free(self.previous_theta);
        self.allocator.free(self.unwrapped_x);
        self.allocator.free(self.unwrapped_theta);
        self.* = undefined;
    }
};

/// Calculate one unwrapped cylindrical center of mass per frame.
///
/// Calls may supply consecutive shards to the same `workspace`. Axial x is
/// unwrapped with period `lx`; theta is unwrapped with period `2*pi`; radial r
/// is averaged directly. Coordinates are logically frame-major within each SoA
/// field and `destination` contains one value per frame.
pub fn com_unwrapped(
    workspace: *Workspace,
    coordinates: coordinates_module.CylindricalCoordinates.Slice,
    particle_count: usize,
    lx: f64,
    destination: CylindricalFrameValues.Slice,
) Error!void {
    if (particle_count == 0 or workspace.unwrapped_x.len != particle_count) return error.InvalidParticleCount;
    if (!std.math.isFinite(lx) or lx <= 0) return error.InvalidPeriod;
    if (coordinates.len % particle_count != 0) return error.BufferSizeMismatch;
    const frame_count = coordinates.len / particle_count;
    if (destination.len != frame_count) return error.BufferSizeMismatch;

    const input_x = coordinates.items(.x);
    const input_theta = coordinates.items(.theta);
    const input_r = coordinates.items(.r);
    const output_x = destination.items(.x);
    const output_theta = destination.items(.theta);
    const output_r = destination.items(.r);
    const theta_period = 2.0 * std.math.pi;

    for (0..frame_count) |frame| {
        var sum_x: f64 = 0;
        var sum_theta: f64 = 0;
        var sum_r: f64 = 0;
        var compensation_x: f64 = 0;
        var compensation_theta: f64 = 0;
        var compensation_r: f64 = 0;
        for (0..particle_count) |particle| {
            const index = frame * particle_count + particle;
            const wrapped_x: f64 = input_x[index];
            const wrapped_theta: f64 = input_theta[index];
            const radius: f64 = input_r[index];
            if (!std.math.isFinite(wrapped_x) or !std.math.isFinite(wrapped_theta) or !std.math.isFinite(radius)) {
                return error.NonFiniteCoordinate;
            }
            if (workspace.initialized) {
                const dx = wrapped_x - workspace.previous_x[particle];
                const dtheta = wrapped_theta - workspace.previous_theta[particle];
                workspace.unwrapped_x[particle] += dx - lx * @round(dx / lx);
                workspace.unwrapped_theta[particle] += dtheta - theta_period * @round(dtheta / theta_period);
            } else {
                workspace.unwrapped_x[particle] = wrapped_x;
                workspace.unwrapped_theta[particle] = wrapped_theta;
            }
            workspace.previous_x[particle] = wrapped_x;
            workspace.previous_theta[particle] = wrapped_theta;
            kahanAdd(&sum_x, &compensation_x, workspace.unwrapped_x[particle]);
            kahanAdd(&sum_theta, &compensation_theta, workspace.unwrapped_theta[particle]);
            kahanAdd(&sum_r, &compensation_r, radius);
        }
        const denominator: f64 = @floatFromInt(particle_count);
        output_x[frame] = @floatCast(sum_x / denominator);
        output_theta[frame] = @floatCast(sum_theta / denominator);
        output_r[frame] = @floatCast(sum_r / denominator);
        workspace.initialized = true;
    }
}

/// Differentiate unwrapped COM values with respect to physical time.
/// Physical time is `step * timestep`; nonuniform positive step spacing is
/// supported with three-point finite differences and one-sided endpoints.
pub fn com_velocity_unwrapped(
    com: CylindricalFrameValues.Slice,
    steps: []const u64,
    timestep: f64,
    destination: CylindricalFrameValues.Slice,
) Error!void {
    if (!std.math.isFinite(timestep) or timestep <= 0) return error.InvalidTimestep;
    if (com.len != steps.len or destination.len != steps.len) return error.BufferSizeMismatch;
    if (steps.len < 2) return error.NotEnoughFrames;
    for (1..steps.len) |index| if (steps[index] <= steps[index - 1]) return error.InvalidStepSequence;
    try finiteDifference(com.items(.x), steps, timestep, destination.items(.x));
    try finiteDifference(com.items(.theta), steps, timestep, destination.items(.theta));
    try finiteDifference(com.items(.r), steps, timestep, destination.items(.r));
}

fn finiteDifference(values: []const f32, steps: []const u64, timestep: f64, derivative: []f32) Error!void {
    if (values.len == 2) {
        const elapsed = stepDistance(steps[0], steps[1], timestep);
        const slope = (@as(f64, values[1]) - @as(f64, values[0])) / elapsed;
        derivative[0] = @floatCast(slope);
        derivative[1] = @floatCast(slope);
        return;
    }
    for (1..values.len - 1) |index| {
        const before = stepDistance(steps[index - 1], steps[index], timestep);
        const after = stepDistance(steps[index], steps[index + 1], timestep);
        derivative[index] = @floatCast(
            -after / (before * (before + after)) * @as(f64, values[index - 1]) +
                (after - before) / (before * after) * @as(f64, values[index]) +
                before / (after * (before + after)) * @as(f64, values[index + 1]),
        );
    }
    const first = stepDistance(steps[0], steps[1], timestep);
    const second = stepDistance(steps[1], steps[2], timestep);
    derivative[0] = @floatCast(
        -(2.0 * first + second) / (first * (first + second)) * @as(f64, values[0]) +
            (first + second) / (first * second) * @as(f64, values[1]) -
            first / (second * (first + second)) * @as(f64, values[2]),
    );
    const last_index = values.len - 1;
    const before = stepDistance(steps[last_index - 2], steps[last_index - 1], timestep);
    const last = stepDistance(steps[last_index - 1], steps[last_index], timestep);
    derivative[last_index] = @floatCast(
        last / (before * (before + last)) * @as(f64, values[last_index - 2]) -
            (last + before) / (before * last) * @as(f64, values[last_index - 1]) +
            (2.0 * last + before) / (last * (before + last)) * @as(f64, values[last_index]),
    );
}

fn stepDistance(before: u64, after: u64, timestep: f64) f64 {
    return @as(f64, @floatFromInt(after - before)) * timestep;
}

fn kahanAdd(sum: *f64, compensation: *f64, value: f64) void {
    const corrected = value - compensation.*;
    const updated = sum.* + corrected;
    compensation.* = (updated - sum.*) - corrected;
    sum.* = updated;
}
