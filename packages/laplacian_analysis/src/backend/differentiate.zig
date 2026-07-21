//! Analytic derivatives of Laplacian-analysis models.

const std = @import("std");
const schema = @import("../schema.zig");

pub const parameter_count = 4;
pub const Parameters = [parameter_count]f64;

pub fn dampedCosineJacobian(
    allocator: std.mem.Allocator,
    time: []const f64,
    parameters: Parameters,
) ![]f64 {
    const count = try std.math.mul(usize, time.len, parameter_count);
    const jacobian = try allocator.alloc(f64, count);
    errdefer allocator.free(jacobian);
    try fillDampedCosineJacobian(jacobian, time, parameters);
    return jacobian;
}

pub fn fillDampedCosineJacobian(
    output: []f64,
    time: []const f64,
    parameters: Parameters,
) !void {
    if (output.len != try std.math.mul(usize, time.len, parameter_count)) {
        return error.DimensionMismatch;
    }
    try schema.validateModelInput(time, &parameters);
    const amplitude = parameters[0];
    const rate = parameters[1];
    const omega = parameters[2];
    const phase = parameters[3];
    const phase_cosine = @cos(phase);
    const phase_sine = @sin(phase);
    for (time, 0..) |sample_time, row| {
        const envelope = @exp(-rate * sample_time);
        const argument = omega * sample_time + phase;
        const sine = @sin(argument);
        const cosine = @cos(argument);
        const start = row * parameter_count;
        output[start] = envelope * cosine - phase_cosine;
        output[start + 1] = -amplitude * sample_time * envelope * cosine;
        output[start + 2] = -amplitude * sample_time * envelope * sine;
        output[start + 3] = -amplitude * envelope * sine + amplitude * phase_sine;
    }
}
