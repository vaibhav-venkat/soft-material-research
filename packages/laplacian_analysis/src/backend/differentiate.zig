//! Analytic derivatives of Laplacian-analysis models.

const std = @import("std");
const schema = @import("../schema.zig");

pub const parameter_count = 4;
pub const Parameters = [parameter_count]f64;

pub const GradientSystem = struct {
    normal_matrix: [parameter_count * parameter_count]f64,
    gradient: Parameters,
    column_norm_squared: Parameters,
};

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

pub fn assembleSoftL1GradientSystem(
    prediction: []const f64,
    observed: []const f64,
    weights: []const f64,
    jacobian: []const f64,
    scale: f64,
) !GradientSystem {
    if (observed.len != prediction.len or weights.len != prediction.len) {
        return error.DimensionMismatch;
    }
    if (jacobian.len != try std.math.mul(usize, prediction.len, parameter_count)) {
        return error.DimensionMismatch;
    }
    if (!std.math.isFinite(scale) or scale <= 0.0) return error.InvalidSoftL1Scale;

    var system = GradientSystem{
        .normal_matrix = @splat(0.0),
        .gradient = @splat(0.0),
        .column_norm_squared = @splat(0.0),
    };
    for (0..prediction.len) |row| {
        const weighted_residual = weights[row] * (prediction[row] - observed[row]);
        const scaled_residual = weighted_residual / scale;
        const robust_weight = 1.0 / @sqrt(std.math.hypot(1.0, scaled_residual));
        const residual = robust_weight * weighted_residual;
        var jacobian_row: Parameters = @splat(0.0);
        for (0..parameter_count) |column| {
            const value = robust_weight * weights[row] *
                jacobian[row * parameter_count + column];
            jacobian_row[column] = value;
            system.column_norm_squared[column] += value * value;
            system.gradient[column] += value * residual;
        }
        for (0..parameter_count) |left| {
            for (0..parameter_count) |right| {
                system.normal_matrix[left * parameter_count + right] +=
                    jacobian_row[left] * jacobian_row[right];
            }
        }
    }
    return system;
}

pub fn projectedScaledGradientMaximum(
    parameters: Parameters,
    gradient: Parameters,
    column_norm_squared: Parameters,
    lower: Parameters,
    upper: Parameters,
) f64 {
    var maximum: f64 = 0.0;
    for (0..parameter_count) |index| {
        const at_lower = parameters[index] <= lower[index];
        const at_upper = parameters[index] >= upper[index];
        const projected = if ((at_lower and gradient[index] > 0.0) or
            (at_upper and gradient[index] < 0.0))
            0.0
        else
            @abs(gradient[index]) / @max(@sqrt(column_norm_squared[index]), 1.0e-12);
        maximum = @max(maximum, projected);
    }
    return maximum;
}
