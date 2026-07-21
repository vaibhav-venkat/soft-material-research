//! Transform axes, complex grid evaluation, and preferred-coordinate searches.

const std = @import("std");
const dynamics_analysis = @import("dynamics_analysis");
const options = @import("options.zig");
const result = @import("result.zig");
const schema = @import("schema.zig");

pub fn transformAxes(
    allocator: std.mem.Allocator,
    correlation: dynamics_analysis.CorrelationSeries,
    transform_options: options.TransformOptions,
) !result.TransformAxes {
    try schema.validateTransformOptions(transform_options);
    const dt = try schema.validateCorrelation(correlation);
    const duration: f64 = correlation.lag_times[correlation.lag_times.len - 1] - correlation.lag_times[0];
    if (!std.math.isFinite(duration) or duration <= 0.0) {
        return error.DurationInvalid;
    }
    const nyquist = std.math.pi / dt;
    const default_omega_max: f64 = @min(std.math.pi / dt, 20.0 * std.math.pi / duration);
    const omega_max = transform_options.omega_max orelse default_omega_max;
    const omega_min = transform_options.omega_min orelse -default_omega_max;
    const r_max = transform_options.r_max;
    const default_r_min: f64 = -10.0 / duration;
    const r_min = transform_options.r_min orelse default_r_min;

    if (!std.math.isFinite(r_min) or !std.math.isFinite(r_max) or r_min >= r_max) {
        return error.InvalidRBounds;
    }
    if (!std.math.isFinite(omega_min) or !std.math.isFinite(omega_max) or
        omega_min >= omega_max or omega_min < -nyquist or omega_max > nyquist)
    {
        return error.InvalidOmegaBounds;
    }

    const r = try allocator.alloc(f64, transform_options.r_points);
    errdefer allocator.free(r);
    const omega = try allocator.alloc(f64, transform_options.omega_points);
    errdefer allocator.free(omega);
    fillLinspace(r, r_min, r_max);
    fillLinspace(omega, omega_min, omega_max);
    return .{ .r = r, .omega = omega };
}

fn fillLinspace(values: []f64, minimum: f64, maximum: f64) void {
    const denominator = @as(f64, @floatFromInt(values.len - 1));
    const step = (maximum - minimum) / denominator;
    for (values, 0..) |*value, index| {
        value.* = minimum + step * @as(f64, @floatFromInt(index));
    }
    values[values.len - 1] = maximum;
}

pub fn preferredAxes(
    allocator: std.mem.Allocator,
    correlation: dynamics_analysis.CorrelationSeries,
    preferred_options: options.TransformOptions,
) !result.TransformAxes {
    if (preferred_options.r_min orelse -1.0 >= 0.0 or preferred_options.omega_max orelse 1.0 <= 0.0) {
        return error.InvalidPreferredRBounds;
    }
    return try transformAxes(
        allocator,
        correlation,
        .{
            .r_min = preferred_options.r_min,
            .r_max = -std.math.floatEps(f64),
            .omega_min = std.math.floatEps(f64),
            .omega_max = preferred_options.omega_max,
            .omega_points = preferred_options.omega_points,
            .r_points = preferred_options.r_points
    });
}

pub fn analyzeLaplace(
    allocator: std.mem.Allocator,
    context: *dynamics_analysis.backend.Context,
    correlation: dynamics_analysis.CorrelationSeries,
    axes: result.TransformAxes,
) !result.LaplaceGrid {
    // Ownership of `axes` transfers to this function on every call. On
    // success the returned grid owns the slices; on failure they are freed.
    errdefer axes.deinit(allocator);
    const spacing = try schema.validateCorrelation(correlation);
    try validateAxis(axes.r);
    try validateAxis(axes.omega);
    _ = context;

    const value_count = std.math.mul(usize, axes.omega.len, axes.r.len) catch {
        return error.GridSizeOverflow;
    };
    const values_real = try allocator.alloc(f64, value_count);
    errdefer allocator.free(values_real);
    const values_imag = try allocator.alloc(f64, value_count);
    errdefer allocator.free(values_imag);
    const weights = try simpsonWeights(allocator, correlation.lag_times.len, spacing);
    defer allocator.free(weights);

    // Flatten in `(omega, r)` row-major order. A future accelerated backend
    // can replace this host traversal while preserving the public layout.
    for (axes.omega, 0..) |omega_value, omega_index| {
        for (axes.r, 0..) |r_value, r_index| {
            const index = omega_index * axes.r.len + r_index;
            const value = try integrateLaplaceSimpson(
                correlation,
                weights,
                r_value,
                omega_value,
            );
            values_real[index] = value.real;
            values_imag[index] = value.imaginary;
        }
    }

    return .{
        .r = axes.r,
        .omega = axes.omega,
        .values_real = values_real,
        .values_imag = values_imag,
        .shape = .{ axes.omega.len, axes.r.len },
    };
}

const ComplexValue = struct { real: f64, imaginary: f64 };

fn integrateLaplaceSimpson(
    correlation: dynamics_analysis.CorrelationSeries,
    weights: []const f64,
    r: f64,
    omega: f64,
) !ComplexValue {
    if (weights.len != correlation.lag_times.len) return error.DimensionMismatch;
    var real: f64 = 0.0;
    var imaginary: f64 = 0.0;
    for (
        correlation.lag_times,
        correlation.pearson,
        weights,
    ) |time, pearson, weight| {
        const envelope = @exp(r * time);
        const phase = omega * time;
        const weighted = weight * pearson * envelope;
        real += weighted * @cos(phase);
        imaginary += weighted * @sin(phase);
    }
    if (!std.math.isFinite(real) or !std.math.isFinite(imaginary)) {
        return error.NonFiniteTransform;
    }
    return .{ .real = real, .imaginary = imaginary };
}

fn simpsonWeights(allocator: std.mem.Allocator, sample_count: usize, spacing: f64) ![]f64 {
    if (sample_count < 2 or !std.math.isFinite(spacing) or spacing <= 0.0) {
        return error.InvalidSamples;
    }
    const weights = try allocator.alloc(f64, sample_count);
    @memset(weights, 0.0);
    if (sample_count == 2) {
        weights[0] = spacing / 2.0;
        weights[1] = spacing / 2.0;
        return weights;
    }
    const simpson_count = if (sample_count % 2 == 0) sample_count - 1 else sample_count;
    for (weights[0..simpson_count], 0..) |*weight, i| {
        const multiplier: f64 = if (i == 0 or i + 1 == simpson_count)
            1.0
        else if (i % 2 == 1)
            4.0
        else
            2.0;
        weight.* = multiplier * spacing / 3.0;
    }
    if (sample_count % 2 == 0) {
        weights[sample_count - 1] += 5.0 * spacing / 12.0;
        weights[sample_count - 2] += 2.0 * spacing / 3.0;
        weights[sample_count - 3] -= spacing / 12.0;
    }
    return weights;
}

fn validateAxis(axis: []const f64) !void {
    if (axis.len < 2) return error.TooFewGridPoints;
    for (axis, 0..) |value, index| {
        if (!std.math.isFinite(value)) return error.NonFiniteGrid;
        if (index > 0 and value <= axis[index - 1]) return error.NonIncreasingGrid;
    }
}

pub fn preferredCoordinate(
    allocator: std.mem.Allocator,
    context: *dynamics_analysis.backend.Context,
    correlation: dynamics_analysis.CorrelationSeries,
    axis: result.PreferredAxis,
    coordinates: []const f64,
) !result.PreferredEstimate {
    _ = allocator;
    _ = context;
    _ = correlation;
    _ = axis;
    _ = coordinates;
    return error.NotImplemented;
}
