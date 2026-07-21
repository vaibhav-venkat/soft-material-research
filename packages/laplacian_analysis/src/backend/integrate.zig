//! Simpson weights and recurrence-based complex Laplace integration.

const std = @import("std");
const dynamics_analysis = @import("dynamics_analysis");

pub const ComplexValue = struct {
    real: f64,
    imaginary: f64,
};

pub fn integrateLaplaceSimpson(
    correlation: dynamics_analysis.CorrelationSeries,
    weights: []const f64,
    r: f64,
    omega: f64,
) !ComplexValue {
    if (weights.len != correlation.lag_times.len) return error.DimensionMismatch;
    var real: f64 = 0.0;
    var imaginary: f64 = 0.0;
    const first_time = correlation.lag_times[0];
    const spacing = correlation.lag_times[1] - first_time;
    var envelope = @exp(r * first_time);
    const envelope_step = @exp(r * spacing);
    const first_phase = omega * first_time;
    const phase_step = omega * spacing;
    var cosine = @cos(first_phase);
    var sine = @sin(first_phase);
    const cosine_step = @cos(phase_step);
    const sine_step = @sin(phase_step);

    for (correlation.pearson, weights, 0..) |pearson, weight, index| {
        const weighted = weight * pearson * envelope;
        real += weighted * cosine;
        imaginary += weighted * sine;

        if (index + 1 < weights.len) {
            envelope *= envelope_step;
            const next_cosine = cosine * cosine_step - sine * sine_step;
            sine = sine * cosine_step + cosine * sine_step;
            cosine = next_cosine;
        }
    }
    if (!std.math.isFinite(real) or !std.math.isFinite(imaginary)) {
        return error.NonFiniteTransform;
    }
    return .{ .real = real, .imaginary = imaginary };
}

pub fn simpsonWeights(
    allocator: std.mem.Allocator,
    sample_count: usize,
    spacing: f64,
) ![]f64 {
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
    for (weights[0..simpson_count], 0..) |*weight, index| {
        const multiplier: f64 = if (index == 0 or index + 1 == simpson_count)
            1.0
        else if (index % 2 == 1)
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
