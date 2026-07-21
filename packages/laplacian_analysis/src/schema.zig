//! Central validation contracts shared by transform and fit implementations.

const std = @import("std");
const dynamics_analysis = @import("dynamics_analysis");
const options = @import("options.zig");

pub fn validateCorrelation(correlation: dynamics_analysis.CorrelationSeries) !f64 {
    const count = correlation.lag_times.len;
    if (count < 2) return error.TooFewLags;
    if (correlation.pearson.len != count or correlation.origin_counts.len != count) {
        return error.DimensionMismatch;
    }
    const spacing = correlation.lag_times[1] - correlation.lag_times[0];
    if (!std.math.isFinite(spacing) or spacing <= 0.0) return error.InvalidLagSpacing;
    for (correlation.lag_times, correlation.pearson, 0..) |time, value, index| {
        if (!std.math.isFinite(time)) return error.NonFiniteTime;
        if (!std.math.isFinite(value)) return error.NonFiniteCorrelation;
        if (index == 0) continue;
        const actual = time - correlation.lag_times[index - 1];
        const tolerance = @max(1.0e-12, 1.0e-10 * @max(@abs(spacing), @abs(actual)));
        if (@abs(actual - spacing) > tolerance) return error.NonUniformLags;
    }
    return spacing;
}

pub fn validateTransformOptions(value: options.TransformOptions) !void {
    if (value.r_points < 2 or value.omega_points < 2) return error.TooFewGridPoints;
    if (!std.math.isFinite(value.r_max)) return error.InvalidRBounds;
    if (value.r_min) |minimum| {
        if (!std.math.isFinite(minimum) or minimum >= value.r_max) return error.InvalidRBounds;
    }
    if (value.omega_min) |minimum| if (!std.math.isFinite(minimum)) {
        return error.InvalidOmegaBounds;
    };
    if (value.omega_max) |maximum| if (!std.math.isFinite(maximum)) {
        return error.InvalidOmegaBounds;
    };
    if (value.omega_min != null and value.omega_max != null and
        value.omega_min.? >= value.omega_max.?) return error.InvalidOmegaBounds;
}

pub fn validateFitOptions(value: options.FitOptions) !void {
    if (!std.math.isFinite(value.soft_l1_scale) or value.soft_l1_scale <= 0.0) {
        return error.InvalidSoftL1Scale;
    }
    if (!std.math.isFinite(value.tolerance) or value.tolerance <= 0.0) {
        return error.InvalidTolerance;
    }
    if (value.maximum_evaluations < 2) return error.InvalidEvaluationLimit;
    if (!std.math.isFinite(value.rank_tolerance) or value.rank_tolerance < 0.0) {
        return error.InvalidRankTolerance;
    }
}
