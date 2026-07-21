//! Constrained damped-cosine model, Jacobian, and robust multistart fit.

const std = @import("std");
const dynamics_analysis = @import("dynamics_analysis");
const options = @import("options.zig");
const result = @import("result.zig");

pub const parameter_count = 4;
pub const Parameters = [parameter_count]f64;

pub fn dampedCosine(
    allocator: std.mem.Allocator,
    time: []const f64,
    parameters: Parameters,
) ![]f64 {
    _ = allocator;
    _ = time;
    _ = parameters;
    return error.NotImplemented;
}

pub fn dampedCosineJacobian(
    allocator: std.mem.Allocator,
    time: []const f64,
    parameters: Parameters,
) ![]f64 {
    _ = allocator;
    _ = time;
    _ = parameters;
    return error.NotImplemented;
}

pub fn fitDampedCosine(
    allocator: std.mem.Allocator,
    context: *dynamics_analysis.backend.Context,
    correlation: dynamics_analysis.CorrelationSeries,
    omega_grid: []const f64,
    fit_options: options.FitOptions,
) !result.DampedCosineFit {
    _ = allocator;
    _ = context;
    _ = correlation;
    _ = omega_grid;
    _ = fit_options;
    return error.NotImplemented;
}
