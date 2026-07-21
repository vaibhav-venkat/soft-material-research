//! Transform axes, complex grid evaluation, and preferred-coordinate searches.

const std = @import("std");
const dynamics_analysis = @import("dynamics_analysis");
const options = @import("options.zig");
const result = @import("result.zig");

pub fn transformAxes(
    allocator: std.mem.Allocator,
    correlation: dynamics_analysis.CorrelationSeries,
    transform_options: options.TransformOptions,
) !result.TransformAxes {
    _ = allocator;
    _ = correlation;
    _ = transform_options;
    return error.NotImplemented;
}

pub fn preferredAxes(
    allocator: std.mem.Allocator,
    correlation: dynamics_analysis.CorrelationSeries,
    preferred_options: options.PreferredOptions,
) !result.PreferredAxes {
    _ = allocator;
    _ = correlation;
    _ = preferred_options;
    return error.NotImplemented;
}

pub fn analyzeLaplace(
    allocator: std.mem.Allocator,
    context: *dynamics_analysis.backend.Context,
    correlation: dynamics_analysis.CorrelationSeries,
    axes: result.TransformAxes,
) !result.LaplaceGrid {
    _ = allocator;
    _ = context;
    _ = correlation;
    _ = axes;
    return error.NotImplemented;
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
