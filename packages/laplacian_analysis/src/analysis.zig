//! Full single-dataset orchestration; Python is responsible for replicates.

const std = @import("std");
const dynamics_analysis = @import("dynamics_analysis");
const fit = @import("fit.zig");
const laplace = @import("laplace.zig");
const Options = @import("options.zig").Options;
const Result = @import("result.zig").Result;

pub fn analyze(
    allocator: std.mem.Allocator,
    context: *dynamics_analysis.backend.Context,
    dataset: dynamics_analysis.input.DatasetInput,
    options: Options,
) !Result {
    const dynamics = try dynamics_analysis.analyze(
        allocator,
        context,
        dataset,
        options.dynamics,
    );
    errdefer dynamics.deinit(allocator);

    const axes = try laplace.transformAxes(allocator, dynamics.correlation, options.transform);
    const grid = try laplace.analyzeLaplace(allocator, context, dynamics.correlation, axes);
    errdefer grid.deinit(allocator);
    const preferred_axes = try laplace.preferredAxes(
        allocator,
        dynamics.correlation,
        options.preferred,
    );
    defer preferred_axes.deinit(allocator);
    const preferred_r = try laplace.preferredCoordinate(
        allocator,
        context,
        dynamics.correlation,
        .r,
        preferred_axes.r,
    );
    const preferred_omega = try laplace.preferredCoordinate(
        allocator,
        context,
        dynamics.correlation,
        .omega,
        preferred_axes.omega,
    );
    const fitted = try fit.fitDampedCosine(
        allocator,
        context,
        dynamics.correlation,
        preferred_axes.omega,
        options.fit,
    );
    return .{
        .dynamics = dynamics,
        .laplace = grid,
        .preferred_r = preferred_r,
        .preferred_omega = preferred_omega,
        .fit = fitted,
    };
}
