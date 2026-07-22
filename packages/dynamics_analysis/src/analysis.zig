//! High-level dynamics analysis orchestration.

const std = @import("std");
const backend = @import("backend/root.zig");
const dynamics = @import("dynamics/root.zig");
const input = @import("input/root.zig");
const Options = @import("options.zig").Options;
const Result = @import("result.zig").Result;

pub fn analyze(
    allocator: std.mem.Allocator,
    context: *backend.Context,
    dataset: input.DatasetInput,
    options: Options,
) !Result {
    const com = try dynamics.analyzeCenterOfMass(allocator, dataset, options);
    errdefer com.deinit(allocator);
    const correlation = try dynamics.analyzeCorrelation(allocator, context, com, options);
    return .{ .com = com, .correlation = correlation };
}
