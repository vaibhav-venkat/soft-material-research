//! Streaming center-of-mass unwrapping and finite-difference declarations.

const std = @import("std");
const input = @import("../input/root.zig");
const Options = @import("../options.zig").Options;
const ComSeries = @import("../result.zig").ComSeries;

pub const ComWorkspace = struct {
    previous_wrapped: []f64,
    unwrapped: []f64,
};

pub fn analyzeCenterOfMass(
    allocator: std.mem.Allocator,
    dataset: input.DatasetInput,
    options: Options,
) error{NotImplemented}!ComSeries {
    _ = allocator;
    _ = dataset;
    _ = options;
    return error.NotImplemented;
}

pub fn finiteDifference(
    allocator: std.mem.Allocator,
    values: []const f64,
    coordinates: []const f64,
) error{NotImplemented}![]f64 {
    _ = allocator;
    _ = values;
    _ = coordinates;
    return error.NotImplemented;
}
