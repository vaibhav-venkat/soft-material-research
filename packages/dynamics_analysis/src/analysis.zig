//! High-level analysis entry point.

const std = @import("std");
const backend = @import("backend/root.zig");
const input = @import("input/root.zig");
const Options = @import("options.zig").Options;
const Result = @import("result.zig").Result;

pub fn analyze(
    allocator: std.mem.Allocator,
    context: *backend.Context,
    dataset: input.DatasetInput,
    options: Options,
) error{NotImplemented}!Result {
    _ = allocator;
    _ = context;
    _ = dataset;
    _ = options;
    return error.NotImplemented;
}
