//! Lagged Pearson-correlation and replicate-aggregation declarations.

const std = @import("std");
const backend = @import("../backend/root.zig");
const Options = @import("../options.zig").Options;
const result = @import("../result.zig");

pub fn analyzeCorrelation(
    allocator: std.mem.Allocator,
    context: *backend.Context,
    com: result.ComSeries,
    options: Options,
) error{NotImplemented}!result.CorrelationSeries {
    _ = allocator;
    _ = context;
    _ = com;
    _ = options;
    return error.NotImplemented;
}
