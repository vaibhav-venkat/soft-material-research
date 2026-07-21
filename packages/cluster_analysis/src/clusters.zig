//! Structural-cluster numerical entry points.

const std = @import("std");
const Options = @import("options.zig").Options;
const Result = @import("result.zig").Result;
const schema = @import("schema.zig");
const kdtree = @import("")
/// Analyze one already-projected surface frame.
///
/// Planned implementation: periodic k-d-tree neighbors, ψ₆ compatibility,
/// connected components, then one selected area ratio per component.
pub fn analyzeStructuralFrame(
    allocator: std.mem.Allocator,
    frame: schema.StructuralFrame,
    options: Options,
) !Result {
    try schema.validateOptions(options);
    try schema.validateStructuralFrame(frame);
    _ = allocator;
    return error.NotImplemented;
}
