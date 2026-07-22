//! Owned cluster-analysis results returned to Python.

const std = @import("std");

/// One ratio per structural cluster, pooled over the selected frames.
pub const Result = struct {
    ratios: []f64,

    pub fn deinit(self: Result, allocator: std.mem.Allocator) void {
        allocator.free(self.ratios);
    }
};
