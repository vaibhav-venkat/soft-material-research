//! Owned cluster-analysis results returned to Python.

const std = @import("std");

/// Ragged, periodically unwrapped cluster coordinates pooled over selected frames.
///
/// Cluster `i` occupies `points[offsets[i]..offsets[i + 1]]`.
pub const Result = struct {
    points: [][2]f64,
    offsets: []usize,

    pub fn deinit(self: Result, allocator: std.mem.Allocator) void {
        allocator.free(self.offsets);
        allocator.free(self.points);
    }
};
