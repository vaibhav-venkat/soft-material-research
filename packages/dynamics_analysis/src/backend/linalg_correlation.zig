//! Linalg-backed correlation declaration.

const std = @import("std");
const linalg = @import("linalg");

pub fn laggedPearson(
    allocator: std.mem.Allocator,
    context: *linalg.Context,
    velocity: []const f64,
    max_lag: usize,
) error{NotImplemented}![]f64 {
    _ = allocator;
    _ = context;
    _ = velocity;
    _ = max_lag;
    return error.NotImplemented;
}
