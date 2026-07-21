//! Small ownership-safe wrapper around jtsiomb/kdtree's C API.

const std = @import("std");
const c = @import("kdtree_c");

pub const Tree2 = struct {
    handle: *c.struct_kdtree,

    pub fn init() !Tree2 {
        return .{ .handle = c.kd_create(2) orelse return error.OutOfMemory };
    }

    pub fn deinit(self: *Tree2) void {
        c.kd_free(self.handle);
        self.* = undefined;
    }

    pub fn clear(self: *Tree2) void {
        c.kd_clear(self.handle);
    }

    pub fn insert(self: *Tree2, point: [2]f64, index: usize) !void {
        const token = std.math.add(usize, index, 1) catch return error.IndexOverflow;
        const data: *anyopaque = @ptrFromInt(token);
        if (c.kd_insert(self.handle, &point, data) != 0) return error.OutOfMemory;
    }

    /// Return caller-owned particle indices within the Euclidean radius.
    pub fn within(
        self: *Tree2,
        allocator: std.mem.Allocator,
        point: [2]f64,
        radius: f64,
    ) ![]usize {
        if (!std.math.isFinite(radius) or radius < 0.0) return error.InvalidRadius;
        const results = c.kd_nearest_range(self.handle, &point, radius) orelse
            return error.OutOfMemory;
        defer c.kd_res_free(results);
        var indices: std.ArrayList(usize) = .empty;
        errdefer indices.deinit(allocator);
        while (c.kd_res_end(results) == 0) {
            const data = c.kd_res_item_data(results) orelse return error.InvalidIndex;
            const token = @intFromPtr(data);
            if (token == 0) return error.InvalidIndex;
            try indices.append(allocator, token - 1);
            _ = c.kd_res_next(results);
        }
        return indices.toOwnedSlice(allocator);
    }
};
