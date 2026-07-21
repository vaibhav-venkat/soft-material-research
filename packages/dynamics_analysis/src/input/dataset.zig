//! Dataset descriptors and an owning collection of mapped shards.

const std = @import("std");
const Shard = @import("shard.zig").Shard;

pub const StaticInput = struct {
    path: []const u8,
};

pub const DatasetInput = struct {
    static: StaticInput,
    shard_paths: []const []const u8,
};

/// Keeps every shard mapping alive for the duration of a synchronous analysis.
pub const ShardSet = struct {
    allocator: std.mem.Allocator,
    shards: []Shard,

    pub fn open(allocator: std.mem.Allocator, paths: []const []const u8) !ShardSet {
        if (paths.len == 0) return error.NoInput;
        const shards = try allocator.alloc(Shard, paths.len);
        errdefer allocator.free(shards);

        var initialized: usize = 0;
        errdefer {
            for (shards[0..initialized]) |*shard| shard.deinit();
        }
        for (paths, 0..) |path, index| {
            shards[index] = try Shard.open(allocator, path);
            initialized += 1;
        }
        return .{ .allocator = allocator, .shards = shards };
    }

    pub fn deinit(self: *ShardSet) void {
        for (self.shards) |*shard| shard.deinit();
        self.allocator.free(self.shards);
        self.* = undefined;
    }
};
