//! Owning memory maps for the static file and frame shards.

const std = @import("std");
const safetensors = @import("safetensors");

pub const StaticInput = struct { path: []const u8 };

pub const DatasetInput = struct {
    static: StaticInput,
    shard_paths: []const []const u8,
};

pub const Shard = struct {
    reader: safetensors.Reader,

    pub fn open(allocator: std.mem.Allocator, path: []const u8) !Shard {
        return .{ .reader = try safetensors.Reader.open(allocator, path) };
    }

    pub fn deinit(self: *Shard) void {
        self.reader.deinit();
    }
};

/// Keeps all mappings alive while the synchronous analysis borrows tensor data.
pub const Dataset = struct {
    allocator: std.mem.Allocator,
    static: safetensors.Reader,
    shards: []Shard,

    pub fn open(allocator: std.mem.Allocator, input: DatasetInput) !Dataset {
        if (input.shard_paths.len == 0) return error.NoInput;
        var static = try safetensors.Reader.open(allocator, input.static.path);
        errdefer static.deinit();
        const shards = try allocator.alloc(Shard, input.shard_paths.len);
        errdefer allocator.free(shards);
        var initialized: usize = 0;
        errdefer for (shards[0..initialized]) |*shard| shard.deinit();
        for (input.shard_paths, 0..) |path, index| {
            shards[index] = try Shard.open(allocator, path);
            initialized += 1;
        }
        return .{ .allocator = allocator, .static = static, .shards = shards };
    }

    pub fn deinit(self: *Dataset) void {
        self.static.deinit();
        for (self.shards) |*shard| shard.deinit();
        self.allocator.free(self.shards);
        self.* = undefined;
    }
};
