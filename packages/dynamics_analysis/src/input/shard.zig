//! Owning memory-mapped safetensor shards and frame-schema inspection.

const std = @import("std");
const safetensors = @import("safetensors");
const schema = @import("../schema.zig");

pub const FrameSchema = schema.FrameSchema;

/// Owns the read-only mapping behind every tensor view returned from this shard.
pub const Shard = struct {
    reader: safetensors.Reader,

    pub fn open(allocator: std.mem.Allocator, path: []const u8) !Shard {
        return .{ .reader = try safetensors.Reader.open(allocator, path) };
    }

    pub fn deinit(self: *Shard) void {
        self.reader.deinit();
    }

    pub fn tensor(self: *const Shard, name: []const u8) !safetensors.TensorView {
        return self.reader.tensor(name);
    }

    pub fn contains(self: *const Shard, name: []const u8) bool {
        return self.reader.contains(name);
    }

    pub fn keys(self: *const Shard) []const []const u8 {
        return self.reader.keys();
    }

    pub fn frameSchema(self: *const Shard) !FrameSchema {
        const coordinates = try self.tensor("coords");
        const steps = try self.tensor("step");
        return schema.inspectFrameSchema(coordinates, steps);
    }
};

pub fn inspectShard(allocator: std.mem.Allocator, path: []const u8) !Shard {
    return Shard.open(allocator, path);
}
