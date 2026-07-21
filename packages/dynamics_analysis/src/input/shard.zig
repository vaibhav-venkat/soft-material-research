//! Frame-shard and safetensor-schema declarations.

const std = @import("std");
const safetensors = @import("safetensors");

pub const FrameSchema = struct {
    frame_count: usize,
    particle_count: usize,
    component_count: usize,
    coordinate_dtype: safetensors.Dtype,
    step_dtype: safetensors.Dtype,
};

pub const Shard = struct {
    path: []const u8,
    schema: FrameSchema,
};

pub fn inspectShard(
    allocator: std.mem.Allocator,
    path: []const u8,
) error{NotImplemented}!Shard {
    _ = allocator;
    _ = path;
    return error.NotImplemented;
}
