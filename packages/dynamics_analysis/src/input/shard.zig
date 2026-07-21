//! Owning memory-mapped safetensor shards and frame-schema inspection.

const std = @import("std");
const safetensors = @import("safetensors");

pub const FrameSchema = struct {
    frame_count: usize,
    particle_count: usize,
    component_count: usize,
    coordinate_dtype: safetensors.Dtype,
    step_dtype: safetensors.Dtype,
};

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
        if (coordinates.shape.len != 3) return error.InvalidCoordinateShape;
        if (steps.shape.len != 1 or steps.shape[0] != coordinates.shape[0]) {
            return error.InvalidStepShape;
        }
        if (coordinates.dtype != .f32 and coordinates.dtype != .f64) {
            return error.InvalidCoordinateDtype;
        }
        if (steps.dtype != .i32 and steps.dtype != .i64) {
            return error.InvalidStepDtype;
        }
        return .{
            .frame_count = coordinates.shape[0],
            .particle_count = coordinates.shape[1],
            .component_count = coordinates.shape[2],
            .coordinate_dtype = coordinates.dtype,
            .step_dtype = steps.dtype,
        };
    }
};

pub fn inspectShard(allocator: std.mem.Allocator, path: []const u8) !Shard {
    return Shard.open(allocator, path);
}
