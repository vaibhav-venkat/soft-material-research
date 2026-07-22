//! Versioned output schema declarations and implemented option validation.

const Options = @import("options.zig").Options;
const std = @import("std");

pub const manifest_schema = "simulation_analysis.manifest.v1";
pub const static_schema = "simulation_analysis.static.v1";
pub const frame_schema = "simulation_analysis.frames.v2";
pub const coordinate_transform = "simulation_analysis.cylindrical.simd.soa.v2";

pub const FieldDtype = enum { u8, u32, u64, f32 };

pub const FieldSpec = struct {
    name: []const u8,
    dtype: FieldDtype,
    rank: u8,
    version: u32,
};

pub const base_fields = [_]FieldSpec{
    .{ .name = "frame_index", .dtype = .u64, .rank = 1, .version = 1 },
    .{ .name = "step", .dtype = .u64, .rank = 1, .version = 1 },
    .{ .name = "box", .dtype = .f32, .rank = 2, .version = 1 },
    .{ .name = "coords", .dtype = .f32, .rank = 3, .version = 2 },
};

pub const property_fields = [_]FieldSpec{
    .{ .name = "com_unwrapped", .dtype = .f32, .rank = 2, .version = 1 },
    .{ .name = "com_velocity_unwrapped", .dtype = .f32, .rank = 2, .version = 1 },
};

pub fn validateOptions(options: Options) !void {
    if (options.input_path.len == 0) return error.EmptyInputPath;
    if (options.output_dir.len == 0) return error.EmptyOutputDirectory;
    if (options.worker_count) |count| if (count == 0) return error.InvalidWorkerCount;
    if (options.target_shard_bytes == 0) return error.InvalidShardSize;
    if (!std.math.isFinite(options.timestep) or options.timestep <= 0) return error.InvalidTimestep;
}
