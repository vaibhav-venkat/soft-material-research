//! Versioned output schema declarations and implemented option validation.

const Options = @import("options.zig").Options;
const std = @import("std");

pub const manifest_schema = "simulation_analysis.manifest.v1";
pub const static_schema = "simulation_analysis.static.v1";
pub const frame_schema = "simulation_analysis.frames.v3";
pub const coordinate_transform = "simulation_analysis.cylindrical.simd.soa.v2";

pub const FieldDtype = enum { u8, u32, u64, i8, f32 };

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
    .{ .name = "hexatic_order", .dtype = .f32, .rank = 2, .version = 1 },
    .{ .name = "psi_6_real", .dtype = .f32, .rank = 2, .version = 1 },
    .{ .name = "psi_6_im", .dtype = .f32, .rank = 2, .version = 1 },
    .{ .name = "disclination", .dtype = .i8, .rank = 2, .version = 1 },
    .{ .name = "hexatic_shell_mask", .dtype = .u8, .rank = 2, .version = 1 },
    .{ .name = "rho", .dtype = .f32, .rank = 2, .version = 1 },
    .{ .name = "polar_cylindrical", .dtype = .f32, .rank = 3, .version = 1 },
    .{ .name = "cluster_sizes", .dtype = .u32, .rank = 1, .version = 1 },
};

pub fn validateOptions(options: Options) !void {
    if (options.input_path.len == 0) return error.EmptyInputPath;
    if (options.output_dir.len == 0) return error.EmptyOutputDirectory;
    if (options.worker_count) |count| if (count == 0) return error.InvalidWorkerCount;
    if (options.target_shard_bytes == 0) return error.InvalidShardSize;
    if (!std.math.isFinite(options.timestep) or options.timestep <= 0) return error.InvalidTimestep;
    if (!std.math.isFinite(options.cylinder_radius) or options.cylinder_radius <= 0) return error.InvalidCylinderRadius;
    if (!std.math.isFinite(options.particle_diameter) or options.particle_diameter <= 0) return error.InvalidParticleDiameter;
    if (!std.math.isFinite(options.effectiveShellDelta()) or options.effectiveShellDelta() <= 0 or
        options.effectiveShellDelta() >= options.cylinder_radius) return error.InvalidShellDelta;
    if (!std.math.isFinite(options.effectiveNeighborRadius()) or options.effectiveNeighborRadius() <= 0) return error.InvalidNeighborRadius;
    if (!std.math.isFinite(options.effectivePocketRadius()) or options.effectivePocketRadius() <= 0) return error.InvalidPocketRadius;
    if (!std.math.isFinite(options.gaussian_cutoff_multiplier) or options.gaussian_cutoff_multiplier <= 0) return error.InvalidGaussianCutoff;
    if (!std.math.isFinite(options.psi6_minimum) or options.psi6_minimum < 0 or options.psi6_minimum > 1) return error.InvalidPsi6Threshold;
    if (!std.math.isFinite(options.misorientation_degrees) or options.misorientation_degrees < 0 or
        options.misorientation_degrees > 30) return error.InvalidMisorientation;
    if (options.minimum_cluster_particles < 2) return error.InvalidMinimumClusterSize;
}
