//! Structural and coherent-motion cluster identification.

const std = @import("std");

pub const tensor_io = @import("safetensors");

pub const Options = struct {
    frame_start: usize = 0,
    frame_stop: ?usize = null,
    lag_frames: usize = 1,
    minimum_particles: usize = 2,
    psi6_minimum: f64 = 0.7,
    motion_cosine_minimum: f64 = 0.8,
};

/// Cluster membership in compressed sparse row form. Each row is one cluster.
pub const Membership = struct {
    offsets: []usize,
    particle_indices: []usize,
    frame_indices: []usize,

    pub fn deinit(self: Membership, allocator: std.mem.Allocator) void {
        allocator.free(self.offsets);
        allocator.free(self.particle_indices);
        allocator.free(self.frame_indices);
    }
};

pub const Result = struct {
    structural: Membership,
    motion: Membership,

    pub fn deinit(self: Result, allocator: std.mem.Allocator) void {
        self.structural.deinit(allocator);
        self.motion.deinit(allocator);
    }
};

/// Validate the inputs through the Zig safetensors reader. Numerical work comes later.
pub fn analyze(
    allocator: std.mem.Allocator,
    safetensor_paths: []const []const u8,
    options: Options,
) !Result {
    if (safetensor_paths.len == 0) return error.NoInput;
    if (options.lag_frames == 0 or options.minimum_particles < 2) return error.InvalidOptions;
    if (!std.math.isFinite(options.psi6_minimum) or options.psi6_minimum < 0.0 or options.psi6_minimum > 1.0) {
        return error.InvalidOptions;
    }
    if (!std.math.isFinite(options.motion_cosine_minimum) or options.motion_cosine_minimum < -1.0 or options.motion_cosine_minimum > 1.0) {
        return error.InvalidOptions;
    }
    if (options.frame_stop) |stop| {
        if (stop <= options.frame_start) return error.InvalidOptions;
    }
    for (safetensor_paths) |path| {
        var reader = try tensor_io.Reader.open(allocator, path);
        reader.deinit();
    }

    const structural = try emptyMembership(allocator);
    errdefer structural.deinit(allocator);
    const motion = try emptyMembership(allocator);
    return .{ .structural = structural, .motion = motion };
}

fn emptyMembership(allocator: std.mem.Allocator) !Membership {
    const offsets = try allocator.alloc(usize, 1);
    errdefer allocator.free(offsets);
    offsets[0] = 0;
    const particle_indices = try allocator.alloc(usize, 0);
    errdefer allocator.free(particle_indices);
    const frame_indices = try allocator.alloc(usize, 0);
    return .{
        .offsets = offsets,
        .particle_indices = particle_indices,
        .frame_indices = frame_indices,
    };
}

pub const COptions = extern struct {
    frame_start: usize,
    frame_stop: usize,
    has_frame_stop: bool,
    lag_frames: usize,
    minimum_particles: usize,
    psi6_minimum: f64,
    motion_cosine_minimum: f64,
};

pub const UsizeBuffer = extern struct { ptr: ?[*]usize, len: usize };
pub const CMembership = extern struct {
    offsets: UsizeBuffer,
    particle_indices: UsizeBuffer,
    frame_indices: UsizeBuffer,
};
pub const CResult = extern struct {
    structural: CMembership,
    motion: CMembership,
};

pub export fn cluster_analysis_api_version() callconv(.c) u32 {
    return 1;
}

pub export fn cluster_analysis_run(
    paths: [*]const [*:0]const u8,
    path_count: usize,
    options: *const COptions,
    output: *CResult,
) callconv(.c) c_int {
    output.* = std.mem.zeroes(CResult);
    const allocator = std.heap.c_allocator;
    const path_slices = allocator.alloc([]const u8, path_count) catch return 2;
    defer allocator.free(path_slices);
    for (0..path_count) |index| path_slices[index] = std.mem.span(paths[index]);

    const result = analyze(allocator, path_slices, .{
        .frame_start = options.frame_start,
        .frame_stop = if (options.has_frame_stop) options.frame_stop else null,
        .lag_frames = options.lag_frames,
        .minimum_particles = options.minimum_particles,
        .psi6_minimum = options.psi6_minimum,
        .motion_cosine_minimum = options.motion_cosine_minimum,
    }) catch |err| return switch (err) {
        error.NoInput, error.InvalidOptions => 1,
        error.OutOfMemory => 2,
        else => 3,
    };
    output.* = .{
        .structural = toCMembership(result.structural),
        .motion = toCMembership(result.motion),
    };
    return 0;
}

pub export fn cluster_analysis_release(result: *CResult) callconv(.c) void {
    const allocator = std.heap.c_allocator;
    freeCMembership(allocator, result.structural);
    freeCMembership(allocator, result.motion);
    result.* = std.mem.zeroes(CResult);
}

fn toCMembership(value: Membership) CMembership {
    return .{
        .offsets = .{ .ptr = value.offsets.ptr, .len = value.offsets.len },
        .particle_indices = .{ .ptr = value.particle_indices.ptr, .len = value.particle_indices.len },
        .frame_indices = .{ .ptr = value.frame_indices.ptr, .len = value.frame_indices.len },
    };
}

fn freeCMembership(allocator: std.mem.Allocator, membership: CMembership) void {
    freeBuffer(allocator, membership.offsets);
    freeBuffer(allocator, membership.particle_indices);
    freeBuffer(allocator, membership.frame_indices);
}

fn freeBuffer(allocator: std.mem.Allocator, buffer: UsizeBuffer) void {
    if (buffer.ptr) |ptr| allocator.free(ptr[0..buffer.len]);
}
