//! Center-of-mass velocity and lagged Pearson-correlation analysis.

const std = @import("std");

pub const tensor_io = @import("safetensors");

pub const Options = struct {
    frame_start: usize = 0,
    frame_stop: ?usize = null,
    timestep: f64 = 1.0,
    max_lag: ?usize = null,
};

/// Owned, flat arrays intended for conversion to NumPy arrays by the Python binding.
pub const Result = struct {
    elapsed_time: []f64,
    com_velocity: []f64,
    lag_time: []f64,
    pearson: []f64,

    pub fn deinit(self: Result, allocator: std.mem.Allocator) void {
        allocator.free(self.elapsed_time);
        allocator.free(self.com_velocity);
        allocator.free(self.lag_time);
        allocator.free(self.pearson);
    }
};

/// Validate the inputs through the Zig safetensors reader. Numerical work comes later.
pub fn analyze(
    allocator: std.mem.Allocator,
    safetensor_paths: []const []const u8,
    options: Options,
) !Result {
    if (safetensor_paths.len == 0) return error.NoInput;
    if (!std.math.isFinite(options.timestep) or options.timestep <= 0.0) {
        return error.InvalidOptions;
    }
    if (options.frame_stop) |stop| {
        if (stop <= options.frame_start) return error.InvalidOptions;
    }
    for (safetensor_paths) |path| {
        var reader = try tensor_io.Reader.open(allocator, path);
        reader.deinit();
    }

    const elapsed_time = try allocator.alloc(f64, 0);
    errdefer allocator.free(elapsed_time);
    const com_velocity = try allocator.alloc(f64, 0);
    errdefer allocator.free(com_velocity);
    const lag_time = try allocator.alloc(f64, 0);
    errdefer allocator.free(lag_time);
    const pearson = try allocator.alloc(f64, 0);
    return .{
        .elapsed_time = elapsed_time,
        .com_velocity = com_velocity,
        .lag_time = lag_time,
        .pearson = pearson,
    };
}

pub const COptions = extern struct {
    frame_start: usize,
    frame_stop: usize,
    has_frame_stop: bool,
    timestep: f64,
    max_lag: usize,
    has_max_lag: bool,
};

pub const F64Buffer = extern struct {
    ptr: ?[*]f64,
    len: usize,
};

pub const CResult = extern struct {
    elapsed_time: F64Buffer,
    com_velocity: F64Buffer,
    lag_time: F64Buffer,
    pearson: F64Buffer,
};

pub export fn dynamics_analysis_api_version() callconv(.c) u32 {
    return 1;
}

/// C ABI used by Python ctypes. Returns 0 on success, 1 for bad arguments,
/// 2 for allocation failure, and 3 for safetensor/file errors.
pub export fn dynamics_analysis_run(
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
        .timestep = options.timestep,
        .max_lag = if (options.has_max_lag) options.max_lag else null,
    }) catch |err| return switch (err) {
        error.NoInput, error.InvalidOptions => 1,
        error.OutOfMemory => 2,
        else => 3,
    };
    output.* = .{
        .elapsed_time = toBuffer(result.elapsed_time),
        .com_velocity = toBuffer(result.com_velocity),
        .lag_time = toBuffer(result.lag_time),
        .pearson = toBuffer(result.pearson),
    };
    return 0;
}

pub export fn dynamics_analysis_release(result: *CResult) callconv(.c) void {
    const allocator = std.heap.c_allocator;
    freeBuffer(allocator, result.elapsed_time);
    freeBuffer(allocator, result.com_velocity);
    freeBuffer(allocator, result.lag_time);
    freeBuffer(allocator, result.pearson);
    result.* = std.mem.zeroes(CResult);
}

fn toBuffer(values: []f64) F64Buffer {
    return .{ .ptr = values.ptr, .len = values.len };
}

fn freeBuffer(allocator: std.mem.Allocator, buffer: F64Buffer) void {
    if (buffer.ptr) |ptr| allocator.free(ptr[0..buffer.len]);
}
