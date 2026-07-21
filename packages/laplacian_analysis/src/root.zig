//! Laplace-transform, preferred-value, and damped-cosine-fit analysis.

const std = @import("std");

pub const tensor_io = @import("safetensors");

pub const Options = struct {
    r_points: usize = 128,
    omega_points: usize = 256,
    r_min: ?f64 = null,
    r_max: f64 = 0.0,
    omega_min: ?f64 = null,
    omega_max: ?f64 = null,
};

pub const DampedCosineFit = struct {
    amplitude: f64,
    rate: f64,
    omega: f64,
    phase: f64,
    offset: f64,
    r_squared: f64,
};

/// Complex values use parallel real and imaginary arrays with shape
/// `(omega.len, r.len)` in row-major order.
pub const Result = struct {
    r: []f64,
    omega: []f64,
    values_real: []f64,
    values_imag: []f64,
    preferred_r: f64,
    preferred_omega: f64,
    fit: DampedCosineFit,

    pub fn deinit(self: Result, allocator: std.mem.Allocator) void {
        allocator.free(self.r);
        allocator.free(self.omega);
        allocator.free(self.values_real);
        allocator.free(self.values_imag);
    }
};

/// Validate the inputs through the Zig safetensors reader. Numerical work comes later.
pub fn analyze(
    allocator: std.mem.Allocator,
    safetensor_paths: []const []const u8,
    options: Options,
) !Result {
    if (safetensor_paths.len == 0) return error.NoInput;
    if (options.r_points < 2 or options.omega_points < 2) return error.InvalidOptions;
    if (!std.math.isFinite(options.r_max)) return error.InvalidOptions;
    for (safetensor_paths) |path| {
        var reader = try tensor_io.Reader.open(allocator, path);
        reader.deinit();
    }

    const r = try allocator.alloc(f64, 0);
    errdefer allocator.free(r);
    const omega = try allocator.alloc(f64, 0);
    errdefer allocator.free(omega);
    const values_real = try allocator.alloc(f64, 0);
    errdefer allocator.free(values_real);
    const values_imag = try allocator.alloc(f64, 0);
    const nan = std.math.nan(f64);
    return .{
        .r = r,
        .omega = omega,
        .values_real = values_real,
        .values_imag = values_imag,
        .preferred_r = nan,
        .preferred_omega = nan,
        .fit = .{
            .amplitude = nan,
            .rate = nan,
            .omega = nan,
            .phase = nan,
            .offset = nan,
            .r_squared = nan,
        },
    };
}

pub const COptions = extern struct {
    r_points: usize,
    omega_points: usize,
    r_min: f64,
    has_r_min: bool,
    r_max: f64,
    omega_min: f64,
    has_omega_min: bool,
    omega_max: f64,
    has_omega_max: bool,
};

pub const F64Buffer = extern struct { ptr: ?[*]f64, len: usize };
pub const CFit = extern struct {
    amplitude: f64,
    rate: f64,
    omega: f64,
    phase: f64,
    offset: f64,
    r_squared: f64,
};
pub const CResult = extern struct {
    r: F64Buffer,
    omega: F64Buffer,
    values_real: F64Buffer,
    values_imag: F64Buffer,
    preferred_r: f64,
    preferred_omega: f64,
    fit: CFit,
};

pub export fn laplacian_analysis_api_version() callconv(.c) u32 {
    return 1;
}

pub export fn laplacian_analysis_run(
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
        .r_points = options.r_points,
        .omega_points = options.omega_points,
        .r_min = if (options.has_r_min) options.r_min else null,
        .r_max = options.r_max,
        .omega_min = if (options.has_omega_min) options.omega_min else null,
        .omega_max = if (options.has_omega_max) options.omega_max else null,
    }) catch |err| return switch (err) {
        error.NoInput, error.InvalidOptions => 1,
        error.OutOfMemory => 2,
        else => 3,
    };
    output.* = .{
        .r = toBuffer(result.r),
        .omega = toBuffer(result.omega),
        .values_real = toBuffer(result.values_real),
        .values_imag = toBuffer(result.values_imag),
        .preferred_r = result.preferred_r,
        .preferred_omega = result.preferred_omega,
        .fit = .{
            .amplitude = result.fit.amplitude,
            .rate = result.fit.rate,
            .omega = result.fit.omega,
            .phase = result.fit.phase,
            .offset = result.fit.offset,
            .r_squared = result.fit.r_squared,
        },
    };
    return 0;
}

pub export fn laplacian_analysis_release(result: *CResult) callconv(.c) void {
    const allocator = std.heap.c_allocator;
    freeBuffer(allocator, result.r);
    freeBuffer(allocator, result.omega);
    freeBuffer(allocator, result.values_real);
    freeBuffer(allocator, result.values_imag);
    result.* = std.mem.zeroes(CResult);
}

fn toBuffer(values: []f64) F64Buffer {
    return .{ .ptr = values.ptr, .len = values.len };
}

fn freeBuffer(allocator: std.mem.Allocator, buffer: F64Buffer) void {
    if (buffer.ptr) |ptr| allocator.free(ptr[0..buffer.len]);
}
