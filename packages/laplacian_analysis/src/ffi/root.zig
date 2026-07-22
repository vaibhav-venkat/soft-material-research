//! Stable C ABI for the Python frontend.

const std = @import("std");
const analysis = @import("../analysis.zig");
const dynamics_analysis = @import("dynamics_analysis");
const Options = @import("../options.zig").Options;

const Gpa = std.heap.DebugAllocator(.{});
const Owner = struct { gpa: Gpa };

pub const COptions = extern struct {
    frame_start: usize,
    frame_stop: usize,
    has_frame_stop: bool,
    timestep: f64,
    max_lag: usize,
    has_max_lag: bool,
    device_ordinal: u32,
    r_min: f64,
    has_r_min: bool,
    r_max: f64,
    r_points: usize,
    omega_min: f64,
    has_omega_min: bool,
    omega_max: f64,
    has_omega_max: bool,
    omega_points: usize,
    preferred_r_min: f64,
    has_preferred_r_min: bool,
    preferred_r_max: f64,
    preferred_r_points: usize,
    preferred_omega_max: f64,
    has_preferred_omega_max: bool,
    preferred_omega_points: usize,
    soft_l1_scale: f64,
    tolerance: f64,
    maximum_evaluations: usize,
    rank_tolerance: f64,
};

pub const F64Buffer = extern struct { ptr: ?[*]f64, len: usize };
pub const CPreferredEstimate = extern struct {
    axis: u8,
    coordinate: f64,
    coordinate_std: f64,
    log10_magnitude: f64,
    at_lower_boundary: bool,
    at_upper_boundary: bool,
    replicate_count: usize,
};
pub const CFit = extern struct {
    amplitude: f64,
    rate: f64,
    omega: f64,
    phase: f64,
    offset: f64,
    r_squared: f64,
    evaluations: usize,
    converged: bool,
    rate_at_lower_boundary: bool,
    rate_at_upper_boundary: bool,
    amplitude_at_upper_boundary: bool,
    prediction: F64Buffer,
};
pub const CResult = extern struct {
    r: F64Buffer,
    omega: F64Buffer,
    values_real: F64Buffer,
    values_imag: F64Buffer,
    shape: [2]usize,
    preferred_r: CPreferredEstimate,
    preferred_omega: CPreferredEstimate,
    fit: CFit,
    owner: ?*anyopaque,
};

pub export fn laplacian_analysis_api_version() callconv(.c) u32 {
    return 1;
}

pub export fn laplacian_analysis_run(
    static_path: [*:0]const u8,
    paths: [*]const [*:0]const u8,
    path_count: usize,
    options: *const COptions,
    output: *CResult,
) callconv(.c) c_int {
    output.* = std.mem.zeroes(CResult);
    if (path_count == 0) return 1;

    const owner = std.heap.page_allocator.create(Owner) catch return 2;
    owner.* = .{ .gpa = .{} };
    var keep_owner = false;
    defer if (!keep_owner) destroyOwner(owner);
    const allocator = owner.gpa.allocator();
    const path_slices = allocator.alloc([]const u8, path_count) catch return 2;
    defer allocator.free(path_slices);
    for (0..path_count) |index| path_slices[index] = std.mem.span(paths[index]);

    var context = dynamics_analysis.backend.Context.init(allocator, .{
        .device_ordinal = options.device_ordinal,
    }) catch return 3;
    defer context.deinit();
    const value = analysis.analyze(
        allocator,
        &context,
        .{
            .static = .{ .path = std.mem.span(static_path) },
            .shard_paths = path_slices,
        },
        fromCOptions(options.*),
    ) catch |err| return switch (err) {
        error.OutOfMemory => 2,
        else => 3,
    };

    output.* = .{
        .r = toBuffer(value.laplace.r),
        .omega = toBuffer(value.laplace.omega),
        .values_real = toBuffer(value.laplace.values_real),
        .values_imag = toBuffer(value.laplace.values_imag),
        .shape = value.laplace.shape,
        .preferred_r = toPreferred(value.preferred_r),
        .preferred_omega = toPreferred(value.preferred_omega),
        .fit = .{
            .amplitude = value.fit.amplitude,
            .rate = value.fit.rate,
            .omega = value.fit.omega,
            .phase = value.fit.phase,
            .offset = value.fit.offset,
            .r_squared = value.fit.r_squared,
            .evaluations = value.fit.evaluations,
            .converged = value.fit.converged,
            .rate_at_lower_boundary = value.fit.rate_at_lower_boundary,
            .rate_at_upper_boundary = value.fit.rate_at_upper_boundary,
            .amplitude_at_upper_boundary = value.fit.amplitude_at_upper_boundary,
            .prediction = toBuffer(value.fit.prediction),
        },
        .owner = owner,
    };
    keep_owner = true;
    return 0;
}

pub export fn laplacian_analysis_release(output: *CResult) callconv(.c) void {
    const opaque_owner = output.owner orelse {
        output.* = std.mem.zeroes(CResult);
        return;
    };
    const owner: *Owner = @ptrCast(@alignCast(opaque_owner));
    const allocator = owner.gpa.allocator();
    freeBuffer(allocator, output.r);
    freeBuffer(allocator, output.omega);
    freeBuffer(allocator, output.values_real);
    freeBuffer(allocator, output.values_imag);
    freeBuffer(allocator, output.fit.prediction);
    output.* = std.mem.zeroes(CResult);
    destroyOwner(owner);
}

fn fromCOptions(value: COptions) Options {
    return .{
        .dynamics = .{
            .frame_start = value.frame_start,
            .frame_stop = if (value.has_frame_stop) value.frame_stop else null,
            .timestep = value.timestep,
            .max_lag = if (value.has_max_lag) value.max_lag else null,
            .device_ordinal = value.device_ordinal,
        },
        .transform = .{
            .r_min = if (value.has_r_min) value.r_min else null,
            .r_max = value.r_max,
            .r_points = value.r_points,
            .omega_min = if (value.has_omega_min) value.omega_min else null,
            .omega_max = if (value.has_omega_max) value.omega_max else null,
            .omega_points = value.omega_points,
        },
        .preferred = .{
            .r_min = if (value.has_preferred_r_min) value.preferred_r_min else null,
            .r_max = value.preferred_r_max,
            .r_points = value.preferred_r_points,
            .omega_max = if (value.has_preferred_omega_max) value.preferred_omega_max else null,
            .omega_points = value.preferred_omega_points,
        },
        .fit = .{
            .soft_l1_scale = value.soft_l1_scale,
            .tolerance = value.tolerance,
            .maximum_evaluations = value.maximum_evaluations,
            .rank_tolerance = value.rank_tolerance,
        },
    };
}

fn toBuffer(values: []f64) F64Buffer {
    return .{ .ptr = values.ptr, .len = values.len };
}

fn toPreferred(value: @import("../result.zig").PreferredEstimate) CPreferredEstimate {
    return .{
        .axis = @intFromEnum(value.axis),
        .coordinate = value.coordinate,
        .coordinate_std = value.coordinate_std,
        .log10_magnitude = value.log10_magnitude,
        .at_lower_boundary = value.at_lower_boundary,
        .at_upper_boundary = value.at_upper_boundary,
        .replicate_count = value.replicate_count,
    };
}

fn freeBuffer(allocator: std.mem.Allocator, buffer: F64Buffer) void {
    if (buffer.ptr) |ptr| allocator.free(ptr[0..buffer.len]);
}

fn destroyOwner(owner: *Owner) void {
    _ = owner.gpa.deinit();
    std.heap.page_allocator.destroy(owner);
}
