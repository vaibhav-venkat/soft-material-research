//! Stable C ABI used by the Python ctypes wrapper.

const std = @import("std");
const analysis = @import("../analysis.zig");
const backend = @import("../backend/root.zig");
const input = @import("../input/root.zig");

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
};

pub const F64Buffer = extern struct { ptr: ?[*]f64, len: usize };
pub const UsizeBuffer = extern struct { ptr: ?[*]usize, len: usize };

pub const CResult = extern struct {
    elapsed_time: F64Buffer,
    center: F64Buffer,
    velocity: F64Buffer,
    lag_indices: UsizeBuffer,
    lag_times: F64Buffer,
    pearson: F64Buffer,
    origin_counts: UsizeBuffer,
    owner: ?*anyopaque,
};

pub export fn dynamics_analysis_api_version() callconv(.c) u32 {
    return 2;
}

pub export fn dynamics_analysis_run(
    static_path: [*:0]const u8,
    paths: [*]const [*:0]const u8,
    path_count: usize,
    options: *const COptions,
    output: *CResult,
) callconv(.c) c_int {
    output.* = std.mem.zeroes(CResult);
    if (path_count == 0 or !std.math.isFinite(options.timestep) or options.timestep <= 0.0) {
        return 1;
    }

    const owner = std.heap.page_allocator.create(Owner) catch return 2;
    owner.* = .{ .gpa = .{} };
    var keep_owner = false;
    defer if (!keep_owner) destroyOwner(owner);
    const allocator = owner.gpa.allocator();

    const path_slices = allocator.alloc([]const u8, path_count) catch return 2;
    defer allocator.free(path_slices);
    for (0..path_count) |index| path_slices[index] = std.mem.span(paths[index]);

    var context = backend.Context.init(allocator, .{
        .device_ordinal = options.device_ordinal,
    }) catch return 3;
    defer context.deinit();
    const result = analysis.analyze(
        allocator,
        &context,
        .{
            .static = .{ .path = std.mem.span(static_path) },
            .shard_paths = path_slices,
        },
        .{
            .frame_start = options.frame_start,
            .frame_stop = if (options.has_frame_stop) options.frame_stop else null,
            .timestep = options.timestep,
            .max_lag = if (options.has_max_lag) options.max_lag else null,
            .device_ordinal = options.device_ordinal,
        },
    ) catch |err| return switch (err) {
        error.OutOfMemory => 2,
        else => 3,
    };

    output.* = .{
        .elapsed_time = f64Buffer(result.com.elapsed_time),
        .center = f64Buffer(result.com.center),
        .velocity = f64Buffer(result.com.velocity),
        .lag_indices = usizeBuffer(result.correlation.lag_indices),
        .lag_times = f64Buffer(result.correlation.lag_times),
        .pearson = f64Buffer(result.correlation.pearson),
        .origin_counts = usizeBuffer(result.correlation.origin_counts),
        .owner = owner,
    };
    keep_owner = true;
    return 0;
}

pub export fn dynamics_analysis_release(output: *CResult) callconv(.c) void {
    const opaque_owner = output.owner orelse {
        output.* = std.mem.zeroes(CResult);
        return;
    };
    const owner: *Owner = @ptrCast(@alignCast(opaque_owner));
    const allocator = owner.gpa.allocator();
    freeF64(allocator, output.elapsed_time);
    freeF64(allocator, output.center);
    freeF64(allocator, output.velocity);
    freeUsize(allocator, output.lag_indices);
    freeF64(allocator, output.lag_times);
    freeF64(allocator, output.pearson);
    freeUsize(allocator, output.origin_counts);
    output.* = std.mem.zeroes(CResult);
    destroyOwner(owner);
}

fn destroyOwner(owner: *Owner) void {
    _ = owner.gpa.deinit();
    std.heap.page_allocator.destroy(owner);
}

fn f64Buffer(values: []f64) F64Buffer {
    return .{ .ptr = values.ptr, .len = values.len };
}

fn usizeBuffer(values: []usize) UsizeBuffer {
    return .{ .ptr = values.ptr, .len = values.len };
}

fn freeF64(allocator: std.mem.Allocator, buffer: F64Buffer) void {
    if (buffer.ptr) |ptr| allocator.free(ptr[0..buffer.len]);
}

fn freeUsize(allocator: std.mem.Allocator, buffer: UsizeBuffer) void {
    if (buffer.ptr) |ptr| allocator.free(ptr[0..buffer.len]);
}
