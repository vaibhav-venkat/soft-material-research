//! Stable C ABI for the Python structural-cluster frontend.

const std = @import("std");
const analysis = @import("../analysis.zig");
const input = @import("../input/root.zig");
const options_module = @import("../options.zig");

const Gpa = std.heap.DebugAllocator(.{});
const Owner = struct { gpa: Gpa };

pub const COptions = extern struct {
    frame_start: usize,
    frame_stop: usize,
    has_frame_stop: bool,
    psi6_minimum: f64,
    misorientation_degrees: f64,
    neighbor_radius_diameters: f64,
    minimum_particles: usize,
    particle_diameter: f64,
};

pub const F64Buffer = extern struct { ptr: ?[*]f64, len: usize };
pub const UsizeBuffer = extern struct { ptr: ?[*]usize, len: usize };
pub const CResult = extern struct {
    points: F64Buffer,
    offsets: UsizeBuffer,
    owner: ?*anyopaque,
};

pub export fn cluster_analysis_api_version() callconv(.c) u32 {
    return 3;
}

pub export fn cluster_analysis_run(
    static_path: [*:0]const u8,
    paths: [*]const [*:0]const u8,
    path_count: usize,
    c_options: *const COptions,
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

    const value = analysis.analyze(
        allocator,
        .{
            .static = .{ .path = std.mem.span(static_path) },
            .shard_paths = path_slices,
        },
        fromCOptions(c_options.*),
    ) catch |err| return switch (err) {
        error.OutOfMemory => 2,
        else => 3,
    };
    output.* = .{
        .points = .{
            .ptr = if (value.points.len == 0) null else @ptrCast(value.points.ptr),
            .len = value.points.len * 2,
        },
        .offsets = .{ .ptr = value.offsets.ptr, .len = value.offsets.len },
        .owner = owner,
    };
    keep_owner = true;
    return 0;
}

pub export fn cluster_analysis_release(output: *CResult) callconv(.c) void {
    const opaque_owner = output.owner orelse {
        output.* = std.mem.zeroes(CResult);
        return;
    };
    const owner: *Owner = @ptrCast(@alignCast(opaque_owner));
    if (output.points.ptr) |ptr| {
        const point_ptr: [*][2]f64 = @ptrCast(@alignCast(ptr));
        owner.gpa.allocator().free(point_ptr[0 .. output.points.len / 2]);
    }
    if (output.offsets.ptr) |ptr| {
        owner.gpa.allocator().free(ptr[0..output.offsets.len]);
    }
    output.* = std.mem.zeroes(CResult);
    destroyOwner(owner);
}

fn fromCOptions(value: COptions) options_module.Options {
    return .{
        .frame_start = value.frame_start,
        .frame_stop = if (value.has_frame_stop) value.frame_stop else null,
        .psi6_minimum = value.psi6_minimum,
        .misorientation_degrees = value.misorientation_degrees,
        .neighbor_radius_diameters = value.neighbor_radius_diameters,
        .minimum_particles = value.minimum_particles,
        .particle_diameter = value.particle_diameter,
    };
}

fn destroyOwner(owner: *Owner) void {
    _ = owner.gpa.deinit();
    std.heap.page_allocator.destroy(owner);
}
