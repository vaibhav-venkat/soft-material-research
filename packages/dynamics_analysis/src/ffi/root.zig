//! Stable C ABI declarations used by the Python ctypes wrapper.

const std = @import("std");
const ShardSet = @import("../input/root.zig").ShardSet;

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

pub export fn dynamics_analysis_run(
    paths: [*]const [*:0]const u8,
    path_count: usize,
    options: *const COptions,
    output: *CResult,
) callconv(.c) c_int {
    _ = options;
    output.* = std.mem.zeroes(CResult);
    if (path_count == 0) return 1;

    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    const path_slices = allocator.alloc([]const u8, path_count) catch return 2;
    defer allocator.free(path_slices);
    for (0..path_count) |index| {
        path_slices[index] = std.mem.span(paths[index]);
    }

    var shards = ShardSet.open(allocator, path_slices) catch |err| return switch (err) {
        error.NoInput => 1,
        error.OutOfMemory => 2,
        else => 3,
    };
    defer shards.deinit();
    return 0;
}

pub export fn dynamics_analysis_release(result: *CResult) callconv(.c) void {
    result.* = std.mem.zeroes(CResult);
}
