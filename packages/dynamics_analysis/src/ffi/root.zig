//! Stable C ABI declarations used by the Python ctypes wrapper.

const std = @import("std");

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
    _ = paths;
    _ = path_count;
    _ = options;
    output.* = std.mem.zeroes(CResult);
    return 4;
}

pub export fn dynamics_analysis_release(result: *CResult) callconv(.c) void {
    result.* = std.mem.zeroes(CResult);
}
