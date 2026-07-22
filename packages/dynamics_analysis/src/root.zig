//! Center-of-mass velocity and lagged Pearson-correlation analysis.

pub const backend = @import("backend/root.zig");
pub const dynamics = @import("dynamics/root.zig");
pub const ffi = @import("ffi/root.zig");
pub const input = @import("input/root.zig");
pub const schema = @import("schema.zig");

pub const Options = @import("options.zig").Options;
pub const ComSeries = @import("result.zig").ComSeries;
pub const CorrelationSeries = @import("result.zig").CorrelationSeries;
pub const Result = @import("result.zig").Result;
pub const analyze = @import("analysis.zig").analyze;

comptime {
    _ = ffi.dynamics_analysis_api_version;
    _ = ffi.dynamics_analysis_run;
    _ = ffi.dynamics_analysis_release;
}
