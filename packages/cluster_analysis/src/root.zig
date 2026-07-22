//! Structural crystal-cluster analysis for cylindrical surfaces.

pub const backend = @import("backend/root.zig");
pub const clusters = @import("clusters.zig");
pub const ffi = @import("ffi/root.zig");
pub const input = @import("input/root.zig");
pub const schema = @import("schema.zig");

pub const Options = @import("options.zig").Options;
pub const RatioMode = @import("options.zig").RatioMode;
pub const Result = @import("result.zig").Result;
pub const StructuralFrame = @import("schema.zig").StructuralFrame;
pub const Workspace = @import("clusters.zig").Workspace;
pub const analyzeStructuralFrame = @import("clusters.zig").analyzeStructuralFrame;
pub const analyzeStructuralFrameWithWorkspace = @import("clusters.zig").analyzeStructuralFrameWithWorkspace;
pub const appendStructuralFrameRatios = @import("clusters.zig").appendStructuralFrameRatios;
pub const analyze = @import("analysis.zig").analyze;

comptime {
    _ = ffi.cluster_analysis_api_version;
    _ = ffi.cluster_analysis_run;
    _ = ffi.cluster_analysis_release;
}
