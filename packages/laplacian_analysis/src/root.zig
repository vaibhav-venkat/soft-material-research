//! Complex Laplace and damped-cosine analysis of dynamics correlations.

pub const dynamics_analysis = @import("dynamics_analysis");
pub const ffi = @import("ffi/root.zig");
pub const schema = @import("schema.zig");

pub const Options = @import("options.zig").Options;
pub const TransformOptions = @import("options.zig").TransformOptions;
pub const PreferredOptions = @import("options.zig").PreferredOptions;
pub const FitOptions = @import("options.zig").FitOptions;
pub const TransformAxes = @import("result.zig").TransformAxes;
pub const PreferredAxes = @import("result.zig").PreferredAxes;
pub const LaplaceGrid = @import("result.zig").LaplaceGrid;
pub const PreferredAxis = @import("result.zig").PreferredAxis;
pub const PreferredEstimate = @import("result.zig").PreferredEstimate;
pub const DampedCosineFit = @import("result.zig").DampedCosineFit;
pub const Result = @import("result.zig").Result;

pub const transformAxes = @import("laplace.zig").transformAxes;
pub const preferredAxes = @import("laplace.zig").preferredAxes;
pub const analyzeLaplace = @import("laplace.zig").analyzeLaplace;
pub const preferredCoordinate = @import("laplace.zig").preferredCoordinate;
pub const Parameters = @import("fit.zig").Parameters;
pub const dampedCosine = @import("fit.zig").dampedCosine;
pub const dampedCosineJacobian = @import("fit.zig").dampedCosineJacobian;
pub const fitDampedCosine = @import("fit.zig").fitDampedCosine;
pub const analyze = @import("analysis.zig").analyze;

comptime {
    _ = ffi.laplacian_analysis_api_version;
    _ = ffi.laplacian_analysis_run;
    _ = ffi.laplacian_analysis_release;
}
