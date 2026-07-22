//! Lagged Pearson-correlation series assembly.

const std = @import("std");
const backend = @import("../backend/root.zig");
const schema = @import("../schema.zig");
const Options = @import("../options.zig").Options;
const result = @import("../result.zig");

pub fn analyzeCorrelation(
    allocator: std.mem.Allocator,
    context: *backend.Context,
    com_result: result.ComSeries,
    options: Options,
) !result.CorrelationSeries {
    const frame_count = com_result.elapsed_time.len;
    const spacing = try schema.uniformComSpacing(com_result.elapsed_time, com_result.velocity.len);

    const available_max_lag = frame_count - 2;
    const max_lag = @min(options.max_lag orelse available_max_lag, available_max_lag);
    const pearson = try backend.laggedPearson(
        allocator,
        context,
        com_result.velocity,
        max_lag,
    );
    errdefer allocator.free(pearson);

    const lag_indices = try allocator.alloc(usize, max_lag + 1);
    errdefer allocator.free(lag_indices);
    const lag_times = try allocator.alloc(f64, max_lag + 1);
    errdefer allocator.free(lag_times);
    const origin_counts = try allocator.alloc(usize, max_lag + 1);
    errdefer allocator.free(origin_counts);
    for (0..max_lag + 1) |lag| {
        lag_indices[lag] = lag;
        lag_times[lag] = @as(f64, @floatFromInt(lag)) * spacing;
        origin_counts[lag] = frame_count - lag;
    }
    return .{
        .lag_indices = lag_indices,
        .lag_times = lag_times,
        .pearson = pearson,
        .origin_counts = origin_counts,
    };
}
