//! Linalg-backed lagged Pearson correlation.

const std = @import("std");
const linalg = @import("linalg");
const schema = @import("../schema.zig");

pub fn laggedPearson(
    allocator: std.mem.Allocator,
    context: *linalg.Context,
    velocity: []const f64,
    max_lag: usize,
) ![]f64 {
    try schema.validatePearsonInput(velocity, max_lag);

    const output = try allocator.alloc(f64, max_lag + 1);
    errdefer allocator.free(output);

    var velocity_vector = try linalg.Vector(f64).fromHost(context, velocity);
    defer velocity_vector.deinit();

    const host_ones = try allocator.alloc(f64, velocity.len);
    defer allocator.free(host_ones);
    @memset(host_ones, 1.0);
    var ones_vector = try linalg.Vector(f64).fromHost(context, host_ones);
    defer ones_vector.deinit();

    const velocity_view = velocity_vector.constView();
    const ones_view = ones_vector.constView();
    for (0..max_lag + 1) |lag| {
        const sample_count = velocity.len - lag;
        const left = linalg.ConstVectorView(f64){
            .context = velocity_view.context,
            .buffer = velocity_view.buffer,
            .offset = velocity_view.offset,
            .len = sample_count,
            .stride = velocity_view.stride,
        };
        const right = linalg.ConstVectorView(f64){
            .context = velocity_view.context,
            .buffer = velocity_view.buffer,
            .offset = velocity_view.offset + lag * velocity_view.stride,
            .len = sample_count,
            .stride = velocity_view.stride,
        };
        const ones = linalg.ConstVectorView(f64){
            .context = ones_view.context,
            .buffer = ones_view.buffer,
            .offset = ones_view.offset,
            .len = sample_count,
            .stride = ones_view.stride,
        };

        const count = @as(f64, @floatFromInt(sample_count));
        const sum_left = try linalg.dot(f64, left, ones);
        const sum_right = try linalg.dot(f64, right, ones);
        const sum_left_sq = try linalg.dot(f64, left, left);
        const sum_right_sq = try linalg.dot(f64, right, right);
        const sum_product = try linalg.dot(f64, left, right);
        const covariance = sum_product - sum_left * sum_right / count;
        const variance_left = sum_left_sq - sum_left * sum_left / count;
        const variance_right = sum_right_sq - sum_right * sum_right / count;
        if (variance_left <= 0.0 or variance_right <= 0.0) {
            return error.ConstantVelocityWindow;
        }
        const coefficient = covariance / @sqrt(variance_left * variance_right);
        if (!std.math.isFinite(coefficient)) return error.NonFiniteCorrelation;
        output[lag] = @min(1.0, @max(-1.0, coefficient));
    }
    return output;
}
