//! Owned analysis result declarations.

const std = @import("std");

pub const ComSeries = struct {
    elapsed_time: []f64,
    center: []f64,
    velocity: []f64,

    pub fn deinit(self: ComSeries, allocator: std.mem.Allocator) void {
        allocator.free(self.elapsed_time);
        allocator.free(self.center);
        allocator.free(self.velocity);
    }
};

pub const CorrelationSeries = struct {
    lag_indices: []usize,
    lag_times: []f64,
    pearson: []f64,
    origin_counts: []usize,

    pub fn deinit(self: CorrelationSeries, allocator: std.mem.Allocator) void {
        allocator.free(self.lag_indices);
        allocator.free(self.lag_times);
        allocator.free(self.pearson);
        allocator.free(self.origin_counts);
    }
};

pub const Result = struct {
    com: ComSeries,
    correlation: CorrelationSeries,

    pub fn deinit(self: Result, allocator: std.mem.Allocator) void {
        self.com.deinit(allocator);
        self.correlation.deinit(allocator);
    }
};
