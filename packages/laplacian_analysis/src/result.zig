//! Owned numerical results matching the Rust analysis model.

const std = @import("std");

pub const TransformAxes = struct {
    r: []f64,
    omega: []f64,

    pub fn deinit(self: TransformAxes, allocator: std.mem.Allocator) void {
        allocator.free(self.r);
        allocator.free(self.omega);
    }
};


/// Complex values are parallel arrays in `(omega, r)` row-major order.
pub const LaplaceGrid = struct {
    r: []f64,
    omega: []f64,
    values_real: []f64,
    values_imag: []f64,
    shape: [2]usize,

    pub fn deinit(self: LaplaceGrid, allocator: std.mem.Allocator) void {
        allocator.free(self.r);
        allocator.free(self.omega);
        allocator.free(self.values_real);
        allocator.free(self.values_imag);
    }
};

pub const PreferredAxis = enum(u8) { omega, r };

pub const PreferredEstimate = struct {
    axis: PreferredAxis,
    coordinate: f64,
    coordinate_std: f64,
    log10_magnitude: f64,
    at_lower_boundary: bool,
    at_upper_boundary: bool,
    replicate_count: usize,
};

pub const DampedCosineFit = struct {
    amplitude: f64,
    rate: f64,
    omega: f64,
    phase: f64,
    offset: f64,
    r_squared: f64,
    evaluations: usize,
    converged: bool,
    rate_at_lower_boundary: bool,
    rate_at_upper_boundary: bool,
    amplitude_at_upper_boundary: bool,
    prediction: []f64,

    pub fn deinit(self: DampedCosineFit, allocator: std.mem.Allocator) void {
        allocator.free(self.prediction);
    }
};

pub const Result = struct {
    laplace: LaplaceGrid,
    preferred_r: PreferredEstimate,
    preferred_omega: PreferredEstimate,
    fit: DampedCosineFit,

    pub fn deinit(self: Result, allocator: std.mem.Allocator) void {
        self.laplace.deinit(allocator);
        self.fit.deinit(allocator);
    }
};
