//! Controls corresponding to the Rust transform, preferred, and fit configs.

const dynamics_analysis = @import("dynamics_analysis");

pub const TransformOptions = struct {
    r_min: ?f64 = null,
    r_max: f64 = 0.0,
    r_points: usize = 161,
    omega_min: ?f64 = null,
    omega_max: ?f64 = null,
    omega_points: usize = 241,
};

pub const PreferredOptions = struct {
    r_min: ?f64 = null,
    r_max: f64 = 0.0,
    r_points: usize = 241,
    omega_max: ?f64 = null,
    omega_points: usize = 241,
};

pub const FitOptions = struct {
    soft_l1_scale: f64 = 0.05,
    tolerance: f64 = 1.0e-8,
    maximum_evaluations: usize = 20_000,
    rank_tolerance: f64 = 1.0e-12,
};

pub const Options = struct {
    dynamics: dynamics_analysis.Options = .{},
    transform: TransformOptions = .{},
    preferred: PreferredOptions = .{},
    fit: FitOptions = .{},
};
