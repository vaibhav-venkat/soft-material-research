//! Public controls for structural crystal clustering.

/// `sqrt_area_fraction` matches the Rust equivalent-circumference ratio.
pub const RatioMode = enum(u8) {
    area_fraction,
    sqrt_area_fraction,
};

pub const Options = struct {
    frame_start: usize = 0,
    frame_stop: ?usize = null,
    psi6_minimum: f64 = 0.7,
    misorientation_degrees: f64 = 5.0,
    neighbor_radius_diameters: f64 = 1.7272,
    minimum_particles: usize = 2,
    particle_diameter: f64 = 1.0,
    ratio_mode: RatioMode = .sqrt_area_fraction,
};
