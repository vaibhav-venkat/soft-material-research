//! Public controls for structural crystal clustering.

pub const Options = struct {
    frame_start: usize = 0,
    frame_stop: ?usize = null,
    psi6_minimum: f64 = 0.7,
    misorientation_degrees: f64 = 5.0,
    neighbor_radius_diameters: f64 = 1.7272,
    minimum_particles: usize = 2,
    particle_diameter: f64 = 1.122462048309373,
};
