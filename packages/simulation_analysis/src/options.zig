//! Public conversion controls.

pub const WriteMode = enum {
    create,
    update,
    overwrite,
};

pub const Options = struct {
    input_path: []const u8,
    output_dir: []const u8,
    cylinder_radius: f64,
    worker_count: ?usize = null,
    target_shard_bytes: usize = 256 * 1024 * 1024,
    timestep: f64 = 1.0,
    particle_diameter: f64 = 1.122462048309373,
    shell_delta: ?f64 = null,
    neighbor_radius: ?f64 = null,
    pocket_radius: ?f64 = null,
    gaussian_cutoff_multiplier: f64 = 5.0,
    psi6_minimum: f64 = 0.7,
    misorientation_degrees: f64 = 5.0,
    minimum_cluster_particles: usize = 2,
    write_mode: WriteMode = .create,
    dry_run: bool = false,

    pub fn effectiveShellDelta(self: Options) f64 {
        return self.shell_delta orelse self.particle_diameter;
    }

    pub fn effectiveNeighborRadius(self: Options) f64 {
        return self.neighbor_radius orelse 1.7272 * self.particle_diameter;
    }

    pub fn effectivePocketRadius(self: Options) f64 {
        return self.pocket_radius orelse 2.0 * self.particle_diameter;
    }
};
