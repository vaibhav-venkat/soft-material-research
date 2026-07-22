//! Command-line parsing for the package-local converter executable.

const std = @import("std");
const Options = @import("options.zig").Options;
const WriteMode = @import("options.zig").WriteMode;

/// Parse borrowed argument strings into conversion options.
pub fn parseArgs(allocator: std.mem.Allocator, arguments: []const []const u8) !Options {
    _ = allocator;
    var input_path: ?[]const u8 = null;
    var output_dir: ?[]const u8 = null;
    var cylinder_radius: ?f64 = null;
    var worker_count: ?usize = null;
    var target_shard_bytes: usize = 256 * 1024 * 1024;
    var timestep: f64 = 1.0e-6;
    var particle_diameter: f64 = 1.122462048309373;
    var shell_delta: ?f64 = null;
    var neighbor_radius: ?f64 = null;
    var pocket_radius: ?f64 = null;
    var gaussian_cutoff_multiplier: f64 = 5.0;
    var psi6_minimum: f64 = 0.7;
    var misorientation_degrees: f64 = 5.0;
    var minimum_cluster_particles: usize = 2;
    var write_mode: WriteMode = .create;
    var mode_set = false;
    var dry_run = false;
    var index: usize = 1;
    while (index < arguments.len) : (index += 1) {
        const argument = arguments[index];
        if (std.mem.eql(u8, argument, "--input")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            input_path = arguments[index];
        } else if (std.mem.eql(u8, argument, "--output-dir")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            output_dir = arguments[index];
        } else if (std.mem.eql(u8, argument, "--workers")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            worker_count = try std.fmt.parseInt(usize, arguments[index], 10);
        } else if (std.mem.eql(u8, argument, "--cylinder-radius")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            cylinder_radius = try std.fmt.parseFloat(f64, arguments[index]);
        } else if (std.mem.eql(u8, argument, "--particle-diameter")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            particle_diameter = try std.fmt.parseFloat(f64, arguments[index]);
        } else if (std.mem.eql(u8, argument, "--shell-delta")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            shell_delta = try std.fmt.parseFloat(f64, arguments[index]);
        } else if (std.mem.eql(u8, argument, "--neighbor-radius")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            neighbor_radius = try std.fmt.parseFloat(f64, arguments[index]);
        } else if (std.mem.eql(u8, argument, "--pocket-radius")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            pocket_radius = try std.fmt.parseFloat(f64, arguments[index]);
        } else if (std.mem.eql(u8, argument, "--gaussian-cutoff-multiplier")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            gaussian_cutoff_multiplier = try std.fmt.parseFloat(f64, arguments[index]);
        } else if (std.mem.eql(u8, argument, "--psi6-minimum")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            psi6_minimum = try std.fmt.parseFloat(f64, arguments[index]);
        } else if (std.mem.eql(u8, argument, "--misorientation-degrees")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            misorientation_degrees = try std.fmt.parseFloat(f64, arguments[index]);
        } else if (std.mem.eql(u8, argument, "--minimum-cluster-particles")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            minimum_cluster_particles = try std.fmt.parseInt(usize, arguments[index], 10);
        } else if (std.mem.eql(u8, argument, "--target-shard-mib")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            const mib = try std.fmt.parseInt(usize, arguments[index], 10);
            target_shard_bytes = std.math.mul(usize, mib, 1024 * 1024) catch return error.SizeOverflow;
        } else if (std.mem.eql(u8, argument, "--timestep")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            timestep = try std.fmt.parseFloat(f64, arguments[index]);
        } else if (std.mem.eql(u8, argument, "--update")) {
            if (mode_set) return error.ConflictingWriteModes;
            write_mode = .update;
            mode_set = true;
        } else if (std.mem.eql(u8, argument, "--overwrite")) {
            if (mode_set) return error.ConflictingWriteModes;
            write_mode = .overwrite;
            mode_set = true;
        } else if (std.mem.eql(u8, argument, "--dry-run")) {
            dry_run = true;
        } else {
            return error.UnknownArgument;
        }
    }
    return .{
        .input_path = input_path orelse return error.MissingInput,
        .output_dir = output_dir orelse return error.MissingOutputDirectory,
        .cylinder_radius = cylinder_radius orelse return error.MissingCylinderRadius,
        .worker_count = worker_count,
        .target_shard_bytes = target_shard_bytes,
        .timestep = timestep,
        .particle_diameter = particle_diameter,
        .shell_delta = shell_delta,
        .neighbor_radius = neighbor_radius,
        .pocket_radius = pocket_radius,
        .gaussian_cutoff_multiplier = gaussian_cutoff_multiplier,
        .psi6_minimum = psi6_minimum,
        .misorientation_degrees = misorientation_degrees,
        .minimum_cluster_particles = minimum_cluster_particles,
        .write_mode = write_mode,
        .dry_run = dry_run,
    };
}
