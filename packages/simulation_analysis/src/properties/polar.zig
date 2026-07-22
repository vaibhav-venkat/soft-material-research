//! Gaussian coarse-grained orientation density in the cylindrical frame.

const std = @import("std");
const data_structures = @import("../data_structures/root.zig");
const coordinates = data_structures.coordinates;
const common = @import("math.zig");

const CellList3 = data_structures.CellList(3);

pub const Options = struct {
    pocket_radius: f32,
    cutoff_multiplier: f32 = 5.0,
};

pub fn calculate(
    allocator: std.mem.Allocator,
    cells: *CellList3,
    orientations: coordinates.Quaternions.Slice,
    theta: []const f32,
    options: Options,
    rho: []f32,
    polar: coordinates.CylindricalVectors.Slice,
) !void {
    const particle_count = cells.next.len;
    if (!std.math.isFinite(options.pocket_radius) or options.pocket_radius <= 0 or
        !std.math.isFinite(options.cutoff_multiplier) or options.cutoff_multiplier <= 0)
        return error.InvalidPolarOptions;
    if (orientations.len != particle_count or theta.len != particle_count or
        rho.len != particle_count or polar.len != particle_count) return error.DimensionMismatch;
    var directions: coordinates.CartesianVectors = .empty;
    defer directions.deinit(allocator);
    try directions.resize(allocator, particle_count);
    var direction_slices = directions.slice();
    for (0..particle_count) |particle| {
        direction_slices.set(particle, try common.activeDirection(orientations.get(particle)));
    }
    var neighbors: std.ArrayList(CellList3.Neighbor) = .empty;
    defer neighbors.deinit(allocator);
    const cutoff = options.pocket_radius * options.cutoff_multiplier;
    for (0..particle_count) |particle| {
        try cells.queryRadius(particle, cutoff, false, null, &neighbors);
        var density: f32 = 0;
        var vector = coordinates.CartesianVector{ .x = 0, .y = 0, .z = 0 };
        for (neighbors.items) |neighbor| {
            const weight = try common.gaussianWeight(neighbor.distance_squared, options.pocket_radius);
            const direction = direction_slices.get(neighbor.index);
            density += weight;
            vector.x += weight * direction.x;
            vector.y += weight * direction.y;
            vector.z += weight * direction.z;
        }
        rho[particle] = density;
        const cylindrical = common.cylindricalVector(vector, theta[particle]);
        polar.items(.x)[particle] = cylindrical.x;
        polar.items(.r)[particle] = cylindrical.r;
        polar.items(.theta)[particle] = cylindrical.theta;
    }
}
