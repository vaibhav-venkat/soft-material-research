//! Shell hexatic order and coordination charge.

const std = @import("std");
const data_structures = @import("../data_structures/root.zig");
const coordinates = data_structures.coordinates;

const CellList3 = data_structures.CellList(3);

pub const Options = struct {
    cylinder_radius: f32,
    shell_delta: f32,
    coordination_radius: f32,
    neighbor_count: usize = 6,
};

pub const Output = struct {
    order: []f32,
    real: []f32,
    imaginary: []f32,
    disclination: []i8,
    shell_mask: []u8,
};

pub fn calculate(
    allocator: std.mem.Allocator,
    cells: *CellList3,
    positions: coordinates.CartesianPositions.Slice,
    options: Options,
    output: Output,
) !void {
    const particle_count = positions.len;
    if (!std.math.isFinite(options.cylinder_radius) or options.cylinder_radius <= 0 or
        !std.math.isFinite(options.shell_delta) or options.shell_delta <= 0 or
        !std.math.isFinite(options.coordination_radius) or options.coordination_radius <= 0 or
        options.neighbor_count == 0) return error.InvalidHexaticOptions;
    if (cells.next.len != particle_count or output.order.len != particle_count or
        output.real.len != particle_count or output.imaginary.len != particle_count or
        output.disclination.len != particle_count or output.shell_mask.len != particle_count)
        return error.DimensionMismatch;
    @memset(output.order, 0);
    @memset(output.real, 0);
    @memset(output.imaginary, 0);
    @memset(output.disclination, 0);
    @memset(output.shell_mask, 0);
    const y = positions.items(.y);
    const z = positions.items(.z);
    const inner_radius = options.cylinder_radius - options.shell_delta;
    for (0..particle_count) |particle| {
        output.shell_mask[particle] = @intFromBool(@sqrt(y[particle] * y[particle] + z[particle] * z[particle]) > inner_radius);
    }
    const eligible = try allocator.alloc(bool, particle_count);
    defer allocator.free(eligible);
    var eligible_count: usize = 0;
    for (eligible, output.shell_mask) |*destination, value| destination.* = value != 0;
    for (eligible) |value| eligible_count += @intFromBool(value);
    if (eligible_count <= options.neighbor_count) return;
    var neighbors: std.ArrayList(CellList3.Neighbor) = .empty;
    defer neighbors.deinit(allocator);
    for (0..particle_count) |particle| {
        if (!eligible[particle]) continue;
        try cells.nearest(particle, options.neighbor_count, eligible, &neighbors);
        const radius = @sqrt(y[particle] * y[particle] + z[particle] * z[particle]);
        if (radius == 0) return error.InvalidSurfacePosition;
        const normal_y = y[particle] / radius;
        const normal_z = z[particle] / radius;
        var real: f32 = 0;
        var imaginary: f32 = 0;
        for (neighbors.items) |neighbor| {
            const tangent = neighbor.displacement[1] * normal_z - neighbor.displacement[2] * normal_y;
            const angle = std.math.atan2(tangent, neighbor.displacement[0]);
            real += @cos(6.0 * angle);
            imaginary += @sin(6.0 * angle);
        }
        const denominator: f32 = @floatFromInt(options.neighbor_count);
        output.real[particle] = real / denominator;
        output.imaginary[particle] = imaginary / denominator;
        output.order[particle] = std.math.hypot(output.real[particle], output.imaginary[particle]);

        try cells.queryRadius(particle, options.coordination_radius, true, eligible, &neighbors);
        const count = std.math.cast(i16, neighbors.items.len) orelse return error.CoordinationOverflow;
        const charge = @as(i16, @intCast(options.neighbor_count)) - count;
        output.disclination[particle] = std.math.cast(i8, charge) orelse return error.CoordinationOverflow;
    }
}
