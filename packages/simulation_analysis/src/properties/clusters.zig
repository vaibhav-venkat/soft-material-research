//! Structural crystal components returned as raw particle-count samples.

const std = @import("std");
const data_structures = @import("../data_structures/root.zig");
const common = @import("math.zig");

const CellList2 = data_structures.CellList(2);

pub const Options = struct {
    axial_period: f32,
    cylinder_radius: f32,
    psi6_minimum: f32 = 0.7,
    misorientation_degrees: f32 = 5.0,
    neighbor_radius: f32,
    minimum_particles: usize = 2,
};

pub fn calculate(
    allocator: std.mem.Allocator,
    x: []const f32,
    theta: []const f32,
    psi_real: []const f32,
    psi_imaginary: []const f32,
    shell_mask: []const u8,
    options: Options,
) !std.ArrayList(u32) {
    const particle_count = x.len;
    if (theta.len != particle_count or psi_real.len != particle_count or
        psi_imaginary.len != particle_count or shell_mask.len != particle_count)
        return error.DimensionMismatch;
    if (!std.math.isFinite(options.axial_period) or options.axial_period <= 0 or
        !std.math.isFinite(options.cylinder_radius) or options.cylinder_radius <= 0 or
        !std.math.isFinite(options.neighbor_radius) or options.neighbor_radius <= 0 or
        !std.math.isFinite(options.psi6_minimum) or options.psi6_minimum < 0 or options.psi6_minimum > 1 or
        !std.math.isFinite(options.misorientation_degrees) or options.misorientation_degrees < 0 or
        options.misorientation_degrees > 30 or options.minimum_particles < 2)
        return error.InvalidClusterOptions;
    const circumference = std.math.tau * options.cylinder_radius;
    const surface = try allocator.alloc(f32, particle_count);
    defer allocator.free(surface);
    for (surface, theta) |*destination, value| destination.* = @mod(value, std.math.tau) * options.cylinder_radius;
    var cells = try CellList2.init(
        allocator,
        .{ x, surface },
        .{
            .{ .lower = -options.axial_period / 2.0, .upper = options.axial_period / 2.0 },
            .{ .lower = 0, .upper = circumference },
        },
        .{ true, true },
        options.neighbor_radius,
    );
    defer cells.deinit();
    const parent = try allocator.alloc(usize, particle_count);
    defer allocator.free(parent);
    const rank = try allocator.alloc(u8, particle_count);
    defer allocator.free(rank);
    const sizes = try allocator.alloc(usize, particle_count);
    defer allocator.free(sizes);
    const ordered = try allocator.alloc(bool, particle_count);
    defer allocator.free(ordered);
    const present = try allocator.alloc(bool, particle_count);
    defer allocator.free(present);
    for (parent, 0..) |*value, index| value.* = index;
    @memset(rank, 0);
    @memset(sizes, 0);
    @memset(present, false);
    for (ordered, 0..) |*value, particle| {
        value.* = shell_mask[particle] != 0 and
            std.math.hypot(psi_real[particle], psi_imaginary[particle]) > options.psi6_minimum;
    }
    var neighbors: std.ArrayList(CellList2.Neighbor) = .empty;
    defer neighbors.deinit(allocator);
    const maximum_misorientation = options.misorientation_degrees * std.math.pi / 180.0;
    for (0..particle_count) |particle| {
        if (!ordered[particle]) continue;
        try cells.queryRadius(particle, options.neighbor_radius, true, ordered, &neighbors);
        for (neighbors.items) |neighbor| {
            if (neighbor.index <= particle) continue;
            const left_orientation = std.math.atan2(psi_imaginary[particle], psi_real[particle]) / 6.0;
            const right_orientation = std.math.atan2(psi_imaginary[neighbor.index], psi_real[neighbor.index]) / 6.0;
            if (common.sixfoldMisorientation(left_orientation, right_orientation) >= maximum_misorientation) continue;
            unionSets(parent, rank, particle, neighbor.index);
            present[particle] = true;
            present[neighbor.index] = true;
        }
    }
    for (present, 0..) |is_present, particle| {
        if (is_present) sizes[find(parent, particle)] += 1;
    }
    var result: std.ArrayList(u32) = .empty;
    errdefer result.deinit(allocator);
    for (sizes) |size| {
        if (size < options.minimum_particles) continue;
        try result.append(allocator, std.math.cast(u32, size) orelse return error.ClusterSizeOverflow);
    }
    return result;
}

fn find(parent: []usize, value: usize) usize {
    var root = value;
    while (parent[root] != root) {
        parent[root] = parent[parent[root]];
        root = parent[root];
    }
    return root;
}

fn unionSets(parent: []usize, rank: []u8, left: usize, right: usize) void {
    var left_root = find(parent, left);
    var right_root = find(parent, right);
    if (left_root == right_root) return;
    if (rank[left_root] < rank[right_root]) std.mem.swap(usize, &left_root, &right_root);
    parent[right_root] = left_root;
    if (rank[left_root] == rank[right_root]) rank[left_root] += 1;
}
