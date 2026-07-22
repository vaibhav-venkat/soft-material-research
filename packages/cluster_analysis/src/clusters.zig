//! Structural-cluster numerical entry points.

const std = @import("std");
const Options = @import("options.zig").Options;
const Result = @import("result.zig").Result;
const schema = @import("schema.zig");
const kdtree = @import("backend/kdtree.zig");

const StructuralOrder = struct {
    orientation: f64,
    ordered: bool,
};

const DisjointSet = struct {
    parent: []usize,
    rank: []u8,

    fn init(allocator: std.mem.Allocator, count: usize) !DisjointSet {
        const parent = try allocator.alloc(usize, count);
        errdefer allocator.free(parent);
        const rank = try allocator.alloc(u8, count);
        errdefer allocator.free(rank);
        for (parent, 0..) |*value, index| value.* = index;
        @memset(rank, 0);
        return .{ .parent = parent, .rank = rank };
    }

    fn deinit(self: *DisjointSet, allocator: std.mem.Allocator) void {
        allocator.free(self.rank);
        allocator.free(self.parent);
        self.* = undefined;
    }

    fn find(self: *DisjointSet, value: usize) usize {
        var root = value;
        while (self.parent[root] != root) {
            self.parent[root] = self.parent[self.parent[root]];
            root = self.parent[root];
        }
        return root;
    }

    fn unionSets(self: *DisjointSet, left: usize, right: usize) void {
        var left_root = self.find(left);
        var right_root = self.find(right);
        if (left_root == right_root) return;
        if (self.rank[left_root] < self.rank[right_root]) {
            std.mem.swap(usize, &left_root, &right_root);
        }
        self.parent[right_root] = left_root;
        if (self.rank[left_root] == self.rank[right_root]) self.rank[left_root] += 1;
    }
};

pub fn analyzeStructuralFrame(
    allocator: std.mem.Allocator,
    frame: schema.StructuralFrame,
    options: Options,
) !Result {
    try schema.validateOptions(options);
    try schema.validateStructuralFrame(frame);

    var tree = try kdtree.Tree2.init();
    defer tree.deinit();

    const lx = frame.periods[0];
    const circumference = frame.periods[1];
    const x_offsets = [_]f64{ -lx, 0.0, lx };
    const surface_offsets = [_]f64{ -circumference, 0.0, circumference };

    for (frame.points, frame.eligible, 0..) |point, eligible, particle_index| {
        if (!eligible) continue;
        for (x_offsets) |dx| {
            for (surface_offsets) |ds| {
                try tree.insert(.{ point[0] + dx, point[1] + ds }, particle_index);
            }
        }
    }

    var spatial_edges: std.ArrayList([2]usize) = .empty;
    defer spatial_edges.deinit(allocator);
    const seen = try allocator.alloc(usize, frame.points.len);
    defer allocator.free(seen);
    @memset(seen, 0);

    const cutoff = options.neighbor_radius_diameters * options.particle_diameter;
    for (frame.points, frame.eligible, 0..) |point, eligible, particle_index| {
        if (!eligible) continue;
        const generation = particle_index + 1;
        const nearby = try tree.within(allocator, point, cutoff);
        defer allocator.free(nearby);

        for (nearby) |neighbor_index| {
            if (neighbor_index == particle_index or seen[neighbor_index] == generation) continue;
            seen[neighbor_index] = generation;
            if (neighbor_index > particle_index) {
                try spatial_edges.append(allocator, .{ particle_index, neighbor_index });
            }
        }
    }

    const order = try allocator.alloc(StructuralOrder, frame.points.len);
    defer allocator.free(order);
    for (frame.psi6, frame.eligible, order) |psi6, eligible, *destination| {
        destination.* = .{
            .orientation = std.math.atan2(psi6[1], psi6[0]) / 6.0,
            .ordered = eligible and std.math.hypot(psi6[0], psi6[1]) > options.psi6_minimum,
        };
    }

    var components = try DisjointSet.init(allocator, frame.points.len);
    defer components.deinit(allocator);
    const present = try allocator.alloc(bool, frame.points.len);
    defer allocator.free(present);
    @memset(present, false);

    const maximum_misorientation = options.misorientation_degrees * std.math.pi / 180.0;
    for (spatial_edges.items) |edge| {
        const left = edge[0];
        const right = edge[1];
        if (!order[left].ordered or !order[right].ordered) continue;
        if (latticeMisorientation(order[left].orientation, order[right].orientation) >=
            maximum_misorientation) continue;

        components.unionSets(left, right);
        present[left] = true;
        present[right] = true;
    }

    const component_sizes = try allocator.alloc(usize, frame.points.len);
    defer allocator.free(component_sizes);
    @memset(component_sizes, 0);
    for (present, 0..) |is_present, particle_index| {
        if (!is_present) continue;
        component_sizes[components.find(particle_index)] += 1;
    }
    var ratios: std.ArrayList(f64) = .empty;
    errdefer ratios.deinit(allocator);
    for (component_sizes) |component_size| {
        if (component_size < options.minimum_particles) continue;
        const count: f64 = @floatFromInt(component_size);
        const cluster_area: f64 = count * std.math.pi * std.math.pow(f64, (options.particle_diameter/2.0), 2);
        const area_fraction = cluster_area / (frame.periods[0] * frame.periods[1]);
        const ratio = switch (options.ratio_mode) {
            .area_fraction => area_fraction,
            .sqrt_area_fraction => @sqrt(area_fraction),
        };
        try ratios.append(allocator, ratio);
    }
    return .{
        .ratios = try ratios.toOwnedSlice(allocator),
    };
}

fn latticeMisorientation(left: f64, right: f64) f64 {
    const period = std.math.pi / 3.0;
    const difference = @mod(left - right, period);
    return @min(difference, period - difference);
}
