//! Structural clustering with a reusable periodic cell-list workspace.

const std = @import("std");
const Options = @import("options.zig").Options;
const Result = @import("result.zig").Result;
const schema = @import("schema.zig");

const StructuralOrder = struct {
    orientation: f64,
    ordered: bool,
};

const empty_cell = std.math.maxInt(usize);

/// Scratch storage retained by one frame worker and reused between frames.
pub const Workspace = struct {
    particle_count: usize,
    usize_storage: []usize,
    heads: []usize,
    next: []usize,
    seen: []usize,
    parent: []usize,
    component_sizes: []usize,
    order: []StructuralOrder,
    rank: []u8,
    present: []bool,
    spatial_edges: std.ArrayList([2]usize) = .empty,

    pub fn init(allocator: std.mem.Allocator, particle_count: usize) !Workspace {
        if (particle_count == 0) return error.NoParticles;
        const storage_len = try std.math.mul(usize, particle_count, 5);
        const usize_storage = try allocator.alloc(usize, storage_len);
        errdefer allocator.free(usize_storage);
        const order = try allocator.alloc(StructuralOrder, particle_count);
        errdefer allocator.free(order);
        const rank = try allocator.alloc(u8, particle_count);
        errdefer allocator.free(rank);
        const present = try allocator.alloc(bool, particle_count);
        errdefer allocator.free(present);

        var spatial_edges: std.ArrayList([2]usize) = .empty;
        errdefer spatial_edges.deinit(allocator);
        const expected_edges = try std.math.mul(usize, particle_count, 4);
        try spatial_edges.ensureTotalCapacity(allocator, expected_edges);

        return .{
            .particle_count = particle_count,
            .usize_storage = usize_storage,
            .heads = usize_storage[0..particle_count],
            .next = usize_storage[particle_count .. 2 * particle_count],
            .seen = usize_storage[2 * particle_count .. 3 * particle_count],
            .parent = usize_storage[3 * particle_count .. 4 * particle_count],
            .component_sizes = usize_storage[4 * particle_count .. 5 * particle_count],
            .order = order,
            .rank = rank,
            .present = present,
            .spatial_edges = spatial_edges,
        };
    }

    pub fn deinit(self: *Workspace, allocator: std.mem.Allocator) void {
        self.spatial_edges.deinit(allocator);
        allocator.free(self.present);
        allocator.free(self.rank);
        allocator.free(self.order);
        allocator.free(self.usize_storage);
        self.* = undefined;
    }
};

const DisjointSet = struct {
    parent: []usize,
    rank: []u8,

    fn reset(self: *DisjointSet) void {
        for (self.parent, 0..) |*value, index| value.* = index;
        @memset(self.rank, 0);
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

/// Convenience entry point for callers analyzing only one frame.
pub fn analyzeStructuralFrame(
    allocator: std.mem.Allocator,
    frame: schema.StructuralFrame,
    options: Options,
) !Result {
    var workspace = try Workspace.init(allocator, frame.points.len);
    defer workspace.deinit(allocator);
    return analyzeStructuralFrameWithWorkspace(allocator, &workspace, frame, options);
}

/// Analyze one frame while retaining all large scratch allocations in `workspace`.
pub fn analyzeStructuralFrameWithWorkspace(
    allocator: std.mem.Allocator,
    workspace: *Workspace,
    frame: schema.StructuralFrame,
    options: Options,
) !Result {
    var ratios: std.ArrayList(f64) = .empty;
    errdefer ratios.deinit(allocator);
    try appendStructuralFrameRatios(allocator, workspace, frame, options, &ratios);
    return .{ .ratios = try ratios.toOwnedSlice(allocator) };
}

/// Append one frame's samples directly to a worker-owned output buffer.
pub fn appendStructuralFrameRatios(
    allocator: std.mem.Allocator,
    workspace: *Workspace,
    frame: schema.StructuralFrame,
    options: Options,
    ratios: *std.ArrayList(f64),
) !void {
    try schema.validateOptions(options);
    try schema.validateStructuralFrame(frame);
    if (workspace.particle_count != frame.points.len) return error.DimensionMismatch;

    const cutoff = options.neighbor_radius_diameters * options.particle_diameter;
    const cell_counts = chooseCellCounts(frame.periods, cutoff, frame.points.len);
    const cell_count = try std.math.mul(usize, cell_counts[0], cell_counts[1]);
    std.debug.assert(cell_count <= workspace.heads.len);
    @memset(workspace.heads[0..cell_count], empty_cell);
    @memset(workspace.seen, 0);
    workspace.spatial_edges.clearRetainingCapacity();

    for (frame.psi6, frame.eligible, workspace.order) |psi6, eligible, *destination| {
        destination.* = .{
            .orientation = std.math.atan2(psi6[1], psi6[0]) / 6.0,
            .ordered = eligible and std.math.hypot(psi6[0], psi6[1]) > options.psi6_minimum,
        };
    }

    // Insert each eligible particle once. Periodicity is handled by wrapped
    // neighboring-cell indices and minimum-image distances during lookup.
    for (frame.points, frame.eligible, 0..) |point, eligible, particle_index| {
        if (!eligible) continue;
        const cell = pointCell(point, frame.periods, cell_counts);
        workspace.next[particle_index] = workspace.heads[cell];
        workspace.heads[cell] = particle_index;
    }

    const neighbor_offsets = [_]i8{ -1, 0, 1 };
    const cutoff_squared = cutoff * cutoff;
    for (frame.points, frame.eligible, 0..) |point, eligible, particle_index| {
        if (!eligible) continue;
        const generation = particle_index + 1;
        const coordinates = pointCellCoordinates(point, frame.periods, cell_counts);
        for (neighbor_offsets) |dx| {
            const cell_x = wrappedCell(coordinates[0], dx, cell_counts[0]);
            for (neighbor_offsets) |ds| {
                const cell_s = wrappedCell(coordinates[1], ds, cell_counts[1]);
                var neighbor_index = workspace.heads[cell_x * cell_counts[1] + cell_s];
                while (neighbor_index != empty_cell) : (neighbor_index = workspace.next[neighbor_index]) {
                    if (neighbor_index == particle_index or
                        workspace.seen[neighbor_index] == generation) continue;
                    workspace.seen[neighbor_index] = generation;
                    if (neighbor_index < particle_index) continue;
                    if (periodicDistanceSquared(point, frame.points[neighbor_index], frame.periods) <=
                        cutoff_squared)
                    {
                        try workspace.spatial_edges.append(
                            allocator,
                            .{ particle_index, neighbor_index },
                        );
                    }
                }
            }
        }
    }

    var components = DisjointSet{ .parent = workspace.parent, .rank = workspace.rank };
    components.reset();
    @memset(workspace.present, false);
    const maximum_misorientation = options.misorientation_degrees * std.math.pi / 180.0;
    for (workspace.spatial_edges.items) |edge| {
        const left = edge[0];
        const right = edge[1];
        if (!workspace.order[left].ordered or !workspace.order[right].ordered) continue;
        if (latticeMisorientation(
            workspace.order[left].orientation,
            workspace.order[right].orientation,
        ) >= maximum_misorientation) continue;
        components.unionSets(left, right);
        workspace.present[left] = true;
        workspace.present[right] = true;
    }

    @memset(workspace.component_sizes, 0);
    for (workspace.present, 0..) |is_present, particle_index| {
        if (!is_present) continue;
        workspace.component_sizes[components.find(particle_index)] += 1;
    }

    const particle_area = std.math.pi * options.particle_diameter * options.particle_diameter / 4.0;
    const surface_area = frame.periods[0] * frame.periods[1];
    for (workspace.component_sizes) |component_size| {
        if (component_size < options.minimum_particles) continue;
        const count: f64 = @floatFromInt(component_size);
        const area_fraction = count * particle_area / surface_area;
        try ratios.append(allocator, switch (options.ratio_mode) {
            .area_fraction => area_fraction,
            .sqrt_area_fraction => @sqrt(area_fraction),
        });
    }
}

fn chooseCellCounts(periods: [2]f64, cutoff: f64, particle_count: usize) [2]usize {
    var counts = [2]usize{
        cappedCellCount(periods[0], cutoff, particle_count),
        cappedCellCount(periods[1], cutoff, particle_count),
    };
    const product: f64 = @as(f64, @floatFromInt(counts[0])) *
        @as(f64, @floatFromInt(counts[1]));
    const capacity: f64 = @floatFromInt(particle_count);
    if (product > capacity) {
        const scale = @sqrt(capacity / product);
        counts[0] = @max(1, @as(usize, @intFromFloat(@floor(
            @as(f64, @floatFromInt(counts[0])) * scale,
        ))));
        counts[1] = @max(1, @as(usize, @intFromFloat(@floor(
            @as(f64, @floatFromInt(counts[1])) * scale,
        ))));
    }
    while (counts[0] > particle_count / counts[1]) {
        if (counts[0] >= counts[1] and counts[0] > 1) counts[0] -= 1 else counts[1] -= 1;
    }
    return counts;
}

fn cappedCellCount(period: f64, cutoff: f64, capacity: usize) usize {
    const ratio = @floor(period / cutoff);
    const capacity_float: f64 = @floatFromInt(capacity);
    if (ratio >= capacity_float) return capacity;
    return @max(1, @as(usize, @intFromFloat(ratio)));
}

fn pointCell(point: [2]f64, periods: [2]f64, counts: [2]usize) usize {
    const coordinates = pointCellCoordinates(point, periods, counts);
    return coordinates[0] * counts[1] + coordinates[1];
}

fn pointCellCoordinates(point: [2]f64, periods: [2]f64, counts: [2]usize) [2]usize {
    var result: [2]usize = undefined;
    for (0..2) |axis| {
        const wrapped = @mod(point[axis], periods[axis]);
        const scaled = wrapped / periods[axis] * @as(f64, @floatFromInt(counts[axis]));
        result[axis] = @min(counts[axis] - 1, @as(usize, @intFromFloat(@floor(scaled))));
    }
    return result;
}

fn wrappedCell(value: usize, offset: i8, count: usize) usize {
    return switch (offset) {
        -1 => if (value == 0) count - 1 else value - 1,
        0 => value,
        1 => if (value + 1 == count) 0 else value + 1,
        else => unreachable,
    };
}

fn periodicDistanceSquared(left: [2]f64, right: [2]f64, periods: [2]f64) f64 {
    var squared: f64 = 0.0;
    for (0..2) |axis| {
        var difference = right[axis] - left[axis];
        difference -= periods[axis] * @round(difference / periods[axis]);
        squared += difference * difference;
    }
    return squared;
}

fn latticeMisorientation(left: f64, right: f64) f64 {
    const period = std.math.pi / 3.0;
    const difference = @mod(left - right, period);
    return @min(difference, period - difference);
}
