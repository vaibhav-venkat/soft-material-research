//! Dimension-generic CPU cell list over borrowed SoA coordinate slices.
//!
//! The flat heads/next representation and explicit geometry are intentionally
//! independent of simulation and storage code so equivalent GPU builders can
//! replace this implementation later.

const std = @import("std");

pub const Bounds = struct {
    lower: f32,
    upper: f32,

    pub fn length(self: Bounds) f32 {
        return self.upper - self.lower;
    }
};

pub fn CellList(comptime dimensions: usize) type {
    if (dimensions != 2 and dimensions != 3) @compileError("CellList supports only 2D or 3D coordinates");
    return struct {
        const Self = @This();
        pub const Neighbor = struct {
            index: usize,
            displacement: [dimensions]f32,
            distance_squared: f32,
        };

        allocator: std.mem.Allocator,
        components: [dimensions][]const f32,
        bounds: [dimensions]Bounds,
        periodic: [dimensions]bool,
        cell_width: f32,
        cell_counts: [dimensions]usize,
        strides: [dimensions]usize,
        heads: []usize,
        next: []usize,
        seen: []usize,
        generation: usize = 0,

        const empty = std.math.maxInt(usize);

        pub fn init(
            allocator: std.mem.Allocator,
            components: [dimensions][]const f32,
            bounds: [dimensions]Bounds,
            periodic: [dimensions]bool,
            cell_width: f32,
        ) !Self {
            try validateInput(components, bounds, cell_width);
            const particle_count = components[0].len;
            var counts: [dimensions]usize = undefined;
            for (0..dimensions) |axis| {
                const raw = @as(usize, @intFromFloat(@floor(bounds[axis].length() / cell_width)));
                counts[axis] = @max(1, @min(@max(particle_count, 1), raw));
            }
            capCellCount(&counts, @max(particle_count, 1));
            const strides = makeStrides(counts);
            const cell_count = strides[dimensions - 1] * counts[dimensions - 1];
            const heads = try allocator.alloc(usize, cell_count);
            errdefer allocator.free(heads);
            const next = try allocator.alloc(usize, particle_count);
            errdefer allocator.free(next);
            const seen = try allocator.alloc(usize, particle_count);
            errdefer allocator.free(seen);
            @memset(seen, 0);
            var result = Self{
                .allocator = allocator,
                .components = components,
                .bounds = bounds,
                .periodic = periodic,
                .cell_width = cell_width,
                .cell_counts = counts,
                .strides = strides,
                .heads = heads,
                .next = next,
                .seen = seen,
            };
            try result.rebuild(components);
            return result;
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.seen);
            self.allocator.free(self.next);
            self.allocator.free(self.heads);
            self.* = undefined;
        }

        pub fn rebuild(self: *Self, components: [dimensions][]const f32) !void {
            if (components[0].len != self.next.len) return error.DimensionMismatch;
            for (components) |component| if (component.len != self.next.len) return error.DimensionMismatch;
            self.components = components;
            @memset(self.heads, empty);
            for (0..self.next.len) |particle| {
                const cell = try self.particleCell(particle);
                self.next[particle] = self.heads[cell];
                self.heads[cell] = particle;
            }
        }

        pub fn queryRadius(
            self: *Self,
            query_index: usize,
            radius: f32,
            exclude_self: bool,
            eligible: ?[]const bool,
            output: *std.ArrayList(Neighbor),
        ) !void {
            if (query_index >= self.next.len) return error.IndexOutOfBounds;
            if (!std.math.isFinite(radius) or radius <= 0) return error.InvalidRadius;
            if (eligible) |mask| if (mask.len != self.next.len) return error.DimensionMismatch;
            output.clearRetainingCapacity();
            self.nextGeneration();
            const center = try self.pointCell(query_index);
            var steps: [dimensions]usize = undefined;
            var widths: [dimensions]f32 = undefined;
            for (0..dimensions) |axis| {
                widths[axis] = self.bounds[axis].length() / @as(f32, @floatFromInt(self.cell_counts[axis]));
                steps[axis] = @min(
                    self.cell_counts[axis] - 1,
                    @as(usize, @intFromFloat(@ceil(radius / widths[axis]))),
                );
            }
            var offsets: [dimensions]isize = undefined;
            for (0..dimensions) |axis| offsets[axis] = -@as(isize, @intCast(steps[axis]));
            const radius_squared = radius * radius;
            while (true) {
                var coordinates: [dimensions]usize = undefined;
                var valid_cell = true;
                for (0..dimensions) |axis| {
                    const raw = @as(isize, @intCast(center[axis])) + offsets[axis];
                    if (self.periodic[axis]) {
                        coordinates[axis] = @intCast(@mod(raw, @as(isize, @intCast(self.cell_counts[axis]))));
                    } else if (raw < 0 or raw >= self.cell_counts[axis]) {
                        valid_cell = false;
                    } else coordinates[axis] = @intCast(raw);
                }
                if (valid_cell) {
                    var neighbor = self.heads[self.flatten(coordinates)];
                    while (neighbor != empty) : (neighbor = self.next[neighbor]) {
                        if (self.seen[neighbor] == self.generation) continue;
                        self.seen[neighbor] = self.generation;
                        if (exclude_self and neighbor == query_index) continue;
                        if (eligible) |mask| if (!mask[neighbor]) continue;
                        var displacement: [dimensions]f32 = undefined;
                        var distance_squared: f32 = 0;
                        for (0..dimensions) |axis| {
                            var difference = self.components[axis][neighbor] - self.components[axis][query_index];
                            if (self.periodic[axis]) {
                                const period = self.bounds[axis].length();
                                difference -= period * @round(difference / period);
                            }
                            displacement[axis] = difference;
                            distance_squared += difference * difference;
                        }
                        if (distance_squared <= radius_squared) try output.append(self.allocator, .{
                            .index = neighbor,
                            .displacement = displacement,
                            .distance_squared = distance_squared,
                        });
                    }
                }
                if (!advanceOffsets(dimensions, &offsets, steps)) break;
            }
        }

        pub fn nearest(
            self: *Self,
            query_index: usize,
            count: usize,
            eligible: ?[]const bool,
            output: *std.ArrayList(Neighbor),
        ) !void {
            if (count == 0) {
                output.clearRetainingCapacity();
                return;
            }
            var radius = self.cell_width;
            const maximum = self.maximumDistance();
            while (true) {
                try self.queryRadius(query_index, radius, true, eligible, output);
                if (output.items.len >= count or radius >= maximum) break;
                radius = @min(maximum, radius * 2.0);
            }
            if (output.items.len < count) return error.NotEnoughNeighbors;
            std.mem.sort(Neighbor, output.items, {}, struct {
                fn lessThan(_: void, left: Neighbor, right: Neighbor) bool {
                    return left.distance_squared < right.distance_squared or
                        (left.distance_squared == right.distance_squared and left.index < right.index);
                }
            }.lessThan);
            output.shrinkRetainingCapacity(count);
        }

        fn maximumDistance(self: Self) f32 {
            var squared: f32 = 0;
            for (0..dimensions) |axis| {
                const span = self.bounds[axis].length() * (if (self.periodic[axis]) @as(f32, 0.5) else 1.0);
                squared += span * span;
            }
            return @sqrt(squared) + std.math.floatEps(f32);
        }

        fn particleCell(self: Self, particle: usize) !usize {
            return self.flatten(try self.pointCell(particle));
        }

        fn pointCell(self: Self, particle: usize) ![dimensions]usize {
            var result: [dimensions]usize = undefined;
            for (0..dimensions) |axis| {
                var value = self.components[axis][particle];
                if (!std.math.isFinite(value)) return error.NonFiniteCoordinate;
                const bound = self.bounds[axis];
                if (self.periodic[axis]) {
                    value = bound.lower + @mod(value - bound.lower, bound.length());
                } else if (value < bound.lower or value > bound.upper) return error.CoordinateOutOfBounds;
                const scaled = (value - bound.lower) / bound.length() * @as(f32, @floatFromInt(self.cell_counts[axis]));
                result[axis] = @min(self.cell_counts[axis] - 1, @as(usize, @intFromFloat(@floor(scaled))));
            }
            return result;
        }

        fn flatten(self: Self, coordinates: [dimensions]usize) usize {
            var result: usize = 0;
            for (0..dimensions) |axis| result += coordinates[axis] * self.strides[axis];
            return result;
        }

        fn nextGeneration(self: *Self) void {
            if (self.generation == std.math.maxInt(usize)) {
                @memset(self.seen, 0);
                self.generation = 1;
            } else self.generation += 1;
        }

        fn validateInput(components: [dimensions][]const f32, bounds: [dimensions]Bounds, width: f32) !void {
            if (!std.math.isFinite(width) or width <= 0) return error.InvalidCellWidth;
            const particle_count = components[0].len;
            if (particle_count == 0) return error.NoParticles;
            for (0..dimensions) |axis| {
                if (components[axis].len != particle_count) return error.DimensionMismatch;
                if (!std.math.isFinite(bounds[axis].lower) or !std.math.isFinite(bounds[axis].upper) or
                    bounds[axis].upper <= bounds[axis].lower) return error.InvalidBounds;
            }
        }

        fn makeStrides(counts: [dimensions]usize) [dimensions]usize {
            var result: [dimensions]usize = undefined;
            result[0] = 1;
            for (1..dimensions) |axis| result[axis] = result[axis - 1] * counts[axis - 1];
            return result;
        }

        fn capCellCount(counts: *[dimensions]usize, capacity: usize) void {
            while (cellProduct(counts.*) > capacity) {
                var largest: usize = 0;
                for (1..dimensions) |axis| if (counts[axis] > counts[largest]) {
                    largest = axis;
                };
                if (counts[largest] == 1) break;
                counts[largest] -= 1;
            }
        }

        fn cellProduct(counts: [dimensions]usize) usize {
            var result: usize = 1;
            for (counts) |count| result = std.math.mul(usize, result, count) catch return std.math.maxInt(usize);
            return result;
        }

        fn advanceOffsets(comptime n: usize, offsets: *[n]isize, steps: [n]usize) bool {
            for (0..n) |axis| {
                const maximum = @as(isize, @intCast(steps[axis]));
                if (offsets[axis] < maximum) {
                    offsets[axis] += 1;
                    return true;
                }
                offsets[axis] = -maximum;
            }
            return false;
        }
    };
}
