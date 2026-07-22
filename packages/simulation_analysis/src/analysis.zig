//! Sharded GSD-to-HDF5 storage and cell-list-backed particle analysis.

const std = @import("std");
const gsd = @import("gsd");
const hdf5 = @import("hdf5");
const Options = @import("options.zig").Options;
const schema = @import("schema.zig");
const coordinate_types = @import("data_structures/coordinates.zig");
const data_structures = @import("data_structures/root.zig");
const properties = @import("properties/root.zig");

pub const ShardRange = struct {
    start: u64,
    stop: u64,
};

pub const Summary = struct {
    frame_count: u64,
    shard_count: usize,
    fields_written: usize,
};

pub const CartesianPosition = coordinate_types.CartesianPosition;
pub const CylindricalCoordinate = coordinate_types.CylindricalCoordinate;
pub const CartesianPositions = coordinate_types.CartesianPositions;
pub const CylindricalCoordinates = coordinate_types.CylindricalCoordinates;
pub const Quaternions = coordinate_types.Quaternions;
pub const CylindricalVectors = coordinate_types.CylindricalVectors;

const ShardState = struct {
    input_path: []const u8,
    output_dir: []const u8,
    ranges: []const ShardRange,
    particle_count: u64,
    update: bool,
    serialize_hdf5: bool,
    next_shard: *std.atomic.Value(usize),
    hdf5_mutex: *std.atomic.Mutex,
    fields_written: usize = 0,
    failure: ?anyerror = null,
};

const MissingFields = struct {
    frame_index: bool,
    step: bool,
    box: bool,
    coords: bool,
    replace_coords: bool = false,

    fn count(self: MissingFields) usize {
        return @as(usize, @intFromBool(self.frame_index)) +
            @as(usize, @intFromBool(self.step)) +
            @as(usize, @intFromBool(self.box)) +
            @as(usize, @intFromBool(self.coords));
    }
};

/// Partition frames into contiguous ranges sized from all fixed particle fields.
/// The caller owns the returned slice. `stop` is exclusive.
pub fn planShards(
    allocator: std.mem.Allocator,
    frame_count: u64,
    particle_count: u64,
    target_bytes: usize,
) ![]ShardRange {
    if (target_bytes == 0) return error.InvalidShardSize;
    if (frame_count == 0) return allocator.alloc(ShardRange, 0);
    const particles = std.math.cast(usize, particle_count) orelse return error.SizeOverflow;
    // coords (12), three psi fields (12), charge/mask (2), rho (4), polar (12).
    const bytes_per_frame = std.math.mul(usize, particles, 42) catch return error.SizeOverflow;
    const frames_per_shard = if (bytes_per_frame == 0) frame_count else @max(
        @as(u64, 1),
        std.math.cast(u64, target_bytes / bytes_per_frame) orelse return error.SizeOverflow,
    );
    const shard_count_u64 = (frame_count + frames_per_shard - 1) / frames_per_shard;
    const shard_count = std.math.cast(usize, shard_count_u64) orelse return error.SizeOverflow;
    const ranges = try allocator.alloc(ShardRange, shard_count);
    var start: u64 = 0;
    for (ranges) |*range| {
        const stop = @min(frame_count, start + frames_per_shard);
        range.* = .{ .start = start, .stop = stop };
        start = stop;
    }
    return ranges;
}

/// Convert SoA Cartesian positions to SoA cylindrical coordinates around the
/// x axis. Theta is in `[0, 2*pi)`.
pub fn transformCylindrical(
    positions: CartesianPositions.Slice,
    coordinates: CylindricalCoordinates.Slice,
) !void {
    return properties.math.transformCylindrical(positions, coordinates);
}

/// Convert the trajectory into `static.h5` plus independently writable frame
/// shards. Workers use separate GSD handles and separate HDF5 files.
pub fn run(allocator: std.mem.Allocator, options: Options) !Summary {
    try schema.validateOptions(options);
    var input = try gsd.File.openRead(allocator, options.input_path);
    defer input.deinit();
    const frame_count = input.frameCount();
    if (frame_count == 0) return error.EmptyTrajectory;
    const particle_chunk = try frameChunk(&input, 0, "particles/N");
    if (particle_chunk.dtype != .u32 or try particle_chunk.elementCount() != 1) return error.InvalidParticleCount;
    var particle_value: [1]u32 = undefined;
    try input.readChunkBytes(particle_chunk, std.mem.asBytes(&particle_value));
    const particle_count: u64 = particle_value[0];
    if (particle_count == 0) return error.InvalidParticleCount;
    var box: [6]f32 = undefined;
    try readFrameChunk(&input, f32, 0, "configuration/box", &box);

    const ranges = try planShards(allocator, frame_count, particle_count, options.target_shard_bytes);
    defer allocator.free(ranges);
    if (options.dry_run) return .{ .frame_count = frame_count, .shard_count = ranges.len, .fields_written = 0 };

    const io = std.Io.Threaded.global_single_threaded.io();
    const cwd = std.Io.Dir.cwd();
    const output_exists = blk: {
        cwd.access(io, options.output_dir, .{}) catch |err| switch (err) {
            error.FileNotFound => break :blk false,
            else => return err,
        };
        break :blk true;
    };
    switch (options.write_mode) {
        .create => if (output_exists) return error.OutputExists,
        .update => if (!output_exists) return error.OutputNotFound,
        .overwrite => if (output_exists) try cwd.deleteTree(io, options.output_dir),
    }
    if (!output_exists or options.write_mode == .overwrite) try cwd.createDirPath(io, options.output_dir);

    try writeStaticFile(allocator, options, &input, frame_count, particle_count);
    const thread_safe_hdf5 = try hdf5.isLibraryThreadSafe();
    var next_shard = std.atomic.Value(usize).init(0);
    var hdf5_mutex: std.atomic.Mutex = .unlocked;
    const requested_workers = options.worker_count orelse (std.Thread.getCpuCount() catch 1);
    const worker_count = @max(@as(usize, 1), @min(requested_workers, ranges.len));
    const states = try allocator.alloc(ShardState, worker_count);
    defer allocator.free(states);
    for (states) |*state| state.* = .{
        .input_path = options.input_path,
        .output_dir = options.output_dir,
        .ranges = ranges,
        .particle_count = particle_count,
        .update = options.write_mode == .update,
        .serialize_hdf5 = !thread_safe_hdf5,
        .next_shard = &next_shard,
        .hdf5_mutex = &hdf5_mutex,
    };

    const threads = try allocator.alloc(std.Thread, worker_count - 1);
    defer allocator.free(threads);
    var spawned: usize = 0;
    for (threads, 0..) |*thread, index| {
        thread.* = std.Thread.spawn(.{}, shardWorker, .{&states[index]}) catch |err| {
            for (threads[0..spawned]) |started| started.join();
            return err;
        };
        spawned += 1;
    }
    shardWorker(&states[worker_count - 1]);
    for (threads) |thread| thread.join();

    var fields_written: usize = 0;
    for (states) |state| {
        if (state.failure) |err| return err;
        fields_written += state.fields_written;
    }
    fields_written += try writeParticleProperties(
        allocator,
        options,
        ranges,
        particle_count,
        box,
    );
    fields_written += try writeComProperties(
        allocator,
        options.output_dir,
        ranges,
        particle_count,
        box[0],
        options.timestep,
    );
    return .{ .frame_count = frame_count, .shard_count = ranges.len, .fields_written = fields_written };
}

fn frameChunk(file: *gsd.File, frame: u64, name: [:0]const u8) !gsd.Chunk {
    if (try file.findChunk(frame, name)) |chunk| return chunk;
    // if (frame != 0) if (try file.findChunk(0, name)) |chunk| return chunk;
    return error.ChunkNotFound;
}

fn readFrameChunk(file: *gsd.File, comptime T: type, frame: u64, name: [:0]const u8, destination: []T) !void {
    const chunk = try frameChunk(file, frame, name);
    if (chunk.dtype != gsd.Dtype.of(T)) return error.DtypeMismatch;
    if (try chunk.elementCount() != destination.len) return error.BufferSizeMismatch;
    try file.readChunkBytes(chunk, std.mem.sliceAsBytes(destination));
}

fn writeStaticFile(
    allocator: std.mem.Allocator,
    options: Options,
    input: *gsd.File,
    frame_count: u64,
    particle_count: u64,
) !void {
    const path = try std.fs.path.join(allocator, &.{ options.output_dir, "static.h5" });
    defer allocator.free(path);
    var file = if (options.write_mode == .update)
        try hdf5.File.openPath(allocator, path, .read_write)
    else
        try hdf5.File.create(allocator, path, .exclusive);
    defer file.deinit();
    if (options.write_mode == .update and try file.attributeExists("cylinder_radius")) {
        if (try file.readAttribute(f64, "cylinder_radius") != options.cylinder_radius or
            try file.readAttribute(f64, "particle_diameter") != options.particle_diameter or
            try file.readAttribute(f64, "shell_delta") != options.effectiveShellDelta() or
            try file.readAttribute(f64, "neighbor_radius") != options.effectiveNeighborRadius() or
            try file.readAttribute(f64, "pocket_radius") != options.effectivePocketRadius() or
            try file.readAttribute(f64, "gaussian_cutoff_multiplier") != options.gaussian_cutoff_multiplier or
            try file.readAttribute(f64, "psi6_minimum") != options.psi6_minimum or
            try file.readAttribute(f64, "misorientation_degrees") != options.misorientation_degrees or
            try file.readAttribute(u64, "minimum_cluster_particles") != @as(u64, @intCast(options.minimum_cluster_particles)))
            return error.AnalysisConfigurationMismatch;
    }
    try file.writeStringAttribute("schema", schema.static_schema);
    try file.writeStringAttribute("coordinate_transform", schema.coordinate_transform);
    try file.writeStringAttribute("source_application", input.header().application);
    try file.writeStringAttribute("source_schema", input.header().schema);
    try file.writeAttribute(u64, "frame_count", frame_count);
    try file.writeAttribute(u64, "particle_count", particle_count);
    try file.writeAttribute(f64, "cylinder_radius", options.cylinder_radius);
    try file.writeAttribute(f64, "particle_diameter", options.particle_diameter);
    try file.writeAttribute(f64, "shell_delta", options.effectiveShellDelta());
    try file.writeAttribute(f64, "neighbor_radius", options.effectiveNeighborRadius());
    try file.writeAttribute(f64, "pocket_radius", options.effectivePocketRadius());
    try file.writeAttribute(f64, "gaussian_cutoff_multiplier", options.gaussian_cutoff_multiplier);
    try file.writeAttribute(f64, "psi6_minimum", options.psi6_minimum);
    try file.writeAttribute(f64, "misorientation_degrees", options.misorientation_degrees);
    try file.writeAttribute(u64, "minimum_cluster_particles", @intCast(options.minimum_cluster_particles));
    try file.flush();
}

fn shardWorker(state: *ShardState) void {
    while (true) {
        const index = state.next_shard.fetchAdd(1, .monotonic);
        if (index >= state.ranges.len) return;
        state.fields_written += writeShard(state, index) catch |err| {
            state.failure = err;
            return;
        };
    }
}

fn writeShard(state: *ShardState, shard_index: usize) !usize {
    const allocator = std.heap.page_allocator;
    const range = state.ranges[shard_index];
    const frame_count_u64 = range.stop - range.start;
    const frame_count = std.math.cast(usize, frame_count_u64) orelse return error.SizeOverflow;
    const particle_count = std.math.cast(usize, state.particle_count) orelse return error.SizeOverflow;
    const coordinate_count = std.math.mul(usize, frame_count, particle_count) catch return error.SizeOverflow;
    const name = try std.fmt.allocPrint(allocator, "frames_{d:0>6}.h5", .{shard_index});
    defer allocator.free(name);
    const path = try std.fs.path.join(allocator, &.{ state.output_dir, name });
    defer allocator.free(path);

    const missing: MissingFields = if (state.update) try inspectMissingFields(state, allocator, path, range) else .{
        .frame_index = true,
        .step = true,
        .box = true,
        .coords = true,
    };
    if (missing.count() == 0) return 0;

    const frames = if (missing.frame_index) try allocator.alloc(u64, frame_count) else null;
    defer if (frames) |values| allocator.free(values);
    const steps = if (missing.step) try allocator.alloc(u64, frame_count) else null;
    defer if (steps) |values| allocator.free(values);
    const boxes = if (missing.box) try allocator.alloc(f32, frame_count * 6) else null;
    defer if (boxes) |values| allocator.free(values);
    const raw_positions = if (missing.coords) try allocator.alloc(f32, particle_count * 3) else null;
    defer if (raw_positions) |values| allocator.free(values);
    var positions: CartesianPositions = .empty;
    defer positions.deinit(allocator);
    if (missing.coords) try positions.resize(allocator, particle_count);
    var coordinates: CylindricalCoordinates = .empty;
    defer coordinates.deinit(allocator);
    if (missing.coords) try coordinates.resize(allocator, coordinate_count);

    if (missing.step or missing.box or missing.coords) {
        var input = try gsd.File.openRead(allocator, state.input_path);
        defer input.deinit();
        for (0..frame_count) |local_frame| {
            const frame = range.start + local_frame;
            if (steps) |values| try readFrameChunk(&input, u64, frame, "configuration/step", values[local_frame .. local_frame + 1]);
            if (boxes) |values| try readFrameChunk(&input, f32, 0, "configuration/box", values[local_frame * 6 ..][0..6]);
            if (missing.coords) {
                var particle_value: [1]u32 = undefined;
                try readFrameChunk(&input, u32, 0, "particles/N", &particle_value);
                if (particle_value[0] != state.particle_count) return error.VariableParticleCount;
                try readFrameChunk(&input, f32, frame, "particles/position", raw_positions.?);
                deinterleaveCartesian(raw_positions.?, positions.slice());
                const output = coordinates.slice().subslice(local_frame * particle_count, particle_count);
                try transformCylindrical(positions.slice(), output);
            }
        }
    }
    if (frames) |values| {
        for (values, 0..) |*frame, local_frame| frame.* = range.start + local_frame;
    }

    if (state.serialize_hdf5) while (!state.hdf5_mutex.tryLock()) std.atomic.spinLoopHint();
    defer if (state.serialize_hdf5) state.hdf5_mutex.unlock();
    var file = if (state.update)
        try hdf5.File.openPath(allocator, path, .read_write)
    else
        try hdf5.File.create(allocator, path, .exclusive);
    defer file.deinit();
    try file.writeStringAttribute("schema", schema.frame_schema);
    try file.writeAttribute(u64, "frame_start", range.start);
    try file.writeAttribute(u64, "frame_stop", range.stop);
    try file.writeAttribute(u64, "particle_count", state.particle_count);

    if (missing.frame_index) {
        var dataset = try file.createDataset(u64, "frame_index", &.{frame_count_u64}, .{ .chunk_shape = &.{frame_count_u64} });
        defer dataset.deinit();
        try dataset.writeAll(u64, frames.?);
    }
    if (missing.step) {
        var dataset = try file.createDataset(u64, "step", &.{frame_count_u64}, .{ .chunk_shape = &.{frame_count_u64} });
        defer dataset.deinit();
        try dataset.writeAll(u64, steps.?);
    }
    if (missing.box) {
        var dataset = try file.createDataset(f32, "box", &.{ frame_count_u64, 6 }, .{ .chunk_shape = &.{ 1, 6 } });
        defer dataset.deinit();
        try dataset.writeAll(f32, boxes.?);
    }
    if (missing.coords) {
        if (missing.replace_coords) try file.deleteLink("coords");
        var dataset = try file.createDataset(f32, "coords", &.{ 3, frame_count_u64, state.particle_count }, .{ .chunk_shape = &.{ 1, 1, state.particle_count } });
        defer dataset.deinit();
        try dataset.writeStringAttribute("components", "x,theta,r");
        try dataset.writeStringAttribute("layout", "component,frame,particle");
        try dataset.writeStringAttribute("transform", schema.coordinate_transform);
        const coordinate_slices = coordinates.slice();
        try dataset.writeHyperslab(f32, &.{ 0, 0, 0 }, &.{ 1, frame_count_u64, state.particle_count }, coordinate_slices.items(.x));
        try dataset.writeHyperslab(f32, &.{ 1, 0, 0 }, &.{ 1, frame_count_u64, state.particle_count }, coordinate_slices.items(.theta));
        try dataset.writeHyperslab(f32, &.{ 2, 0, 0 }, &.{ 1, frame_count_u64, state.particle_count }, coordinate_slices.items(.r));
    }
    try file.flush();
    return missing.count();
}

const MissingParticleProperties = struct {
    hexatic_order: bool,
    psi_real: bool,
    psi_imaginary: bool,
    disclination: bool,
    shell_mask: bool,
    rho: bool,
    polar: bool,
    clusters: bool,

    fn count(self: MissingParticleProperties) usize {
        return @as(usize, @intFromBool(self.hexatic_order)) + @as(usize, @intFromBool(self.psi_real)) +
            @as(usize, @intFromBool(self.psi_imaginary)) + @as(usize, @intFromBool(self.disclination)) +
            @as(usize, @intFromBool(self.shell_mask)) + @as(usize, @intFromBool(self.rho)) +
            @as(usize, @intFromBool(self.polar)) + @as(usize, @intFromBool(self.clusters));
    }

    fn needsHexatic(self: MissingParticleProperties) bool {
        return self.hexatic_order or self.psi_real or self.psi_imaginary or
            self.disclination or self.shell_mask or self.clusters;
    }

    fn needsPolar(self: MissingParticleProperties) bool {
        return self.rho or self.polar;
    }
};

const PropertyDatasets = struct {
    order: ?*hdf5.Dataset,
    real: ?*hdf5.Dataset,
    imaginary: ?*hdf5.Dataset,
    disclination: ?*hdf5.Dataset,
    mask: ?*hdf5.Dataset,
    rho: ?*hdf5.Dataset,
    polar: ?*hdf5.Dataset,
};

const PropertyFrameState = struct {
    input_path: []const u8,
    options: Options,
    range: ShardRange,
    particle_count: usize,
    particle_count_u64: u64,
    box: [6]f32,
    missing: MissingParticleProperties,
    datasets: PropertyDatasets,
    clusters: []std.ArrayList(u32),
    next_frame: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    failed: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    hdf5_mutex: std.atomic.Mutex = .unlocked,
    failure_mutex: std.atomic.Mutex = .unlocked,
    failure: ?anyerror = null,

    fn recordFailure(self: *PropertyFrameState, err: anyerror) void {
        self.failed.store(true, .release);
        while (!self.failure_mutex.tryLock()) std.atomic.spinLoopHint();
        defer self.failure_mutex.unlock();
        if (self.failure == null) self.failure = err;
    }
};

fn writeParticleProperties(
    allocator: std.mem.Allocator,
    options: Options,
    ranges: []const ShardRange,
    particle_count_u64: u64,
    box: [6]f32,
) !usize {
    const particle_count = std.math.cast(usize, particle_count_u64) orelse return error.SizeOverflow;
    var written: usize = 0;
    for (ranges, 0..) |range, shard_index| {
        const path = try shardPath(allocator, options.output_dir, shard_index);
        defer allocator.free(path);
        var file = try hdf5.File.openPath(allocator, path, .read_write);
        defer file.deinit();
        try file.writeStringAttribute("schema", schema.frame_schema);
        const frame_count_u64 = range.stop - range.start;
        const particle_shape = [_]u64{ frame_count_u64, particle_count_u64 };
        const polar_shape = [_]u64{ 3, frame_count_u64, particle_count_u64 };
        const missing = MissingParticleProperties{
            .hexatic_order = try propertyMissing(allocator, &file, "hexatic_order", .f32, &particle_shape, "frame,particle"),
            .psi_real = try propertyMissing(allocator, &file, "psi_6_real", .f32, &particle_shape, "frame,particle"),
            .psi_imaginary = try propertyMissing(allocator, &file, "psi_6_im", .f32, &particle_shape, "frame,particle"),
            .disclination = try propertyMissing(allocator, &file, "disclination", .i8, &particle_shape, "frame,particle"),
            .shell_mask = try propertyMissing(allocator, &file, "hexatic_shell_mask", .u8, &particle_shape, "frame,particle"),
            .rho = try propertyMissing(allocator, &file, "rho", .f32, &particle_shape, "frame,particle"),
            .polar = try propertyMissing(allocator, &file, "polar_cylindrical", .f32, &polar_shape, "component,frame,particle"),
            .clusters = try propertyMissing(allocator, &file, "cluster_sizes", .u32, null, "sample"),
        };
        if (missing.count() == 0) continue;

        var order_dataset: ?hdf5.Dataset = if (missing.hexatic_order)
            try createParticleDataset(f32, &file, "hexatic_order", frame_count_u64, particle_count_u64)
        else
            null;
        defer if (order_dataset) |*dataset| dataset.deinit();
        var real_dataset: ?hdf5.Dataset = if (missing.psi_real)
            try createParticleDataset(f32, &file, "psi_6_real", frame_count_u64, particle_count_u64)
        else
            null;
        defer if (real_dataset) |*dataset| dataset.deinit();
        var imaginary_dataset: ?hdf5.Dataset = if (missing.psi_imaginary)
            try createParticleDataset(f32, &file, "psi_6_im", frame_count_u64, particle_count_u64)
        else
            null;
        defer if (imaginary_dataset) |*dataset| dataset.deinit();
        var disclination_dataset: ?hdf5.Dataset = if (missing.disclination)
            try createParticleDataset(i8, &file, "disclination", frame_count_u64, particle_count_u64)
        else
            null;
        defer if (disclination_dataset) |*dataset| dataset.deinit();
        var mask_dataset: ?hdf5.Dataset = if (missing.shell_mask)
            try createParticleDataset(u8, &file, "hexatic_shell_mask", frame_count_u64, particle_count_u64)
        else
            null;
        defer if (mask_dataset) |*dataset| dataset.deinit();
        var rho_dataset: ?hdf5.Dataset = if (missing.rho)
            try createParticleDataset(f32, &file, "rho", frame_count_u64, particle_count_u64)
        else
            null;
        defer if (rho_dataset) |*dataset| dataset.deinit();
        var polar_dataset: ?hdf5.Dataset = if (missing.polar)
            try createPolarDataset(&file, frame_count_u64, particle_count_u64)
        else
            null;
        defer if (polar_dataset) |*dataset| dataset.deinit();

        const frame_count = std.math.cast(usize, frame_count_u64) orelse return error.SizeOverflow;
        const frame_clusters = try allocator.alloc(std.ArrayList(u32), frame_count);
        defer allocator.free(frame_clusters);
        for (frame_clusters) |*samples| samples.* = .empty;
        defer for (frame_clusters) |*samples| samples.deinit(std.heap.page_allocator);
        var state = PropertyFrameState{
            .input_path = options.input_path,
            .options = options,
            .range = range,
            .particle_count = particle_count,
            .particle_count_u64 = particle_count_u64,
            .box = box,
            .missing = missing,
            .datasets = .{
                .order = if (order_dataset) |*dataset| dataset else null,
                .real = if (real_dataset) |*dataset| dataset else null,
                .imaginary = if (imaginary_dataset) |*dataset| dataset else null,
                .disclination = if (disclination_dataset) |*dataset| dataset else null,
                .mask = if (mask_dataset) |*dataset| dataset else null,
                .rho = if (rho_dataset) |*dataset| dataset else null,
                .polar = if (polar_dataset) |*dataset| dataset else null,
            },
            .clusters = frame_clusters,
        };
        const requested_workers = options.worker_count orelse (std.Thread.getCpuCount() catch 1);
        const worker_count = @max(@as(usize, 1), @min(requested_workers, frame_count));
        const threads = try allocator.alloc(std.Thread, worker_count - 1);
        defer allocator.free(threads);
        var spawned: usize = 0;
        for (threads) |*thread| {
            thread.* = std.Thread.spawn(.{}, propertyFrameWorker, .{&state}) catch |err| {
                state.recordFailure(err);
                break;
            };
            spawned += 1;
        }
        propertyFrameWorker(&state);
        for (threads[0..spawned]) |thread| thread.join();
        if (state.failure) |err| return err;

        var cluster_sizes: std.ArrayList(u32) = .empty;
        defer cluster_sizes.deinit(allocator);
        for (frame_clusters) |samples| try cluster_sizes.appendSlice(allocator, samples.items);
        if (missing.clusters) {
            var dataset = try file.createDataset(u32, "cluster_sizes", &.{cluster_sizes.items.len}, .{});
            defer dataset.deinit();
            try dataset.writeStringAttribute("layout", "sample");
            try dataset.writeAttribute(u32, "field_version", 1);
            try dataset.writeAll(u32, cluster_sizes.items);
        }
        try file.flush();
        written += missing.count();
    }
    return written;
}

fn propertyFrameWorker(state: *PropertyFrameState) void {
    analyzePropertyFrames(state) catch |err| state.recordFailure(err);
}

fn analyzePropertyFrames(state: *PropertyFrameState) !void {
    const allocator = std.heap.page_allocator;
    const particle_count = state.particle_count;
    const raw_positions = try allocator.alloc(f32, particle_count * 3);
    defer allocator.free(raw_positions);
    const raw_orientations = try allocator.alloc(f32, particle_count * 4);
    defer allocator.free(raw_orientations);
    var positions: CartesianPositions = .empty;
    defer positions.deinit(allocator);
    try positions.resize(allocator, particle_count);
    var coordinates: CylindricalCoordinates = .empty;
    defer coordinates.deinit(allocator);
    try coordinates.resize(allocator, particle_count);
    var orientations: Quaternions = .empty;
    defer orientations.deinit(allocator);
    try orientations.resize(allocator, particle_count);
    var polar_values: CylindricalVectors = .empty;
    defer polar_values.deinit(allocator);
    if (state.missing.needsPolar()) try polar_values.resize(allocator, particle_count);
    const order = try allocator.alloc(f32, particle_count);
    defer allocator.free(order);
    const psi_real = try allocator.alloc(f32, particle_count);
    defer allocator.free(psi_real);
    const psi_imaginary = try allocator.alloc(f32, particle_count);
    defer allocator.free(psi_imaginary);
    const disclination = try allocator.alloc(i8, particle_count);
    defer allocator.free(disclination);
    const shell_mask = try allocator.alloc(u8, particle_count);
    defer allocator.free(shell_mask);
    const rho = try allocator.alloc(f32, particle_count);
    defer allocator.free(rho);
    var input = try gsd.File.openRead(allocator, state.input_path);
    defer input.deinit();

    const frame_count = state.clusters.len;
    while (!state.failed.load(.acquire)) {
        const local_frame = state.next_frame.fetchAdd(1, .monotonic);
        if (local_frame >= frame_count) return;
        const local_frame_u64: u64 = @intCast(local_frame);
        const frame = state.range.start + local_frame_u64;
        try readFrameChunk(&input, f32, frame, "particles/position", raw_positions);
        deinterleaveCartesian(raw_positions, positions.slice());
        try transformCylindrical(positions.slice(), coordinates.slice());
        if (state.missing.needsPolar()) {
            try readFrameChunk(&input, f32, frame, "particles/orientation", raw_orientations);
            deinterleaveQuaternions(raw_orientations, orientations.slice());
        }
        const position_slices = positions.slice();
        var cells = try data_structures.CellList(3).init(
            allocator,
            .{ position_slices.items(.x), position_slices.items(.y), position_slices.items(.z) },
            .{
                .{ .lower = -state.box[0] / 2.0, .upper = state.box[0] / 2.0 },
                .{ .lower = -state.box[1] / 2.0, .upper = state.box[1] / 2.0 },
                .{ .lower = -state.box[2] / 2.0, .upper = state.box[2] / 2.0 },
            },
            .{ true, false, false },
            @floatCast(@max(state.options.effectiveNeighborRadius(), state.options.effectivePocketRadius())),
        );
        defer cells.deinit();
        if (state.missing.needsHexatic()) try properties.hexatic.calculate(
            allocator,
            &cells,
            positions.slice(),
            .{
                .cylinder_radius = @floatCast(state.options.cylinder_radius),
                .shell_delta = @floatCast(state.options.effectiveShellDelta()),
                .coordination_radius = @floatCast(state.options.effectiveNeighborRadius()),
            },
            .{
                .order = order,
                .real = psi_real,
                .imaginary = psi_imaginary,
                .disclination = disclination,
                .shell_mask = shell_mask,
            },
        );
        if (state.missing.needsPolar()) try properties.polar.calculate(
            allocator,
            &cells,
            orientations.slice(),
            coordinates.slice().items(.theta),
            .{
                .pocket_radius = @floatCast(state.options.effectivePocketRadius()),
                .cutoff_multiplier = @floatCast(state.options.gaussian_cutoff_multiplier),
            },
            rho,
            polar_values.slice(),
        );
        if (state.missing.clusters) {
            state.clusters[local_frame] = try properties.clusters.calculate(
                allocator,
                position_slices.items(.x),
                coordinates.slice().items(.theta),
                psi_real,
                psi_imaginary,
                shell_mask,
                .{
                    .axial_period = state.box[0],
                    .cylinder_radius = @floatCast(state.options.cylinder_radius),
                    .psi6_minimum = @floatCast(state.options.psi6_minimum),
                    .misorientation_degrees = @floatCast(state.options.misorientation_degrees),
                    .neighbor_radius = @floatCast(state.options.effectiveNeighborRadius()),
                    .minimum_particles = state.options.minimum_cluster_particles,
                },
            );
        }
        try writePropertyFrame(
            state,
            local_frame_u64,
            order,
            psi_real,
            psi_imaginary,
            disclination,
            shell_mask,
            rho,
            polar_values.slice(),
        );
    }
}

fn writePropertyFrame(
    state: *PropertyFrameState,
    local_frame: u64,
    order: []const f32,
    psi_real: []const f32,
    psi_imaginary: []const f32,
    disclination: []const i8,
    shell_mask: []const u8,
    rho: []const f32,
    polar: CylindricalVectors.Slice,
) !void {
    while (!state.hdf5_mutex.tryLock()) std.atomic.spinLoopHint();
    defer state.hdf5_mutex.unlock();
    const offset = [_]u64{ local_frame, 0 };
    const count = [_]u64{ 1, state.particle_count_u64 };
    if (state.datasets.order) |dataset| try dataset.writeHyperslab(f32, &offset, &count, order);
    if (state.datasets.real) |dataset| try dataset.writeHyperslab(f32, &offset, &count, psi_real);
    if (state.datasets.imaginary) |dataset| try dataset.writeHyperslab(f32, &offset, &count, psi_imaginary);
    if (state.datasets.disclination) |dataset| try dataset.writeHyperslab(i8, &offset, &count, disclination);
    if (state.datasets.mask) |dataset| try dataset.writeHyperslab(u8, &offset, &count, shell_mask);
    if (state.datasets.rho) |dataset| try dataset.writeHyperslab(f32, &offset, &count, rho);
    if (state.datasets.polar) |dataset| {
        try dataset.writeHyperslab(f32, &.{ 0, local_frame, 0 }, &.{ 1, 1, state.particle_count_u64 }, polar.items(.x));
        try dataset.writeHyperslab(f32, &.{ 1, local_frame, 0 }, &.{ 1, 1, state.particle_count_u64 }, polar.items(.r));
        try dataset.writeHyperslab(f32, &.{ 2, local_frame, 0 }, &.{ 1, 1, state.particle_count_u64 }, polar.items(.theta));
    }
}

fn propertyMissing(
    allocator: std.mem.Allocator,
    file: *hdf5.File,
    name: []const u8,
    dtype: hdf5.Dtype,
    expected_shape: ?[]const u64,
    expected_layout: []const u8,
) !bool {
    if (!try file.objectExists(name)) return true;
    var dataset = try file.openDataset(name);
    defer dataset.deinit();
    if (try dataset.dtype() != dtype) return error.IncompatiblePropertyField;
    const shape = try dataset.shapeAlloc(allocator);
    defer allocator.free(shape);
    if (expected_shape) |expected| {
        if (!std.mem.eql(u64, shape, expected)) return error.IncompatiblePropertyField;
    } else if (shape.len != 1) return error.IncompatiblePropertyField;
    if (!try dataset.attributeExists("layout") or !try dataset.attributeExists("field_version"))
        return error.IncompatiblePropertyField;
    const layout = try dataset.readStringAttributeAlloc(allocator, "layout");
    defer allocator.free(layout);
    if (!std.mem.eql(u8, layout, expected_layout) or try dataset.readAttribute(u32, "field_version") != 1)
        return error.IncompatiblePropertyField;
    return false;
}

fn createParticleDataset(
    comptime T: type,
    file: *hdf5.File,
    name: []const u8,
    frame_count: u64,
    particle_count: u64,
) !hdf5.Dataset {
    var dataset = try file.createDataset(T, name, &.{ frame_count, particle_count }, .{ .chunk_shape = &.{ 1, particle_count } });
    errdefer dataset.deinit();
    try dataset.writeStringAttribute("layout", "frame,particle");
    try dataset.writeAttribute(u32, "field_version", 1);
    return dataset;
}

fn createPolarDataset(file: *hdf5.File, frame_count: u64, particle_count: u64) !hdf5.Dataset {
    var dataset = try file.createDataset(f32, "polar_cylindrical", &.{ 3, frame_count, particle_count }, .{ .chunk_shape = &.{ 1, 1, particle_count } });
    errdefer dataset.deinit();
    try dataset.writeStringAttribute("layout", "component,frame,particle");
    try dataset.writeStringAttribute("components", "x,r,theta");
    try dataset.writeAttribute(u32, "field_version", 1);
    return dataset;
}

fn writeCylindricalFrameProperty(
    file: *hdf5.File,
    name: []const u8,
    frame_count: u64,
    values: properties.CylindricalFrameValues.Slice,
    timestep: ?f64,
) !void {
    var dataset = try file.createDataset(f32, name, &.{ 3, frame_count }, .{ .chunk_shape = &.{ 1, frame_count } });
    defer dataset.deinit();
    try dataset.writeStringAttribute("components", "x,theta,r");
    try dataset.writeStringAttribute("layout", "component,frame");
    if (timestep) |value| {
        try dataset.writeAttribute(f64, "timestep", value);
        try dataset.writeStringAttribute("time_basis", "simulation_step*timestep");
    }
    try dataset.writeHyperslab(f32, &.{ 0, 0 }, &.{ 1, frame_count }, values.items(.x));
    try dataset.writeHyperslab(f32, &.{ 1, 0 }, &.{ 1, frame_count }, values.items(.theta));
    try dataset.writeHyperslab(f32, &.{ 2, 0 }, &.{ 1, frame_count }, values.items(.r));
}

fn writeComProperties(
    allocator: std.mem.Allocator,
    output_dir: []const u8,
    ranges: []const ShardRange,
    particle_count_u64: u64,
    lx: f64,
    timestep: f64,
) !usize {
    var properties_missing = false;
    for (ranges, 0..) |_, shard_index| {
        const path = try shardPath(allocator, output_dir, shard_index);
        defer allocator.free(path);
        var file = try hdf5.File.openPath(allocator, path, .read_only);
        defer file.deinit();
        if (!try file.objectExists("com_unwrapped") or !try file.objectExists("com_velocity_unwrapped")) {
            properties_missing = true;
        }
    }
    if (!properties_missing) return 0;

    const particle_count = std.math.cast(usize, particle_count_u64) orelse return error.SizeOverflow;
    const total_frames_u64 = ranges[ranges.len - 1].stop;
    const total_frames = std.math.cast(usize, total_frames_u64) orelse return error.SizeOverflow;
    var centers: properties.CylindricalFrameValues = .empty;
    defer centers.deinit(allocator);
    try centers.resize(allocator, total_frames);
    var velocities: properties.CylindricalFrameValues = .empty;
    defer velocities.deinit(allocator);
    try velocities.resize(allocator, total_frames);
    const steps = try allocator.alloc(u64, total_frames);
    defer allocator.free(steps);
    var workspace = try properties.com.Workspace.init(allocator, particle_count);
    defer workspace.deinit();

    for (ranges, 0..) |range, shard_index| {
        const path = try shardPath(allocator, output_dir, shard_index);
        defer allocator.free(path);
        var file = try hdf5.File.openPath(allocator, path, .read_only);
        defer file.deinit();
        const shard_frames_u64 = range.stop - range.start;
        const shard_frames = std.math.cast(usize, shard_frames_u64) orelse return error.SizeOverflow;
        const coordinate_count = std.math.mul(usize, shard_frames, particle_count) catch return error.SizeOverflow;
        var coordinates: CylindricalCoordinates = .empty;
        defer coordinates.deinit(allocator);
        try coordinates.resize(allocator, coordinate_count);
        var coordinate_dataset = try file.openDataset("coords");
        defer coordinate_dataset.deinit();
        const coordinate_slices = coordinates.slice();
        try coordinate_dataset.readHyperslab(f32, &.{ 0, 0, 0 }, &.{ 1, shard_frames_u64, particle_count_u64 }, coordinate_slices.items(.x));
        try coordinate_dataset.readHyperslab(f32, &.{ 1, 0, 0 }, &.{ 1, shard_frames_u64, particle_count_u64 }, coordinate_slices.items(.theta));
        try coordinate_dataset.readHyperslab(f32, &.{ 2, 0, 0 }, &.{ 1, shard_frames_u64, particle_count_u64 }, coordinate_slices.items(.r));
        const start = std.math.cast(usize, range.start) orelse return error.SizeOverflow;
        var step_dataset = try file.openDataset("step");
        defer step_dataset.deinit();
        try step_dataset.readAll(u64, steps[start..][0..shard_frames]);
        try properties.com_unwrapped(
            &workspace,
            coordinates.slice(),
            particle_count,
            lx,
            centers.slice().subslice(start, shard_frames),
        );
    }
    try properties.com_velocity_unwrapped(centers.slice(), steps, timestep, velocities.slice());

    var written: usize = 0;
    for (ranges, 0..) |range, shard_index| {
        const path = try shardPath(allocator, output_dir, shard_index);
        defer allocator.free(path);
        var file = try hdf5.File.openPath(allocator, path, .read_write);
        defer file.deinit();
        const shard_frames_u64 = range.stop - range.start;
        const shard_frames = std.math.cast(usize, shard_frames_u64) orelse return error.SizeOverflow;
        const start = std.math.cast(usize, range.start) orelse return error.SizeOverflow;
        if (!try file.objectExists("com_unwrapped")) {
            try writeCylindricalFrameProperty(
                &file,
                "com_unwrapped",
                shard_frames_u64,
                centers.slice().subslice(start, shard_frames),
                null,
            );
            written += 1;
        }
        if (!try file.objectExists("com_velocity_unwrapped")) {
            try writeCylindricalFrameProperty(
                &file,
                "com_velocity_unwrapped",
                shard_frames_u64,
                velocities.slice().subslice(start, shard_frames),
                timestep,
            );
            written += 1;
        }
        try file.flush();
    }
    return written;
}

fn shardPath(allocator: std.mem.Allocator, output_dir: []const u8, shard_index: usize) ![]u8 {
    const name = try std.fmt.allocPrint(allocator, "frames_{d:0>6}.h5", .{shard_index});
    defer allocator.free(name);
    return std.fs.path.join(allocator, &.{ output_dir, name });
}

fn deinterleaveCartesian(values: []const f32, positions: CartesianPositions.Slice) void {
    std.debug.assert(values.len == positions.len * 3);
    const x = positions.items(.x);
    const y = positions.items(.y);
    const z = positions.items(.z);
    for (0..positions.len) |particle| {
        x[particle] = values[particle * 3];
        y[particle] = values[particle * 3 + 1];
        z[particle] = values[particle * 3 + 2];
    }
}

fn deinterleaveQuaternions(values: []const f32, orientations: Quaternions.Slice) void {
    std.debug.assert(values.len == orientations.len * 4);
    const w = orientations.items(.w);
    const x = orientations.items(.x);
    const y = orientations.items(.y);
    const z = orientations.items(.z);
    for (0..orientations.len) |particle| {
        w[particle] = values[particle * 4];
        x[particle] = values[particle * 4 + 1];
        y[particle] = values[particle * 4 + 2];
        z[particle] = values[particle * 4 + 3];
    }
}

fn inspectMissingFields(
    state: *ShardState,
    allocator: std.mem.Allocator,
    path: []const u8,
    range: ShardRange,
) !MissingFields {
    if (state.serialize_hdf5) while (!state.hdf5_mutex.tryLock()) std.atomic.spinLoopHint();
    defer if (state.serialize_hdf5) state.hdf5_mutex.unlock();
    var file = try hdf5.File.openPath(allocator, path, .read_only);
    defer file.deinit();
    const coords_exists = try file.objectExists("coords");
    var coords_compatible = false;
    if (coords_exists) {
        var dataset = try file.openDataset("coords");
        defer dataset.deinit();
        const shape = try dataset.shapeAlloc(allocator);
        defer allocator.free(shape);
        const expected_shape = [_]u64{ 3, range.stop - range.start, state.particle_count };
        const layout_matches = if (try dataset.attributeExists("layout")) blk: {
            const layout = try dataset.readStringAttributeAlloc(allocator, "layout");
            defer allocator.free(layout);
            break :blk std.mem.eql(u8, layout, "component,frame,particle");
        } else false;
        coords_compatible = std.mem.eql(u64, shape, &expected_shape) and
            layout_matches and try dataset.dtype() == .f32;
    }
    return .{
        .frame_index = !try file.objectExists("frame_index"),
        .step = !try file.objectExists("step"),
        .box = !try file.objectExists("box"),
        .coords = !coords_compatible,
        .replace_coords = coords_exists and !coords_compatible,
    };
}
