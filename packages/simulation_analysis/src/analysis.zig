//! Sharded GSD-to-HDF5 storage for simulation coordinates and basic metadata.

const std = @import("std");
const gsd = @import("gsd");
const hdf5 = @import("hdf5");
const Options = @import("options.zig").Options;
const schema = @import("schema.zig");
const coordinate_types = @import("coordinates.zig");
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

/// Partition frames into contiguous ranges sized from the coordinate payload.
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
    const values_per_frame = std.math.mul(usize, particles, 3) catch return error.SizeOverflow;
    const bytes_per_frame = std.math.mul(usize, values_per_frame, @sizeOf(f32)) catch return error.SizeOverflow;
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
    if (positions.len != coordinates.len) return error.BufferSizeMismatch;
    const lane_count = std.simd.suggestVectorLength(f32) orelse 4;
    const Vector = @Vector(lane_count, f32);
    var particle: usize = 0;
    const count = positions.len;
    const input_x = positions.items(.x);
    const input_y = positions.items(.y);
    const input_z = positions.items(.z);
    const output_x = coordinates.items(.x);
    const output_theta = coordinates.items(.theta);
    const output_r = coordinates.items(.r);
    while (particle + lane_count <= count) : (particle += lane_count) {
        const x: Vector = input_x[particle..][0..lane_count].*;
        const y: Vector = input_y[particle..][0..lane_count].*;
        const z: Vector = input_z[particle..][0..lane_count].*;
        const radius = @sqrt(y * y + z * z);
        const angle = atan2_simd(lane_count, y, z);
        output_x[particle..][0..lane_count].* = x;
        output_theta[particle..][0..lane_count].* = angle;
        output_r[particle..][0..lane_count].* = radius;
    }
    while (particle < count) : (particle += 1) {
        const y = input_y[particle];
        const z = input_z[particle];
        output_x[particle] = input_x[particle];
        output_theta[particle] = normalizedAtan2(y, z);
        output_r[particle] = @sqrt(y * y + z * z);
    }
}

pub fn atan2_simd(comptime N: comptime_int, y: @Vector(N, f32), x: @Vector(N, f32)) @Vector(N, f32) {
    const V = @Vector(N, f32);
    const zero: V = @splat(0.0);
    const one: V = @splat(1.0);
    const pi: V = @splat(std.math.pi);
    const two_pi: V = @splat(2.0 * std.math.pi);
    const half_pi: V = @splat(std.math.pi / 2.0);
    const ax = @abs(x);
    const ay = @abs(y);

    const swap = ay > ax;
    const lo = @select(f32, swap, ax, ay);
    const hi = @select(f32, swap, ay, ax);

    const is_zero = hi == zero;
    const clamped_hi = @select(f32, is_zero, one, hi);

    const r = lo / clamped_hi;
    const r2 = r * r;
    // atan(r) ~ r * P(r^2)
    var p: V = @splat(0.02084517);
    p = @mulAdd(V, p, r2, @as(V, @splat(-0.08515649)));
    p = @mulAdd(V, p, r2, @as(V, @splat(0.18015942)));
    p = @mulAdd(V, p, r2, @as(V, @splat(-0.33030482)));
    p = @mulAdd(V, p, r2, @as(V, @splat(0.99986633)));

    var angle = r * p;
    angle = @select(f32, swap, half_pi - angle, angle);

    angle = @select(f32, x < zero, pi - angle, angle);
    angle = @select(f32, y < zero, -angle, angle);
    angle = @select(f32, angle < zero, angle + two_pi, angle);
    return @select(f32, is_zero, zero, angle);
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

fn normalizedAtan2(y: f32, z: f32) f32 {
    if (y == 0 and z == 0) return 0;
    const angle = std.math.atan2(y, z);
    return if (angle < 0) angle + 2.0 * std.math.pi else angle;
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
    try file.writeStringAttribute("schema", schema.static_schema);
    try file.writeStringAttribute("coordinate_transform", schema.coordinate_transform);
    try file.writeStringAttribute("source_application", input.header().application);
    try file.writeStringAttribute("source_schema", input.header().schema);
    try file.writeAttribute(u64, "frame_count", frame_count);
    try file.writeAttribute(u64, "particle_count", particle_count);
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
