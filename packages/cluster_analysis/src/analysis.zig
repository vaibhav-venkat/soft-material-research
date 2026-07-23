//! Dataset-level structural cluster orchestration.

const std = @import("std");
const safetensors = @import("safetensors");
const clusters = @import("clusters.zig");
const input = @import("input/root.zig");
const Options = @import("options.zig").Options;
const Result = @import("result.zig").Result;
const schema = @import("schema.zig");

const FrameJob = struct {
    coordinates: safetensors.TensorView,
    psi_real: safetensors.TensorView,
    psi_imaginary: safetensors.TensorView,
    mask: safetensors.TensorView,
    local_frame: usize,
};

const WorkerState = struct {
    allocator: std.mem.Allocator,
    jobs: []const FrameJob,
    next_job: *std.atomic.Value(usize),
    particle_count: usize,
    periods: [2]f64,
    options: Options,
    points: std.ArrayList([2]f64) = .empty,
    offsets: std.ArrayList(usize) = .empty,
    failure: ?anyerror = null,
};

pub fn analyze(
    allocator: std.mem.Allocator,
    dataset_input: input.DatasetInput,
    options: Options,
) !Result {
    try schema.validateOptions(options);
    var dataset = try input.Dataset.open(allocator, dataset_input);
    defer dataset.deinit();
    const geometry = try schema.readGeometry(&dataset.static);

    var reference: ?schema.FrameSchema = null;
    var total_frames: usize = 0;
    for (dataset.shards) |*shard| {
        const candidate = try schema.inspectFrameSchema(&shard.reader);
        if (reference) |value| try schema.requireCompatibleFrames(value, candidate) else reference = candidate;
        total_frames = try std.math.add(usize, total_frames, candidate.frame_count);
    }
    const frame_stop = options.frame_stop orelse total_frames;
    if (options.frame_start >= frame_stop or frame_stop > total_frames) {
        return error.InvalidFrameRange;
    }

    const selected_frame_count = frame_stop - options.frame_start;
    const jobs = try allocator.alloc(FrameJob, selected_frame_count);
    defer allocator.free(jobs);
    var job_index: usize = 0;
    var shard_frame_start: usize = 0;
    for (dataset.shards) |*shard| {
        const coordinates = try shard.reader.tensor("coords");
        const psi_real = try shard.reader.tensor("psi_real");
        const psi_imaginary = try shard.reader.tensor("psi_imag");
        const mask = try shard.reader.tensor("hexatic_shell_mask");
        const shard_frame_stop = try std.math.add(usize, shard_frame_start, coordinates.shape[0]);
        const selected_start = @max(options.frame_start, shard_frame_start);
        const selected_stop = @min(frame_stop, shard_frame_stop);
        if (selected_start < selected_stop) {
            const local_start = selected_start - shard_frame_start;
            const local_stop = selected_stop - shard_frame_start;
            for (local_start..local_stop) |local_frame| {
                jobs[job_index] = .{
                    .coordinates = coordinates,
                    .psi_real = psi_real,
                    .psi_imaginary = psi_imaginary,
                    .mask = mask,
                    .local_frame = local_frame,
                };
                job_index += 1;
            }
        }
        shard_frame_start = shard_frame_stop;
    }
    std.debug.assert(job_index == jobs.len);

    var next_job = std.atomic.Value(usize).init(0);
    const cpu_count = std.Thread.getCpuCount() catch 1;
    const maximum_cluster_workers: usize = 8;
    const worker_count = @max(
        @as(usize, 1),
        @min(maximum_cluster_workers, @min(cpu_count, jobs.len)),
    );
    const states = try allocator.alloc(WorkerState, worker_count);
    defer allocator.free(states);
    for (states) |*state| {
        state.* = .{
            .allocator = allocator,
            .jobs = jobs,
            .next_job = &next_job,
            .particle_count = reference.?.particle_count,
            .periods = .{ geometry.axial_period, geometry.circumference },
            .options = options,
        };
    }
    defer for (states) |*state| {
        state.points.deinit(allocator);
        state.offsets.deinit(allocator);
    };

    const threads = try allocator.alloc(std.Thread, worker_count - 1);
    defer allocator.free(threads);
    var spawned: usize = 0;
    for (threads, 0..) |*thread, index| {
        thread.* = std.Thread.spawn(.{}, analyzeFrames, .{&states[index]}) catch |err| {
            for (threads[0..spawned]) |started| started.join();
            return err;
        };
        spawned += 1;
    }
    analyzeFrames(&states[worker_count - 1]);
    for (threads) |thread| thread.join();
    for (states) |state| if (state.failure) |err| return err;

    var points: std.ArrayList([2]f64) = .empty;
    errdefer points.deinit(allocator);
    var offsets: std.ArrayList(usize) = .empty;
    errdefer offsets.deinit(allocator);
    try offsets.append(allocator, 0);
    for (states) |state| {
        const base = points.items.len;
        try points.appendSlice(allocator, state.points.items);
        for (state.offsets.items[1..]) |offset| {
            try offsets.append(allocator, base + offset);
        }
    }

    return .{
        .points = try points.toOwnedSlice(allocator),
        .offsets = try offsets.toOwnedSlice(allocator),
    };
}

fn analyzeFrames(state: *WorkerState) void {
    const points = state.allocator.alloc([2]f64, state.particle_count) catch |err| {
        state.failure = err;
        return;
    };
    defer state.allocator.free(points);
    const psi6 = state.allocator.alloc([2]f64, state.particle_count) catch |err| {
        state.failure = err;
        return;
    };
    defer state.allocator.free(psi6);
    const eligible = state.allocator.alloc(bool, state.particle_count) catch |err| {
        state.failure = err;
        return;
    };
    defer state.allocator.free(eligible);
    var workspace = clusters.Workspace.init(state.allocator, state.particle_count) catch |err| {
        state.failure = err;
        return;
    };
    defer workspace.deinit(state.allocator);
    state.offsets.append(state.allocator, 0) catch |err| {
        state.failure = err;
        return;
    };

    while (true) {
        const job_index = state.next_job.fetchAdd(1, .monotonic);
        if (job_index >= state.jobs.len) return;
        const job = state.jobs[job_index];
        decodeFrame(
            points,
            psi6,
            eligible,
            job.coordinates,
            job.psi_real,
            job.psi_imaginary,
            job.mask,
            job.local_frame,
            state.periods[1],
        ) catch |err| {
            state.failure = err;
            return;
        };
        clusters.appendStructuralFrameClusters(
            state.allocator,
            &workspace,
            .{
                .points = points,
                .psi6 = psi6,
                .eligible = eligible,
                .periods = state.periods,
            },
            state.options,
            &state.points,
            &state.offsets,
        ) catch |err| {
            state.failure = err;
            return;
        };
    }
}

fn decodeFrame(
    points: [][2]f64,
    psi6: [][2]f64,
    eligible: []bool,
    coordinates: safetensors.TensorView,
    psi_real: safetensors.TensorView,
    psi_imaginary: safetensors.TensorView,
    mask: safetensors.TensorView,
    frame_index: usize,
    circumference: f64,
) !void {
    const coordinate_frame = try coordinates.frame(frame_index);
    switch (coordinate_frame.dtype) {
        .f32 => fillPoints(f32, points, try coordinate_frame.values(f32), circumference),
        .f64 => fillPoints(f64, points, try coordinate_frame.values(f64), circumference),
        else => return error.InvalidFloatDtype,
    }

    const real_frame = try psi_real.frame(frame_index);
    switch (real_frame.dtype) {
        .f32 => fillPsi6Component(f32, psi6, try real_frame.values(f32), 0),
        .f64 => fillPsi6Component(f64, psi6, try real_frame.values(f64), 0),
        else => return error.InvalidFloatDtype,
    }
    const imaginary_frame = try psi_imaginary.frame(frame_index);
    switch (imaginary_frame.dtype) {
        .f32 => fillPsi6Component(f32, psi6, try imaginary_frame.values(f32), 1),
        .f64 => fillPsi6Component(f64, psi6, try imaginary_frame.values(f64), 1),
        else => return error.InvalidFloatDtype,
    }

    const mask_frame = try mask.frame(frame_index);
    if (mask_frame.dtype != .bool or mask_frame.bytes.len != eligible.len) {
        return error.InvalidMaskDtype;
    }
    for (mask_frame.bytes, eligible) |value, *destination| {
        destination.* = value != 0;
    }
}

fn fillPoints(
    comptime T: type,
    points: [][2]f64,
    coordinates: []align(1) const T,
    circumference: f64,
) void {
    const azimuth_scale = circumference / std.math.tau;
    for (points, 0..) |*point, particle| {
        const start = particle * 3;
        point.* = .{
            @as(f64, @floatCast(coordinates[start])),
            @as(f64, @floatCast(coordinates[start + 1])) * azimuth_scale,
        };
    }
}

fn fillPsi6Component(
    comptime T: type,
    psi6: [][2]f64,
    source: []align(1) const T,
    comptime component: usize,
) void {
    for (psi6, source) |*destination, value| {
        destination[component] = @as(f64, @floatCast(value));
    }
}
