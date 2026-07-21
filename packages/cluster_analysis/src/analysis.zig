//! Dataset-level structural cluster orchestration.

const std = @import("std");
const input = @import("input/root.zig");
const Options = @import("options.zig").Options;
const Result = @import("result.zig").Result;
const schema = @import("schema.zig");

pub fn analyze(
    allocator: std.mem.Allocator,
    dataset_input: input.DatasetInput,
    options: Options,
) !Result {
    try schema.validateOptions(options);
    var dataset = try input.Dataset.open(allocator, dataset_input);
    defer dataset.deinit();
    _ = try schema.readGeometry(&dataset.static);
    var reference: ?schema.FrameSchema = null;
    for (dataset.shards) |*shard| {
        const candidate = try schema.inspectFrameSchema(&shard.reader);
        if (reference) |value| try schema.requireCompatibleFrames(value, candidate) else reference = candidate;
    }

    // TODO: stream selected frames, project cylinder coordinates, and call the
    // structural kernel without retaining whole-trajectory decoded arrays.
    return error.NotImplemented;
}
