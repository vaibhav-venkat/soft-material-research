const std = @import("std");
const simulation_analysis = @import("simulation_analysis");

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const arguments = try init.minimal.args.toSlice(allocator);
    const options = try simulation_analysis.cli.parseArgs(allocator, arguments);
    const summary = try simulation_analysis.run(allocator, options);
    std.debug.print("frames={d} shards={d} fields_written={d}\n", .{
        summary.frame_count,
        summary.shard_count,
        summary.fields_written,
    });
}
