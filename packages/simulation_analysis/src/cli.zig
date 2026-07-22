//! Command-line parsing for the package-local converter executable.

const std = @import("std");
const Options = @import("options.zig").Options;
const WriteMode = @import("options.zig").WriteMode;

/// Parse borrowed argument strings into conversion options.
pub fn parseArgs(allocator: std.mem.Allocator, arguments: []const []const u8) !Options {
    _ = allocator;
    var input_path: ?[]const u8 = null;
    var output_dir: ?[]const u8 = null;
    var worker_count: ?usize = null;
    var target_shard_bytes: usize = 256 * 1024 * 1024;
    var timestep: f64 = 1.0e-6;
    var write_mode: WriteMode = .create;
    var mode_set = false;
    var dry_run = false;
    var index: usize = 1;
    while (index < arguments.len) : (index += 1) {
        const argument = arguments[index];
        if (std.mem.eql(u8, argument, "--input")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            input_path = arguments[index];
        } else if (std.mem.eql(u8, argument, "--output-dir")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            output_dir = arguments[index];
        } else if (std.mem.eql(u8, argument, "--workers")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            worker_count = try std.fmt.parseInt(usize, arguments[index], 10);
        } else if (std.mem.eql(u8, argument, "--target-shard-mib")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            const mib = try std.fmt.parseInt(usize, arguments[index], 10);
            target_shard_bytes = std.math.mul(usize, mib, 1024 * 1024) catch return error.SizeOverflow;
        } else if (std.mem.eql(u8, argument, "--timestep")) {
            index += 1;
            if (index >= arguments.len) return error.MissingArgumentValue;
            timestep = try std.fmt.parseFloat(f64, arguments[index]);
        } else if (std.mem.eql(u8, argument, "--update")) {
            if (mode_set) return error.ConflictingWriteModes;
            write_mode = .update;
            mode_set = true;
        } else if (std.mem.eql(u8, argument, "--overwrite")) {
            if (mode_set) return error.ConflictingWriteModes;
            write_mode = .overwrite;
            mode_set = true;
        } else if (std.mem.eql(u8, argument, "--dry-run")) {
            dry_run = true;
        } else {
            return error.UnknownArgument;
        }
    }
    return .{
        .input_path = input_path orelse return error.MissingInput,
        .output_dir = output_dir orelse return error.MissingOutputDirectory,
        .worker_count = worker_count,
        .target_shard_bytes = target_shard_bytes,
        .timestep = timestep,
        .write_mode = write_mode,
        .dry_run = dry_run,
    };
}
