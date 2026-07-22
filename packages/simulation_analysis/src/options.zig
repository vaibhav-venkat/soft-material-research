//! Public conversion controls.

pub const WriteMode = enum {
    create,
    update,
    overwrite,
};

pub const Options = struct {
    input_path: []const u8,
    output_dir: []const u8,
    worker_count: ?usize = null,
    target_shard_bytes: usize = 256 * 1024 * 1024,
    write_mode: WriteMode = .create,
    dry_run: bool = false,
};
