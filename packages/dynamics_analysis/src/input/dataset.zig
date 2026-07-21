//! Dataset and static-safetensor descriptors.

pub const StaticInput = struct {
    path: []const u8,
};

pub const DatasetInput = struct {
    static: StaticInput,
    shard_paths: []const []const u8,
};
