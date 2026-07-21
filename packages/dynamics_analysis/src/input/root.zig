//! Safetensor dataset, shard, and schema declarations.

pub const safetensors = @import("safetensors");
pub const DatasetInput = @import("dataset.zig").DatasetInput;
pub const StaticInput = @import("dataset.zig").StaticInput;
pub const ShardSet = @import("dataset.zig").ShardSet;
pub const FrameSchema = @import("shard.zig").FrameSchema;
pub const Shard = @import("shard.zig").Shard;
pub const inspectShard = @import("shard.zig").inspectShard;
