//! GSD-to-HDF5 simulation conversion scaffold.

pub const analysis = @import("analysis.zig");
pub const cli = @import("cli.zig");
pub const schema = @import("schema.zig");
pub const data_structures = @import("data_structures/root.zig");
pub const properties = @import("properties/root.zig");

pub const Options = @import("options.zig").Options;
pub const WriteMode = @import("options.zig").WriteMode;
pub const ShardRange = @import("analysis.zig").ShardRange;
pub const Summary = @import("analysis.zig").Summary;
pub const planShards = @import("analysis.zig").planShards;
pub const transformCylindrical = @import("analysis.zig").transformCylindrical;
pub const run = @import("analysis.zig").run;
