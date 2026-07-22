//! GSD-to-HDF5 simulation conversion scaffold.

pub const analysis = @import("analysis.zig");
pub const cli = @import("cli.zig");
pub const schema = @import("schema.zig");
pub const coordinates = @import("coordinates.zig");
pub const properties = @import("properties/root.zig");

pub const Options = @import("options.zig").Options;
pub const WriteMode = @import("options.zig").WriteMode;
pub const ShardRange = @import("analysis.zig").ShardRange;
pub const Summary = @import("analysis.zig").Summary;
pub const CartesianPosition = @import("analysis.zig").CartesianPosition;
pub const CylindricalCoordinate = @import("analysis.zig").CylindricalCoordinate;
pub const CartesianPositions = @import("analysis.zig").CartesianPositions;
pub const CylindricalCoordinates = @import("analysis.zig").CylindricalCoordinates;
pub const planShards = @import("analysis.zig").planShards;
pub const transformCylindrical = @import("analysis.zig").transformCylindrical;
pub const run = @import("analysis.zig").run;
