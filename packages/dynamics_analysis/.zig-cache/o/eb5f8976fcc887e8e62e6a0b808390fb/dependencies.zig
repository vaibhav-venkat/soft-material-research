pub const packages = struct {
    pub const @"../safetensors" = struct {
        pub const build_root = "/Users/vaibhavvenkat/structures_and_functions/packages/dynamics_analysis/../safetensors";
        pub const build_zig = @import("../safetensors");
        pub const deps: []const struct { []const u8, []const u8 } = &.{
        };
    };
};

pub const root_deps: []const struct { []const u8, []const u8 } = &.{
    .{ "safetensors", "../safetensors" },
};
