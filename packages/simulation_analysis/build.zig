const std = @import("std");

fn hdf5Prefix(b: *std.Build) []const u8 {
    if (b.option([]const u8, "hdf5-path", "HDF5 installation prefix")) |path| return path;
    if (b.graph.environ_map.get("CONDA_PREFIX")) |path| return path;
    return "../../.pixi/envs/default";
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const hdf5_path = hdf5Prefix(b);
    const gsd_dependency = b.dependency("gsd", .{ .target = target, .optimize = optimize });
    const hdf5_dependency = b.dependency("hdf5", .{
        .target = target,
        .optimize = optimize,
        .@"hdf5-path" = hdf5_path,
    });

    const module = b.addModule("simulation_analysis", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    module.addImport("gsd", gsd_dependency.module("gsd"));
    module.addImport("hdf5", hdf5_dependency.module("hdf5"));
    module.link_libc = true;

    const executable_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    executable_module.addImport("simulation_analysis", module);
    const executable = b.addExecutable(.{
        .name = "simulation-analysis",
        .root_module = executable_module,
    });
    b.installArtifact(executable);

    const test_module = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_module.addImport("simulation_analysis", module);
    test_module.addImport("gsd", gsd_dependency.module("gsd"));
    test_module.addImport("hdf5", hdf5_dependency.module("hdf5"));
    const module_tests = b.addTest(.{ .root_module = test_module });

    const check_step = b.step("check", "Compile simulation-analysis and its red TDD suite");
    check_step.dependOn(&executable.step);
    check_step.dependOn(&module_tests.step);
    const run_tests = b.addRunArtifact(module_tests);
    const test_step = b.step("test", "Run the intentionally red simulation-analysis TDD suite");
    test_step.dependOn(&run_tests.step);
}
