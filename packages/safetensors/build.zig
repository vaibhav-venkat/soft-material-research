const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const module = b.addModule("safetensors", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const test_module = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_module.addImport("safetensors", module);

    const module_tests = b.addTest(.{ .root_module = test_module });
    const check_step = b.step("check", "Compile the safetensors package and tests");
    check_step.dependOn(&module_tests.step);

    const run_tests = b.addRunArtifact(module_tests);
    const test_step = b.step("test", "Run safetensors reader tests");
    test_step.dependOn(&run_tests.step);
}
