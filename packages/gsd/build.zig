const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const translate = b.addTranslateC(.{
        .root_source_file = b.path("src/vendor/gsd.h"),
        .target = target,
        .optimize = optimize,
    });
    const gsd_c = translate.createModule();

    const module = b.addModule("gsd", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    module.addImport("gsd_c", gsd_c);
    module.addCSourceFile(.{ .file = b.path("src/vendor/gsd.c") });
    module.addIncludePath(b.path("src/vendor"));
    module.link_libc = true;

    const test_module = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_module.addImport("gsd", module);
    const module_tests = b.addTest(.{ .root_module = test_module });

    const check_step = b.step("check", "Compile the GSD wrapper and tests");
    check_step.dependOn(&module_tests.step);
    const run_tests = b.addRunArtifact(module_tests);
    const test_step = b.step("test", "Run the GSD wrapper tests");
    test_step.dependOn(&run_tests.step);
}
