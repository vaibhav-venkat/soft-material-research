const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const safetensors_dependency = b.dependency("safetensors", .{
        .target = target,
        .optimize = optimize,
    });
    const translate_kdtree = b.addTranslateC(.{
        .root_source_file = b.path("vendor/kdtree/kdtree.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_kdtree.addIncludePath(b.path("vendor/kdtree"));

    const module = b.addModule("cluster_analysis", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    module.addImport("safetensors", safetensors_dependency.module("safetensors"));
    module.addImport("kdtree_c", translate_kdtree.createModule());
    module.addIncludePath(b.path("vendor/kdtree"));
    module.addCSourceFile(.{
        .file = b.path("vendor/kdtree/kdtree.c"),
        .flags = &.{"-std=c99"},
    });
    module.link_libc = true;
    module.linkSystemLibrary("m", .{});

    const library = b.addLibrary(.{
        .name = "cluster_analysis",
        .root_module = module,
        .linkage = .dynamic,
    });
    b.installArtifact(library);

    const test_module = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_module.addImport("cluster_analysis", module);
    const module_tests = b.addTest(.{ .root_module = test_module });
    const check_step = b.step("check", "Compile the cluster analysis package");
    check_step.dependOn(&library.step);
    check_step.dependOn(&module_tests.step);

    const run_tests = b.addRunArtifact(module_tests);
    const test_step = b.step("test", "Run cluster analysis package tests");
    test_step.dependOn(&run_tests.step);
}
