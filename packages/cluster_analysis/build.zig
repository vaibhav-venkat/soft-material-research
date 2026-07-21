const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const safetensors_dependency = b.dependency("safetensors", .{
        .target = target,
        .optimize = optimize,
    });

    const module = b.addModule("cluster_analysis", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    module.addImport("safetensors", safetensors_dependency.module("safetensors"));
    module.link_libc = true;

    const library = b.addLibrary(.{
        .name = "cluster_analysis",
        .root_module = module,
        .linkage = .dynamic,
    });
    b.installArtifact(library);

    const module_tests = b.addTest(.{ .root_module = module });
    const check_step = b.step("check", "Compile the cluster analysis package");
    check_step.dependOn(&library.step);
    check_step.dependOn(&module_tests.step);
}
