const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const backend = b.option([]const u8, "backend", "Linalg backend: accelerate or cuda") orelse
        if (target.result.os.tag == .macos) "accelerate" else
        @panic("missing required -Dbackend=accelerate|cuda");
    const safetensors_dependency = b.dependency("safetensors", .{
        .target = target,
        .optimize = optimize,
    });
    const linalg_dependency = if (std.mem.eql(u8, backend, "cuda")) blk: {
        const cuda_path = b.option([]const u8, "cuda-path", "CUDA toolkit root") orelse
            @panic("the CUDA backend requires -Dcuda-path=/path/to/cuda");
        break :blk b.dependency("linalg", .{
            .target = target,
            .optimize = optimize,
            .backend = backend,
            .@"cuda-path" = cuda_path,
        });
    } else b.dependency("linalg", .{
        .target = target,
        .optimize = optimize,
        .backend = backend,
    });

    const module = b.addModule("dynamics_analysis", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    module.addImport("safetensors", safetensors_dependency.module("safetensors"));
    module.addImport("linalg", linalg_dependency.module("linalg"));
    module.link_libc = true;

    const library = b.addLibrary(.{
        .name = "dynamics_analysis",
        .root_module = module,
        .linkage = .dynamic,
    });
    b.installArtifact(library);

    const test_module = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_module.addImport("dynamics_analysis", module);
    const module_tests = b.addTest(.{ .root_module = test_module });
    const check_step = b.step("check", "Compile the dynamics analysis package");
    check_step.dependOn(&library.step);
    check_step.dependOn(&module_tests.step);

    const run_tests = b.addRunArtifact(module_tests);
    const test_step = b.step("test", "Run dynamics analysis package tests");
    test_step.dependOn(&run_tests.step);
}
