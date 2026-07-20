const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const backend = b.option([]const u8, "backend", "Required backend: accelerate or cuda") orelse
        @panic("missing required -Dbackend=accelerate|cuda");

    const is_accelerate = std.mem.eql(u8, backend, "accelerate");
    const is_cuda = std.mem.eql(u8, backend, "cuda");
    if (!is_accelerate and !is_cuda) {
        @panic("unsupported linalg backend; use -Dbackend=accelerate|cuda");
    }
    if (is_accelerate and target.result.os.tag != .macos) {
        @panic("the accelerate backend requires a macOS target");
    }
    if (is_cuda) {
        @panic("the cuda backend contract exists but is not implemented in this worktree");
    }

    const build_options = b.addOptions();
    build_options.addOption([]const u8, "backend", backend);

    const macos_sdk = b.option(
        []const u8,
        "macos-sdk",
        "Path to the macOS SDK used to translate Accelerate headers",
    ) orelse "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk";
    const accelerate_frameworks = std.fs.path.join(b.allocator, &.{
        macos_sdk,
        "System/Library/Frameworks/Accelerate.framework/Frameworks",
    }) catch @panic("out of memory while constructing the Accelerate framework path");
    const translate_accelerate = b.addTranslateC(.{
        .root_source_file = b.path("src/backend/accelerate.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_accelerate.defineCMacro("ACCELERATE_NEW_LAPACK", "1");
    translate_accelerate.addSystemFrameworkPath(b.graph.cwdRelativePath(accelerate_frameworks));
    const accelerate_c = translate_accelerate.createModule();

    const module = b.addModule("linalg", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    module.addOptions("linalg_build_options", build_options);
    module.addImport("accelerate_c", accelerate_c);
    module.addCSourceFile(.{ .file = b.path("src/backend/accelerate_shim.c") });
    module.link_libc = true;
    module.linkFramework("Accelerate", .{});

    const test_module = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_module.addImport("linalg", module);
    const module_tests = b.addTest(.{ .root_module = test_module });
    const check_step = b.step("check", "Compile the linalg package and tests");
    check_step.dependOn(&module_tests.step);

    const run_tests = b.addRunArtifact(module_tests);
    const test_step = b.step("test", "Run backend-neutral linalg tests");
    test_step.dependOn(&run_tests.step);
}
