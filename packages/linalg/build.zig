const std = @import("std");

fn addLibraryPathIfPresent(b: *std.Build, module: *std.Build.Module, path: []const u8) void {
    std.Io.Dir.accessAbsolute(b.graph.io, path, .{}) catch return;
    module.addLibraryPath(.{ .cwd_relative = path });
}

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
    if (is_cuda and target.result.os.tag != .linux) {
        @panic("the cuda backend requires a Linux target");
    }

    const build_options = b.addOptions();
    build_options.addOption([]const u8, "backend", backend);

    const module = b.addModule("linalg", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    module.addOptions("linalg_build_options", build_options);
    module.link_libc = true;

    if (is_accelerate) {
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
        translate_accelerate.addSystemFrameworkPath(.{ .cwd_relative = accelerate_frameworks });
        module.addImport("accelerate_c", translate_accelerate.createModule());
        module.addCSourceFile(.{ .file = b.path("src/backend/accelerate_shim.c") });
        module.linkFramework("Accelerate", .{});
    } else {
        const cuda_path = b.option([]const u8, "cuda-path", "Required CUDA 12.9+ toolkit root") orelse
            @panic("the cuda backend requires -Dcuda-path=/path/to/cuda-12.9-or-newer");
        const cuda_include = std.fs.path.join(b.allocator, &.{ cuda_path, "include" }) catch @panic("out of memory");
        const cuda_lib = std.fs.path.join(b.allocator, &.{ cuda_path, "lib" }) catch @panic("out of memory");
        const cuda_lib64 = std.fs.path.join(b.allocator, &.{ cuda_path, "lib64" }) catch @panic("out of memory");
        const cuda_target_lib = std.fs.path.join(b.allocator, &.{ cuda_path, "targets/x86_64-linux/lib" }) catch @panic("out of memory");
        const translate_cuda = b.addTranslateC(.{
            .root_source_file = b.path("src/backend/cuda.h"),
            .target = target,
            .optimize = optimize,
        });
        translate_cuda.addIncludePath(.{ .cwd_relative = cuda_include });
        const cuda_c = translate_cuda.createModule();
        cuda_c.addIncludePath(.{ .cwd_relative = cuda_include });
        module.addImport("cuda_c", cuda_c);
        module.addIncludePath(.{ .cwd_relative = cuda_include });
        addLibraryPathIfPresent(b, module, cuda_lib);
        addLibraryPathIfPresent(b, module, cuda_lib64);
        addLibraryPathIfPresent(b, module, cuda_target_lib);
        module.linkSystemLibrary("cudart", .{});
        module.linkSystemLibrary("cublas", .{});
        module.linkSystemLibrary("cusolver", .{});
    }

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
