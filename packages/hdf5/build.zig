const std = @import("std");

fn configureHdf5(
    b: *std.Build,
    module: *std.Build.Module,
    hdf5_path: []const u8,
) void {
    const include_path = std.fs.path.join(b.allocator, &.{ hdf5_path, "include" }) catch @panic("out of memory");
    const library_path = std.fs.path.join(b.allocator, &.{ hdf5_path, "lib" }) catch @panic("out of memory");
    module.addIncludePath(.{ .cwd_relative = include_path });
    module.addLibraryPath(.{ .cwd_relative = library_path });
    module.linkSystemLibrary("hdf5", .{});
    module.link_libc = true;
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const hdf5_path = b.option([]const u8, "hdf5-path", "HDF5 installation prefix") orelse
        @panic("missing required -Dhdf5-path=/path/to/hdf5-prefix");
    const include_path = std.fs.path.join(b.allocator, &.{ hdf5_path, "include" }) catch @panic("out of memory");

    const translate = b.addTranslateC(.{
        .root_source_file = b.path("src/hdf5_shim.h"),
        .target = target,
        .optimize = optimize,
    });
    translate.addIncludePath(.{ .cwd_relative = include_path });
    const hdf5_c = translate.createModule();

    const module = b.addModule("hdf5", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    module.addImport("hdf5_c", hdf5_c);
    configureHdf5(b, module, hdf5_path);

    const test_module = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_module.addImport("hdf5", module);
    configureHdf5(b, test_module, hdf5_path);
    const module_tests = b.addTest(.{ .root_module = test_module });

    const check_step = b.step("check", "Compile the HDF5 wrapper and tests");
    check_step.dependOn(&module_tests.step);
    const run_tests = b.addRunArtifact(module_tests);
    const test_step = b.step("test", "Run the HDF5 wrapper tests");
    test_step.dependOn(&run_tests.step);
}
