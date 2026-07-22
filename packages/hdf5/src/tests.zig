const std = @import("std");
const hdf5 = @import("hdf5");

fn fixturePath(allocator: std.mem.Allocator, temporary: std.testing.TmpDir) ![]u8 {
    return std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/data.h5", .{temporary.sub_path});
}

fn expectNumericRoundTrip(
    file: *hdf5.File,
    comptime T: type,
    name: []const u8,
    value: T,
) !void {
    var dataset = try file.createDataset(T, name, &.{1}, .{});
    defer dataset.deinit();
    try dataset.writeAll(T, &.{value});
    var result: [1]T = undefined;
    try dataset.readAll(T, &result);
    try std.testing.expectEqual(value, result[0]);
    try std.testing.expectEqual(hdf5.Dtype.of(T), try dataset.dtype());
}

test "chunked datasets round trip whole arrays and hyperslabs" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try fixturePath(std.testing.allocator, temporary);
    defer std.testing.allocator.free(path);

    var file = try hdf5.File.create(std.testing.allocator, path, .exclusive);
    defer file.deinit();
    var group = try file.createGroup("/frames/nested");
    group.deinit();
    var dataset = try file.createDataset(f32, "/frames/nested/coords", &.{ 3, 4 }, .{
        .chunk_shape = &.{ 1, 4 },
    });
    defer dataset.deinit();
    const initial = [_]f32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    try dataset.writeAll(f32, &initial);
    try dataset.writeHyperslab(f32, &.{ 1, 1 }, &.{ 1, 2 }, &.{ 40, 50 });
    try dataset.writeAttribute(u32, "field_version", 3);
    try dataset.writeAttribute(u32, "field_version", 4);
    try dataset.writeStringAttribute("components", "x,theta,r");
    try dataset.writeStringAttribute("components", "x,theta,r");
    try file.writeAttribute(u64, "frame_count", 3);
    try file.writeStringAttribute("schema", "simulation_analysis.frames.v1");
    try file.flush();

    var selected: [4]f32 = undefined;
    try dataset.readHyperslab(f32, &.{ 1, 0 }, &.{ 1, 4 }, &selected);
    try std.testing.expectEqualSlices(f32, &.{ 4, 40, 50, 7 }, &selected);
    try std.testing.expect(try dataset.attributeExists("field_version"));
    try std.testing.expect(!(try dataset.attributeExists("missing")));
    try std.testing.expectEqual(@as(u32, 4), try dataset.readAttribute(u32, "field_version"));
    const components = try dataset.readStringAttributeAlloc(std.testing.allocator, "components");
    defer std.testing.allocator.free(components);
    try std.testing.expectEqualStrings("x,theta,r", components);
    const shape = try dataset.shapeAlloc(std.testing.allocator);
    defer std.testing.allocator.free(shape);
    try std.testing.expectEqualSlices(u64, &.{ 3, 4 }, shape);
    const chunk = (try dataset.chunkShapeAlloc(std.testing.allocator)).?;
    defer std.testing.allocator.free(chunk);
    try std.testing.expectEqualSlices(u64, &.{ 1, 4 }, chunk);
    try std.testing.expectEqual(hdf5.Dtype.f32, try dataset.dtype());
}

test "fixed datasets may represent an empty sample collection" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const file_path = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}/empty.h5", .{temporary.sub_path});
    defer std.testing.allocator.free(file_path);
    var file = try hdf5.File.create(std.testing.allocator, file_path, .exclusive);
    defer file.deinit();
    var dataset = try file.createDataset(u32, "samples", &.{0}, .{});
    defer dataset.deinit();
    try dataset.writeAll(u32, &.{});
    const shape = try dataset.shapeAlloc(std.testing.allocator);
    defer std.testing.allocator.free(shape);
    try std.testing.expectEqualSlices(u64, &.{0}, shape);
}

test "files reopen without loading datasets and preserve metadata" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try fixturePath(std.testing.allocator, temporary);
    defer std.testing.allocator.free(path);
    var writer = try hdf5.File.create(std.testing.allocator, path, .exclusive);
    var values = try writer.createDataset(u64, "steps", &.{4}, .{});
    try values.writeAll(u64, &.{ 10, 20, 30, 40 });
    values.deinit();
    try writer.writeStringAttribute("application", "zig-test");
    try writer.close();

    var reader = try hdf5.File.openPath(std.testing.allocator, path, .read_only);
    defer reader.deinit();
    try std.testing.expect(try reader.objectExists("steps"));
    try std.testing.expect(!(try reader.objectExists("missing")));
    var opened = try reader.openDataset("steps");
    defer opened.deinit();
    var middle: [2]u64 = undefined;
    try opened.readHyperslab(u64, &.{1}, &.{2}, &middle);
    try std.testing.expectEqualSlices(u64, &.{ 20, 30 }, &middle);
    const application = try reader.readStringAttributeAlloc(std.testing.allocator, "application");
    defer std.testing.allocator.free(application);
    try std.testing.expectEqualStrings("zig-test", application);
}

test "all supported numeric HDF5 types round trip" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try fixturePath(std.testing.allocator, temporary);
    defer std.testing.allocator.free(path);
    var file = try hdf5.File.create(std.testing.allocator, path, .exclusive);
    defer file.deinit();
    try expectNumericRoundTrip(&file, u8, "u8", 8);
    try expectNumericRoundTrip(&file, u16, "u16", 16);
    try expectNumericRoundTrip(&file, u32, "u32", 32);
    try expectNumericRoundTrip(&file, u64, "u64", 64);
    try expectNumericRoundTrip(&file, i8, "i8", -8);
    try expectNumericRoundTrip(&file, i16, "i16", -16);
    try expectNumericRoundTrip(&file, i32, "i32", -32);
    try expectNumericRoundTrip(&file, i64, "i64", -64);
    try expectNumericRoundTrip(&file, f32, "f32", 3.25);
    try expectNumericRoundTrip(&file, f64, "f64", 6.5);
}

test "staging links move and delete without touching established datasets" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try fixturePath(std.testing.allocator, temporary);
    defer std.testing.allocator.free(path);
    var file = try hdf5.File.create(std.testing.allocator, path, .exclusive);
    defer file.deinit();
    var group = try file.createGroup("/_staging");
    group.deinit();
    var established = try file.createDataset(u32, "step", &.{2}, .{});
    try established.writeAll(u32, &.{ 1, 2 });
    established.deinit();
    var staged = try file.createDataset(f32, "/_staging/coords", &.{2}, .{});
    try staged.writeAll(f32, &.{ 3, 4 });
    staged.deinit();
    try file.moveLink("/_staging/coords", "/coords");
    try std.testing.expect(try file.objectExists("coords"));
    try std.testing.expect(try file.objectExists("step"));
    try file.deleteLink("coords");
    try std.testing.expect(!(try file.objectExists("coords")));
    try std.testing.expect(try file.objectExists("step"));
}

test "invalid shapes, selections, types, and exclusive creation fail" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try fixturePath(std.testing.allocator, temporary);
    defer std.testing.allocator.free(path);
    var file = try hdf5.File.create(std.testing.allocator, path, .exclusive);
    defer file.deinit();
    try std.testing.expectError(error.InvalidRank, file.createDataset(f32, "scalar", &.{}, .{}));
    try std.testing.expectError(
        error.InvalidShape,
        file.createDataset(f32, "bad_chunk", &.{2}, .{ .chunk_shape = &.{3} }),
    );
    var dataset = try file.createDataset(u32, "values", &.{ 2, 2 }, .{});
    defer dataset.deinit();
    try std.testing.expectError(error.BufferSizeMismatch, dataset.writeAll(u32, &.{1}));
    var wrong_type: [4]f32 = undefined;
    try std.testing.expectError(error.DtypeMismatch, dataset.readAll(f32, &wrong_type));
    var one: [1]u32 = undefined;
    try std.testing.expectError(
        error.SelectionOutOfBounds,
        dataset.readHyperslab(u32, &.{ 2, 0 }, &.{ 1, 1 }, &one),
    );
    try std.testing.expectError(error.Hdf5Failure, hdf5.File.create(std.testing.allocator, path, .exclusive));
}

test "linked library reports its thread safety configuration" {
    _ = try hdf5.isLibraryThreadSafe();
}
