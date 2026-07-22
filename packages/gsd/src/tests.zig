const std = @import("std");
const gsd = @import("gsd");

fn fixturePath(allocator: std.mem.Allocator, temporary: std.testing.TmpDir) ![]u8 {
    return std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/trajectory.gsd", .{temporary.sub_path});
}

test "create, write, reopen, and partially select chunks" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try fixturePath(std.testing.allocator, temporary);
    defer std.testing.allocator.free(path);

    var writer = try gsd.File.create(
        std.testing.allocator,
        path,
        "zig-test",
        "hoomd",
        gsd.makeVersion(1, 4),
        true,
    );
    const box = [_]f32{ 10, 20, 30, 0, 0, 0 };
    const positions = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const step0 = [_]u64{100};
    try writer.writeChunk(f32, "configuration/box", 6, 1, &box);
    try writer.writeChunk(u64, "configuration/step", 1, 1, &step0);
    try writer.writeChunk(f32, "particles/position", 2, 3, &positions);
    try writer.endFrame();
    const step1 = [_]u64{200};
    try writer.writeChunk(u64, "configuration/step", 1, 1, &step1);
    try writer.endFrame();
    try writer.close();

    var reader = try gsd.File.openRead(std.testing.allocator, path);
    defer reader.deinit();
    try std.testing.expectEqual(@as(u64, 2), reader.frameCount());
    const header = reader.header();
    try std.testing.expectEqualStrings("zig-test", header.application);
    try std.testing.expectEqualStrings("hoomd", header.schema);
    try std.testing.expectEqual(gsd.makeVersion(1, 4), header.schema_version);

    const position_chunk = (try reader.findChunk(0, "particles/position")).?;
    try std.testing.expectEqual(@as(u64, 2), position_chunk.rows);
    try std.testing.expectEqual(@as(u32, 3), position_chunk.columns);
    try std.testing.expectEqual(gsd.Dtype.f32, position_chunk.dtype);
    var selected_positions: [6]f32 = undefined;
    _ = try reader.readChunkInto(f32, 0, "particles/position", &selected_positions);
    try std.testing.expectEqualSlices(f32, &positions, &selected_positions);
    try std.testing.expect((try reader.findChunk(1, "configuration/box")) == null);
}

test "chunk reads reject wrong types and buffer lengths" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try fixturePath(std.testing.allocator, temporary);
    defer std.testing.allocator.free(path);
    var writer = try gsd.File.create(std.testing.allocator, path, "test", "schema", 0, true);
    try writer.writeChunk(u32, "values", 2, 1, &.{ 3, 4 });
    try writer.endFrame();
    try writer.close();

    var reader = try gsd.File.openRead(std.testing.allocator, path);
    defer reader.deinit();
    var wrong_type: [2]f32 = undefined;
    try std.testing.expectError(error.DtypeMismatch, reader.readChunkInto(f32, 0, "values", &wrong_type));
    var too_short: [1]u32 = undefined;
    try std.testing.expectError(error.BufferSizeMismatch, reader.readChunkInto(u32, 0, "values", &too_short));
    try std.testing.expectError(error.ChunkNotFound, reader.requireChunk(0, "missing"));
}

test "all numeric GSD element types round trip through files" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try fixturePath(std.testing.allocator, temporary);
    defer std.testing.allocator.free(path);
    var writer = try gsd.File.create(std.testing.allocator, path, "test", "schema", 0, true);
    try writer.writeChunk(u8, "u8", 1, 1, &.{8});
    try writer.writeChunk(u16, "u16", 1, 1, &.{16});
    try writer.writeChunk(u32, "u32", 1, 1, &.{32});
    try writer.writeChunk(u64, "u64", 1, 1, &.{64});
    try writer.writeChunk(i8, "i8", 1, 1, &.{-8});
    try writer.writeChunk(i16, "i16", 1, 1, &.{-16});
    try writer.writeChunk(i32, "i32", 1, 1, &.{-32});
    try writer.writeChunk(i64, "i64", 1, 1, &.{-64});
    try writer.writeChunk(f32, "f32", 1, 1, &.{3.25});
    try writer.writeChunk(f64, "f64", 1, 1, &.{6.5});
    try writer.endFrame();
    try writer.close();

    var reader = try gsd.File.openRead(std.testing.allocator, path);
    defer reader.deinit();
    var u8_value: [1]u8 = undefined;
    var u16_value: [1]u16 = undefined;
    var u32_value: [1]u32 = undefined;
    var u64_value: [1]u64 = undefined;
    var i8_value: [1]i8 = undefined;
    var i16_value: [1]i16 = undefined;
    var i32_value: [1]i32 = undefined;
    var i64_value: [1]i64 = undefined;
    var f32_value: [1]f32 = undefined;
    var f64_value: [1]f64 = undefined;
    _ = try reader.readChunkInto(u8, 0, "u8", &u8_value);
    _ = try reader.readChunkInto(u16, 0, "u16", &u16_value);
    _ = try reader.readChunkInto(u32, 0, "u32", &u32_value);
    _ = try reader.readChunkInto(u64, 0, "u64", &u64_value);
    _ = try reader.readChunkInto(i8, 0, "i8", &i8_value);
    _ = try reader.readChunkInto(i16, 0, "i16", &i16_value);
    _ = try reader.readChunkInto(i32, 0, "i32", &i32_value);
    _ = try reader.readChunkInto(i64, 0, "i64", &i64_value);
    _ = try reader.readChunkInto(f32, 0, "f32", &f32_value);
    _ = try reader.readChunkInto(f64, 0, "f64", &f64_value);
    try std.testing.expectEqual(@as(u8, 8), u8_value[0]);
    try std.testing.expectEqual(@as(u16, 16), u16_value[0]);
    try std.testing.expectEqual(@as(u32, 32), u32_value[0]);
    try std.testing.expectEqual(@as(u64, 64), u64_value[0]);
    try std.testing.expectEqual(@as(i8, -8), i8_value[0]);
    try std.testing.expectEqual(@as(i16, -16), i16_value[0]);
    try std.testing.expectEqual(@as(i32, -32), i32_value[0]);
    try std.testing.expectEqual(@as(i64, -64), i64_value[0]);
    try std.testing.expectEqual(@as(f32, 3.25), f32_value[0]);
    try std.testing.expectEqual(@as(f64, 6.5), f64_value[0]);
}

test "exclusive creation and invalid files report errors" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try fixturePath(std.testing.allocator, temporary);
    defer std.testing.allocator.free(path);
    var writer = try gsd.File.create(std.testing.allocator, path, "test", "schema", 0, true);
    try writer.close();
    try std.testing.expectError(
        error.Io,
        gsd.File.create(std.testing.allocator, path, "test", "schema", 0, true),
    );

    const io = std.testing.io;
    const bad = try temporary.dir.createFile(io, "bad.gsd", .{});
    try bad.writePositionalAll(io, "not a gsd file", 0);
    bad.close(io);
    const bad_path = try std.fmt.allocPrint(
        std.testing.allocator,
        ".zig-cache/tmp/{s}/bad.gsd",
        .{temporary.sub_path},
    );
    defer std.testing.allocator.free(bad_path);
    try std.testing.expectError(error.NotGsd, gsd.File.openRead(std.testing.allocator, bad_path));
}
