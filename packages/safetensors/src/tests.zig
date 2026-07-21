const std = @import("std");
const safetensors = @import("safetensors");

fn makeFixture(allocator: std.mem.Allocator, header: []const u8, data: []const u8) ![]u8 {
    const bytes = try allocator.alloc(u8, 8 + header.len + data.len);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8 .. 8 + header.len], header);
    @memcpy(bytes[8 + header.len ..], data);
    return bytes;
}

fn expectFixtureError(expected: anyerror, header: []const u8, data: []const u8) !void {
    const bytes = try makeFixture(std.testing.allocator, header, data);
    defer std.testing.allocator.free(bytes);
    try std.testing.expectError(expected, safetensors.Reader.fromBytes(std.testing.allocator, bytes));
}

test "official-compatible bytes expose metadata, ordered keys, and typed tensors" {
    // Exact bytes emitted by safetensors.numpy.save from safetensors 0.8.0.
    const bytes =
        "\x90\x00\x00\x00\x00\x00\x00\x00" ++
        "{\"__metadata__\":{\"schema\":\"test.v1\"},\"a\":{\"dtype\":\"F64\",\"shape\":[],\"data_offsets\":[0,8]},\"z\":{\"dtype\":\"I8\",\"shape\":[2],\"data_offsets\":[8,10]}}  " ++
        "\x00\x00\x00\x00\x00\x00\x0c\x40\x04\x05";

    var reader = try safetensors.Reader.fromBytes(std.testing.allocator, bytes);
    defer reader.deinit();

    try std.testing.expectEqualDeep(&[_][]const u8{ "a", "z" }, reader.keys());
    try std.testing.expectEqualDeep(&[_][]const u8{ "a", "z" }, reader.offsetKeys());
    try std.testing.expect(reader.contains("a"));
    try std.testing.expect(!reader.contains("missing"));
    try std.testing.expectEqualStrings("test.v1", reader.metadata().?.get("schema").?);

    const tensor = try reader.tensor("a");
    try std.testing.expectEqual(safetensors.Dtype.f64, tensor.dtype);
    try std.testing.expectEqual(@as(usize, 1), tensor.elementCount());
    try std.testing.expectEqual(@as(f64, 3.5), (try tensor.values(f64))[0]);
    try std.testing.expectError(error.TensorNotFound, reader.tensor("missing"));
}

test "all v0.8 dtype codes are represented" {
    const cases = [_]struct { []const u8, safetensors.Dtype, usize }{
        .{ "BOOL", .bool, 8 },
        .{ "F4", .f4, 4 },
        .{ "F6_E2M3", .f6_e2m3, 6 },
        .{ "F6_E3M2", .f6_e3m2, 6 },
        .{ "U8", .u8, 8 },
        .{ "I8", .i8, 8 },
        .{ "F8_E5M2", .f8_e5m2, 8 },
        .{ "F8_E4M3", .f8_e4m3, 8 },
        .{ "F8_E8M0", .f8_e8m0, 8 },
        .{ "F8_E4M3FNUZ", .f8_e4m3fnuz, 8 },
        .{ "F8_E5M2FNUZ", .f8_e5m2fnuz, 8 },
        .{ "I16", .i16, 16 },
        .{ "U16", .u16, 16 },
        .{ "F16", .f16, 16 },
        .{ "BF16", .bf16, 16 },
        .{ "I32", .i32, 32 },
        .{ "U32", .u32, 32 },
        .{ "F32", .f32, 32 },
        .{ "C64", .c64, 64 },
        .{ "F64", .f64, 64 },
        .{ "I64", .i64, 64 },
        .{ "U64", .u64, 64 },
    };
    for (cases) |case| {
        const dtype = try safetensors.Dtype.parse(case[0]);
        try std.testing.expectEqual(case[1], dtype);
        try std.testing.expectEqualStrings(case[0], dtype.code());
        try std.testing.expectEqual(case[2], dtype.bits());
    }
    try std.testing.expectError(error.UnknownDtype, safetensors.Dtype.parse("F128"));
}

test "frame and leading range views are zero-copy and checked" {
    const header =
        \\{"frames":{"dtype":"U16","shape":[3,2],"data_offsets":[0,12]}}
    ;
    var data: [12]u8 = undefined;
    for (0..6) |index| {
        std.mem.writeInt(u16, data[index * 2 ..][0..2], @intCast(10 + index), .little);
    }
    const bytes = try makeFixture(std.testing.allocator, header, &data);
    defer std.testing.allocator.free(bytes);
    var reader = try safetensors.Reader.fromBytes(std.testing.allocator, bytes);
    defer reader.deinit();

    const frames = try reader.tensor("frames");
    const second = try frames.frame(1);
    try std.testing.expectEqualDeep(&[_]usize{2}, second.shape);
    try std.testing.expectEqualDeep(&[_]u16{ 12, 13 }, try second.values(u16));
    try std.testing.expectEqual(@intFromPtr(frames.bytes.ptr) + 4, @intFromPtr(second.bytes.ptr));

    const range = try frames.leadingRange(1, 3);
    try std.testing.expectEqual(@as(usize, 2), range.leading_len);
    try std.testing.expectEqualDeep(&[_]usize{2}, range.trailing_shape);
    try std.testing.expectEqualDeep(&[_]u16{ 12, 13, 14, 15 }, try range.values(u16));
    try std.testing.expectError(error.DtypeMismatch, range.values(f32));
    try std.testing.expectError(error.IndexOutOfBounds, frames.frame(3));
    try std.testing.expectError(error.InvalidRange, frames.leadingRange(2, 1));
    try std.testing.expectError(error.IndexOutOfBounds, frames.leadingRange(0, 4));

    const scalar_header =
        \\{"scalar":{"dtype":"U8","shape":[],"data_offsets":[0,1]}}
    ;
    const scalar_bytes = try makeFixture(std.testing.allocator, scalar_header, &.{7});
    defer std.testing.allocator.free(scalar_bytes);
    var scalar_reader = try safetensors.Reader.fromBytes(std.testing.allocator, scalar_bytes);
    defer scalar_reader.deinit();
    try std.testing.expectError(error.LeadingDimensionRequired, (try scalar_reader.tensor("scalar")).frame(0));
}

test "packed tensors remain readable as bytes and reject half-byte frames" {
    const header =
        \\{"packed":{"dtype":"F4","shape":[4,1],"data_offsets":[0,2]}}
    ;
    const bytes = try makeFixture(std.testing.allocator, header, &.{ 0x21, 0x43 });
    defer std.testing.allocator.free(bytes);
    var reader = try safetensors.Reader.fromBytes(std.testing.allocator, bytes);
    defer reader.deinit();
    const packed_tensor = try reader.tensor("packed");
    try std.testing.expectEqualDeep(&[_]u8{ 0x21, 0x43 }, packed_tensor.bytes);
    try std.testing.expectError(error.SliceNotByteAligned, packed_tensor.frame(0));
}

test "empty metadata remains distinct from absent metadata" {
    const with_header =
        \\{"__metadata__":{},"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}
    ;
    const with_bytes = try makeFixture(std.testing.allocator, with_header, &.{});
    defer std.testing.allocator.free(with_bytes);
    var with_reader = try safetensors.Reader.fromBytes(std.testing.allocator, with_bytes);
    defer with_reader.deinit();
    try std.testing.expectEqual(@as(usize, 0), with_reader.metadata().?.len());

    const without_header =
        \\{"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}
    ;
    const without_bytes = try makeFixture(std.testing.allocator, without_header, &.{});
    defer std.testing.allocator.free(without_bytes);
    var without_reader = try safetensors.Reader.fromBytes(std.testing.allocator, without_bytes);
    defer without_reader.deinit();
    try std.testing.expect(without_reader.metadata() == null);
}

test "malformed files fail validation without leaking" {
    try std.testing.expectError(error.HeaderTooSmall, safetensors.Reader.fromBytes(std.testing.allocator, "tiny"));

    var too_large: [8]u8 = undefined;
    std.mem.writeInt(u64, &too_large, safetensors.max_header_size + 1, .little);
    try std.testing.expectError(error.HeaderTooLarge, safetensors.Reader.fromBytes(std.testing.allocator, &too_large));

    var truncated_header: [8]u8 = undefined;
    std.mem.writeInt(u64, &truncated_header, 12, .little);
    try std.testing.expectError(error.InvalidHeaderLength, safetensors.Reader.fromBytes(std.testing.allocator, &truncated_header));

    try expectFixtureError(error.InvalidHeader, " {}", &.{});
    try expectFixtureError(error.InvalidHeader, "{not json}", &.{});
    try expectFixtureError(error.DuplicateField,
        \\{"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0]},"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}
    , &.{});
    try expectFixtureError(error.InvalidMetadata,
        \\{"__metadata__":{"bad":3}}
    , &.{});
    try expectFixtureError(error.UnknownDtype,
        \\{"x":{"dtype":"F128","shape":[1],"data_offsets":[0,1]}}
    , &.{0});
    try expectFixtureError(error.InvalidShape,
        \\{"x":{"dtype":"U8","shape":[-1],"data_offsets":[0,0]}}
    , &.{});
    try expectFixtureError(error.ValidationOverflow,
        \\{"x":{"dtype":"U8","shape":[18446744073709551615,2],"data_offsets":[0,0]}}
    , &.{});
    try expectFixtureError(error.InvalidOffsets,
        \\{"x":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}
    , &.{ 0, 1 });
    try expectFixtureError(error.InvalidOffsets,
        \\{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"y":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}
    , &.{ 0, 1 });
    try expectFixtureError(error.InvalidTensorSize,
        \\{"x":{"dtype":"U16","shape":[1],"data_offsets":[0,1]}}
    , &.{0});
    try expectFixtureError(error.InvalidTensorSize,
        \\{"x":{"dtype":"F4","shape":[1],"data_offsets":[0,1]}}
    , &.{0});
    try expectFixtureError(error.IncompletePayload,
        \\{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}
    , &.{ 0, 1 });
}

test "mmap opening keeps a large sparse payload outside the reader allocator" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const io = std.testing.io;
    const file = try temporary.dir.createFile(io, "large.safetensors", .{ .read = true });
    defer file.close(io);

    const frame_count: usize = 65_536;
    const frame_bytes: usize = 1_024;
    const payload_len = frame_count * frame_bytes;
    const header =
        \\{"frames":{"dtype":"U8","shape":[65536,1024],"data_offsets":[0,67108864]}}
    ;
    const prefix = try makeFixture(std.testing.allocator, header, &.{});
    defer std.testing.allocator.free(prefix);
    try file.writePositionalAll(io, prefix, 0);
    try file.setLength(io, prefix.len + payload_len);
    const distant_offset = prefix.len + (frame_count - 1) * frame_bytes;
    try file.writePositionalAll(io, &.{0xab}, distant_offset);

    const path = try std.fmt.allocPrint(
        std.testing.allocator,
        ".zig-cache/tmp/{s}/large.safetensors",
        .{temporary.sub_path},
    );
    defer std.testing.allocator.free(path);

    var allocator_storage: [64 * 1024]u8 = undefined;
    var fixed = std.heap.FixedBufferAllocator.init(&allocator_storage);
    var reader = try safetensors.Reader.open(fixed.allocator(), path);
    const allocated_during_open = fixed.end_index;
    defer reader.deinit();
    try std.testing.expect(allocated_during_open < allocator_storage.len);
    try std.testing.expect(allocated_during_open < payload_len / 1_000);

    const distant = try (try reader.tensor("frames")).frame(frame_count - 1);
    try std.testing.expectEqual(@as(usize, frame_bytes), distant.byteLen());
    try std.testing.expectEqual(@as(u8, 0xab), distant.bytes[0]);
    try std.testing.expectEqual(@as(u8, 0), distant.bytes[frame_bytes - 1]);
}
