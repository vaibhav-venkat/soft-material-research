//! Read-only, zero-copy safetensors v0.8 reader.
//!
//! `Reader.open` memory maps a completed file read-only. Tensor and frame views
//! borrow that mapping, so the reader must outlive every view and the input file
//! must not be modified or replaced before `Reader.deinit`.

const std = @import("std");
const builtin = @import("builtin");

pub const max_header_size: usize = 100_000_000;
const header_length_size: usize = 8;
const JsonValue = std.json.Value;
const ParsedJson = std.json.Parsed(JsonValue);
const page_alignment = std.heap.page_size_min;

pub const Error = error{
    HeaderTooSmall,
    HeaderTooLarge,
    InvalidHeaderLength,
    InvalidHeader,
    DuplicateField,
    InvalidMetadata,
    InvalidTensor,
    UnknownDtype,
    InvalidShape,
    InvalidOffsets,
    InvalidTensorSize,
    IncompletePayload,
    ValidationOverflow,
    TensorNotFound,
    LeadingDimensionRequired,
    IndexOutOfBounds,
    InvalidRange,
    SliceNotByteAligned,
    DtypeMismatch,
    UnsupportedEndian,
};

/// Every dtype recognized by safetensors v0.8.
pub const Dtype = enum {
    bool,
    f4,
    f6_e2m3,
    f6_e3m2,
    u8,
    i8,
    f8_e5m2,
    f8_e4m3,
    f8_e8m0,
    f8_e4m3fnuz,
    f8_e5m2fnuz,
    i16,
    u16,
    f16,
    bf16,
    i32,
    u32,
    f32,
    c64,
    f64,
    i64,
    u64,

    pub fn parse(dtype_code: []const u8) Error!Dtype {
        const table = [_]struct { []const u8, Dtype }{
            .{ "BOOL", .bool },
            .{ "F4", .f4 },
            .{ "F6_E2M3", .f6_e2m3 },
            .{ "F6_E3M2", .f6_e3m2 },
            .{ "U8", .u8 },
            .{ "I8", .i8 },
            .{ "F8_E5M2", .f8_e5m2 },
            .{ "F8_E4M3", .f8_e4m3 },
            .{ "F8_E8M0", .f8_e8m0 },
            .{ "F8_E4M3FNUZ", .f8_e4m3fnuz },
            .{ "F8_E5M2FNUZ", .f8_e5m2fnuz },
            .{ "I16", .i16 },
            .{ "U16", .u16 },
            .{ "F16", .f16 },
            .{ "BF16", .bf16 },
            .{ "I32", .i32 },
            .{ "U32", .u32 },
            .{ "F32", .f32 },
            .{ "C64", .c64 },
            .{ "F64", .f64 },
            .{ "I64", .i64 },
            .{ "U64", .u64 },
        };
        for (table) |entry| {
            if (std.mem.eql(u8, dtype_code, entry[0])) return entry[1];
        }
        return error.UnknownDtype;
    }

    pub fn code(self: Dtype) []const u8 {
        return switch (self) {
            .bool => "BOOL",
            .f4 => "F4",
            .f6_e2m3 => "F6_E2M3",
            .f6_e3m2 => "F6_E3M2",
            .u8 => "U8",
            .i8 => "I8",
            .f8_e5m2 => "F8_E5M2",
            .f8_e4m3 => "F8_E4M3",
            .f8_e8m0 => "F8_E8M0",
            .f8_e4m3fnuz => "F8_E4M3FNUZ",
            .f8_e5m2fnuz => "F8_E5M2FNUZ",
            .i16 => "I16",
            .u16 => "U16",
            .f16 => "F16",
            .bf16 => "BF16",
            .i32 => "I32",
            .u32 => "U32",
            .f32 => "F32",
            .c64 => "C64",
            .f64 => "F64",
            .i64 => "I64",
            .u64 => "U64",
        };
    }

    pub fn bits(self: Dtype) usize {
        return switch (self) {
            .f4 => 4,
            .f6_e2m3, .f6_e3m2 => 6,
            .bool, .u8, .i8, .f8_e5m2, .f8_e4m3, .f8_e8m0, .f8_e4m3fnuz, .f8_e5m2fnuz => 8,
            .i16, .u16, .f16, .bf16 => 16,
            .i32, .u32, .f32 => 32,
            .c64, .f64, .i64, .u64 => 64,
        };
    }
};

pub const MetadataEntry = struct {
    key: []const u8,
    value: []const u8,
};

pub const MetadataView = struct {
    entries: []const MetadataEntry,

    pub fn len(self: MetadataView) usize {
        return self.entries.len;
    }

    pub fn get(self: MetadataView, key: []const u8) ?[]const u8 {
        for (self.entries) |entry| {
            if (std.mem.eql(u8, entry.key, key)) return entry.value;
        }
        return null;
    }
};

pub const TensorView = struct {
    dtype: Dtype,
    shape: []const usize,
    bytes: []const u8,

    pub fn byteLen(self: TensorView) usize {
        return self.bytes.len;
    }

    pub fn elementCount(self: TensorView) usize {
        return elementCountChecked(self.shape) catch unreachable;
    }

    /// Return an unaligned native typed view without copying.
    pub fn values(self: TensorView, comptime T: type) Error![]align(1) const T {
        return typedValues(T, self.dtype, self.bytes);
    }

    /// Select one axis-0 frame and drop that leading dimension.
    pub fn frame(self: TensorView, index: usize) Error!TensorView {
        if (self.shape.len == 0) return error.LeadingDimensionRequired;
        if (index >= self.shape[0]) return error.IndexOutOfBounds;
        const stride = try trailingByteLen(self.dtype, self.shape[1..]);
        const start = std.math.mul(usize, index, stride) catch return error.ValidationOverflow;
        return .{
            .dtype = self.dtype,
            .shape = self.shape[1..],
            .bytes = self.bytes[start .. start + stride],
        };
    }

    /// Select a contiguous half-open axis-0 range without copying.
    pub fn leadingRange(self: TensorView, start: usize, stop: usize) Error!LeadingRangeView {
        if (self.shape.len == 0) return error.LeadingDimensionRequired;
        if (start > stop) return error.InvalidRange;
        if (stop > self.shape[0]) return error.IndexOutOfBounds;
        const stride = try trailingByteLen(self.dtype, self.shape[1..]);
        const byte_start = std.math.mul(usize, start, stride) catch return error.ValidationOverflow;
        const byte_stop = std.math.mul(usize, stop, stride) catch return error.ValidationOverflow;
        return .{
            .dtype = self.dtype,
            .leading_len = stop - start,
            .trailing_shape = self.shape[1..],
            .bytes = self.bytes[byte_start..byte_stop],
        };
    }
};

pub const LeadingRangeView = struct {
    dtype: Dtype,
    leading_len: usize,
    trailing_shape: []const usize,
    bytes: []const u8,

    pub fn byteLen(self: LeadingRangeView) usize {
        return self.bytes.len;
    }

    pub fn elementCount(self: LeadingRangeView) usize {
        const trailing = elementCountChecked(self.trailing_shape) catch unreachable;
        return std.math.mul(usize, self.leading_len, trailing) catch unreachable;
    }

    pub fn values(self: LeadingRangeView, comptime T: type) Error![]align(1) const T {
        return typedValues(T, self.dtype, self.bytes);
    }
};

const TensorInfo = struct {
    name: []const u8,
    dtype: Dtype,
    shape: []const usize,
    start: usize,
    stop: usize,
};

/// Parsed safetensors metadata plus either a borrowed buffer or owned mmap.
pub const Reader = struct {
    allocator: std.mem.Allocator,
    bytes: []const u8,
    mapping: ?[]align(page_alignment) const u8,
    parsed: ParsedJson,
    tensors: []TensorInfo,
    lexical_names: [][]const u8,
    offset_names: [][]const u8,
    metadata_entries: []MetadataEntry,
    has_metadata: bool,
    data_start: usize,

    /// Open and mmap a completed regular file read-only.
    pub fn open(allocator: std.mem.Allocator, path: []const u8) !Reader {
        if (builtin.os.tag == .windows or builtin.os.tag == .wasi) {
            @compileError("Reader.open currently supports POSIX targets only");
        }
        const io = std.Io.Threaded.global_single_threaded.io();
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        const file_len_u64 = try file.length(io);
        const file_len = std.math.cast(usize, file_len_u64) orelse return error.ValidationOverflow;
        if (file_len < header_length_size) return error.HeaderTooSmall;
        const mapped = try std.posix.mmap(
            null,
            file_len,
            .{ .READ = true },
            .{ .TYPE = .SHARED },
            file.handle,
            0,
        );
        errdefer std.posix.munmap(mapped);

        var reader = try init(allocator, mapped, mapped);
        reader.mapping = mapped;
        return reader;
    }

    /// Parse a borrowed complete safetensors buffer without copying payloads.
    pub fn fromBytes(allocator: std.mem.Allocator, bytes: []const u8) !Reader {
        return init(allocator, bytes, null);
    }

    fn init(
        allocator: std.mem.Allocator,
        bytes: []const u8,
        mapping: ?[]align(page_alignment) const u8,
    ) !Reader {
        if (bytes.len < header_length_size) return error.HeaderTooSmall;
        const header_len_u64 = std.mem.readInt(u64, bytes[0..header_length_size], .little);
        const header_len = std.math.cast(usize, header_len_u64) orelse return error.HeaderTooLarge;
        if (header_len > max_header_size) return error.HeaderTooLarge;
        const data_start = std.math.add(usize, header_length_size, header_len) catch return error.InvalidHeaderLength;
        if (data_start > bytes.len) return error.InvalidHeaderLength;
        const header = bytes[header_length_size..data_start];
        if (header.len == 0 or header[0] != '{') return error.InvalidHeader;
        if (!std.unicode.utf8ValidateSlice(header)) return error.InvalidHeader;

        const parsed = std.json.parseFromSlice(JsonValue, allocator, header, .{
            .duplicate_field_behavior = .@"error",
        }) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.DuplicateField => return error.DuplicateField,
            else => return error.InvalidHeader,
        };
        errdefer parsed.deinit();
        const root = switch (parsed.value) {
            .object => |object| object,
            else => return error.InvalidHeader,
        };

        var tensor_count: usize = root.count();
        if (root.contains("__metadata__")) tensor_count -= 1;
        const tensors = try allocator.alloc(TensorInfo, tensor_count);
        errdefer allocator.free(tensors);
        var initialized_tensors: usize = 0;
        errdefer for (tensors[0..initialized_tensors]) |tensor_info| allocator.free(tensor_info.shape);
        const lexical_names = try allocator.alloc([]const u8, tensor_count);
        errdefer allocator.free(lexical_names);
        const offset_names = try allocator.alloc([]const u8, tensor_count);
        errdefer allocator.free(offset_names);

        var metadata_count: usize = 0;
        const has_metadata = root.contains("__metadata__");
        if (root.get("__metadata__")) |metadata_value| {
            metadata_count = switch (metadata_value) {
                .object => |object| object.count(),
                else => return error.InvalidMetadata,
            };
        }
        const metadata_entries = try allocator.alloc(MetadataEntry, metadata_count);
        errdefer allocator.free(metadata_entries);

        var tensor_index: usize = 0;
        var metadata_index: usize = 0;
        var iterator = root.iterator();
        while (iterator.next()) |entry| {
            if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) {
                const object = switch (entry.value_ptr.*) {
                    .object => |value| value,
                    else => return error.InvalidMetadata,
                };
                var metadata_iterator = object.iterator();
                while (metadata_iterator.next()) |metadata_entry| {
                    const value = switch (metadata_entry.value_ptr.*) {
                        .string => |string| string,
                        else => return error.InvalidMetadata,
                    };
                    metadata_entries[metadata_index] = .{
                        .key = metadata_entry.key_ptr.*,
                        .value = value,
                    };
                    metadata_index += 1;
                }
                continue;
            }
            tensors[tensor_index] = try parseTensor(allocator, entry.key_ptr.*, entry.value_ptr.*);
            tensor_index += 1;
            initialized_tensors = tensor_index;
        }

        std.mem.sort(TensorInfo, tensors, {}, tensorOffsetLessThan);
        std.mem.sort(MetadataEntry, metadata_entries, {}, metadataLessThan);
        var expected_start: usize = 0;
        for (tensors, 0..) |tensor_info, index| {
            if (tensor_info.start != expected_start or tensor_info.stop < tensor_info.start) return error.InvalidOffsets;
            const elements = try elementCountChecked(tensor_info.shape);
            const bit_len = std.math.mul(usize, elements, tensor_info.dtype.bits()) catch return error.ValidationOverflow;
            if (bit_len % 8 != 0) return error.InvalidTensorSize;
            if (tensor_info.stop - tensor_info.start != bit_len / 8) return error.InvalidTensorSize;
            expected_start = tensor_info.stop;
            offset_names[index] = tensor_info.name;
            lexical_names[index] = tensor_info.name;
        }
        const expected_file_len = std.math.add(usize, data_start, expected_start) catch return error.ValidationOverflow;
        if (expected_file_len != bytes.len) return error.IncompletePayload;
        std.mem.sort([]const u8, lexical_names, {}, stringLessThan);

        return .{
            .allocator = allocator,
            .bytes = bytes,
            .mapping = mapping,
            .parsed = parsed,
            .tensors = tensors,
            .lexical_names = lexical_names,
            .offset_names = offset_names,
            .metadata_entries = metadata_entries,
            .has_metadata = has_metadata,
            .data_start = data_start,
        };
    }

    pub fn deinit(self: *Reader) void {
        self.allocator.free(self.metadata_entries);
        self.allocator.free(self.offset_names);
        self.allocator.free(self.lexical_names);
        for (self.tensors) |tensor_info| self.allocator.free(tensor_info.shape);
        self.allocator.free(self.tensors);
        self.parsed.deinit();
        if (self.mapping) |mapping| std.posix.munmap(mapping);
        self.* = undefined;
    }

    pub fn keys(self: *const Reader) []const []const u8 {
        return self.lexical_names;
    }

    pub fn offsetKeys(self: *const Reader) []const []const u8 {
        return self.offset_names;
    }

    pub fn metadata(self: *const Reader) ?MetadataView {
        if (!self.has_metadata) return null;
        return .{ .entries = self.metadata_entries };
    }

    pub fn contains(self: *const Reader, name: []const u8) bool {
        _ = self.find(name) orelse return false;
        return true;
    }

    pub fn tensor(self: *const Reader, name: []const u8) Error!TensorView {
        const info = self.find(name) orelse return error.TensorNotFound;
        const start = self.data_start + info.start;
        const stop = self.data_start + info.stop;
        return .{
            .dtype = info.dtype,
            .shape = info.shape,
            .bytes = self.bytes[start..stop],
        };
    }

    fn find(self: *const Reader, name: []const u8) ?*const TensorInfo {
        for (self.tensors) |*tensor_info| {
            if (std.mem.eql(u8, tensor_info.name, name)) return tensor_info;
        }
        return null;
    }
};

fn parseTensor(allocator: std.mem.Allocator, name: []const u8, value: JsonValue) !TensorInfo {
    if (name.len == 0) return error.InvalidTensor;
    const object = switch (value) {
        .object => |entry| entry,
        else => return error.InvalidTensor,
    };
    if (object.count() != 3) return error.InvalidTensor;
    const dtype_value = object.get("dtype") orelse return error.InvalidTensor;
    const dtype_code = switch (dtype_value) {
        .string => |string| string,
        else => return error.InvalidTensor,
    };
    const dtype = try Dtype.parse(dtype_code);

    const shape_value = object.get("shape") orelse return error.InvalidTensor;
    const shape_array = switch (shape_value) {
        .array => |array| array.items,
        else => return error.InvalidShape,
    };
    const shape = try allocator.alloc(usize, shape_array.len);
    errdefer allocator.free(shape);
    for (shape_array, 0..) |dimension, index| {
        shape[index] = try jsonUsize(dimension, error.InvalidShape);
    }

    const offsets_value = object.get("data_offsets") orelse return error.InvalidTensor;
    const offsets = switch (offsets_value) {
        .array => |array| array.items,
        else => return error.InvalidOffsets,
    };
    if (offsets.len != 2) return error.InvalidOffsets;
    return .{
        .name = name,
        .dtype = dtype,
        .shape = shape,
        .start = try jsonUsize(offsets[0], error.InvalidOffsets),
        .stop = try jsonUsize(offsets[1], error.InvalidOffsets),
    };
}

fn jsonUsize(value: JsonValue, comptime invalid: Error) Error!usize {
    return switch (value) {
        .integer => |integer| if (integer < 0) invalid else @intCast(integer),
        .number_string => |number| std.fmt.parseInt(usize, number, 10) catch |err| switch (err) {
            error.Overflow => error.ValidationOverflow,
            else => invalid,
        },
        else => invalid,
    };
}

fn elementCountChecked(shape: []const usize) Error!usize {
    var count: usize = 1;
    for (shape) |dimension| {
        count = std.math.mul(usize, count, dimension) catch return error.ValidationOverflow;
    }
    return count;
}

fn trailingByteLen(dtype: Dtype, shape: []const usize) Error!usize {
    const elements = try elementCountChecked(shape);
    const bits = std.math.mul(usize, elements, dtype.bits()) catch return error.ValidationOverflow;
    if (bits % 8 != 0) return error.SliceNotByteAligned;
    return bits / 8;
}

fn expectedDtype(comptime T: type) ?Dtype {
    return switch (T) {
        u8 => .u8,
        i8 => .i8,
        u16 => .u16,
        i16 => .i16,
        f16 => .f16,
        u32 => .u32,
        i32 => .i32,
        f32 => .f32,
        u64 => .u64,
        i64 => .i64,
        f64 => .f64,
        else => null,
    };
}

fn typedValues(comptime T: type, dtype: Dtype, bytes: []const u8) Error![]align(1) const T {
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedEndian;
    const expected = expectedDtype(T) orelse return error.DtypeMismatch;
    if (dtype != expected) return error.DtypeMismatch;
    if (bytes.len % @sizeOf(T) != 0) return error.InvalidTensorSize;
    const pointer: [*]align(1) const T = @ptrCast(bytes.ptr);
    return pointer[0 .. bytes.len / @sizeOf(T)];
}

fn tensorOffsetLessThan(_: void, left: TensorInfo, right: TensorInfo) bool {
    if (left.start != right.start) return left.start < right.start;
    if (left.stop != right.stop) return left.stop < right.stop;
    return std.mem.lessThan(u8, left.name, right.name);
}

fn metadataLessThan(_: void, left: MetadataEntry, right: MetadataEntry) bool {
    return std.mem.lessThan(u8, left.key, right.key);
}

fn stringLessThan(_: void, left: []const u8, right: []const u8) bool {
    return std.mem.lessThan(u8, left, right);
}
