//! Thin, owning Zig wrapper around the GSD C API.

const std = @import("std");
const c = @import("gsd_c");

pub const Error = error{
    Io,
    InvalidArgument,
    NotGsd,
    UnsupportedVersion,
    CorruptFile,
    OutOfMemory,
    NameListFull,
    NotWritable,
    NotReadable,
    UnknownGsdError,
    ChunkNotFound,
    DtypeMismatch,
    BufferSizeMismatch,
    SizeOverflow,
    InvalidShape,
};

pub const Dtype = enum(u8) {
    u8 = c.GSD_TYPE_UINT8,
    u16 = c.GSD_TYPE_UINT16,
    u32 = c.GSD_TYPE_UINT32,
    u64 = c.GSD_TYPE_UINT64,
    i8 = c.GSD_TYPE_INT8,
    i16 = c.GSD_TYPE_INT16,
    i32 = c.GSD_TYPE_INT32,
    i64 = c.GSD_TYPE_INT64,
    f32 = c.GSD_TYPE_FLOAT,
    f64 = c.GSD_TYPE_DOUBLE,
    character = c.GSD_TYPE_CHARACTER,

    pub fn fromC(value: u8) Error!Dtype {
        return switch (value) {
            c.GSD_TYPE_UINT8 => .u8,
            c.GSD_TYPE_UINT16 => .u16,
            c.GSD_TYPE_UINT32 => .u32,
            c.GSD_TYPE_UINT64 => .u64,
            c.GSD_TYPE_INT8 => .i8,
            c.GSD_TYPE_INT16 => .i16,
            c.GSD_TYPE_INT32 => .i32,
            c.GSD_TYPE_INT64 => .i64,
            c.GSD_TYPE_FLOAT => .f32,
            c.GSD_TYPE_DOUBLE => .f64,
            c.GSD_TYPE_CHARACTER => .character,
            else => error.InvalidArgument,
        };
    }

    pub fn of(comptime T: type) Dtype {
        return switch (T) {
            u8 => .u8,
            u16 => .u16,
            u32 => .u32,
            u64 => .u64,
            i8 => .i8,
            i16 => .i16,
            i32 => .i32,
            i64 => .i64,
            f32 => .f32,
            f64 => .f64,
            else => @compileError("unsupported GSD element type: " ++ @typeName(T)),
        };
    }

    pub fn size(self: Dtype) usize {
        return c.gsd_sizeof_type(@enumFromInt(@intFromEnum(self)));
    }
};

pub const OpenMode = enum {
    read_only,
    read_write,
    append,
};

pub const Header = struct {
    application: []const u8,
    schema: []const u8,
    schema_version: u32,
    file_version: u32,
};

pub const Chunk = struct {
    entry: [*c]const c.struct_gsd_index_entry,
    frame: u64,
    rows: u64,
    columns: u32,
    dtype: Dtype,

    pub fn elementCount(self: Chunk) Error!usize {
        const rows = std.math.cast(usize, self.rows) orelse return error.SizeOverflow;
        return std.math.mul(usize, rows, self.columns) catch error.SizeOverflow;
    }

    pub fn byteLen(self: Chunk) Error!usize {
        return std.math.mul(usize, try self.elementCount(), self.dtype.size()) catch error.SizeOverflow;
    }
};

pub const File = struct {
    handle: c.struct_gsd_handle,
    open: bool,

    pub fn openPath(allocator: std.mem.Allocator, path: []const u8, mode: OpenMode) !File {
        const path_z = try allocator.dupeSentinel(u8, path, 0);
        defer allocator.free(path_z);
        var result = File{ .handle = std.mem.zeroes(c.struct_gsd_handle), .open = false };
        try check(c.gsd_open(&result.handle, path_z.ptr, cOpenMode(mode)));
        result.open = true;
        return result;
    }

    pub fn openRead(allocator: std.mem.Allocator, path: []const u8) !File {
        return openPath(allocator, path, .read_only);
    }

    pub fn create(
        allocator: std.mem.Allocator,
        path: []const u8,
        application: []const u8,
        schema: []const u8,
        schema_version: u32,
        exclusive: bool,
    ) !File {
        const path_z = try allocator.dupeSentinel(u8, path, 0);
        defer allocator.free(path_z);
        const application_z = try allocator.dupeSentinel(u8, application, 0);
        defer allocator.free(application_z);
        const schema_z = try allocator.dupeSentinel(u8, schema, 0);
        defer allocator.free(schema_z);
        var result = File{ .handle = std.mem.zeroes(c.struct_gsd_handle), .open = false };
        try check(c.gsd_create_and_open(
            &result.handle,
            path_z.ptr,
            application_z.ptr,
            schema_z.ptr,
            schema_version,
            c.GSD_OPEN_READWRITE,
            @intFromBool(exclusive),
        ));
        result.open = true;
        return result;
    }

    pub fn deinit(self: *File) void {
        if (self.open) {
            _ = c.gsd_close(&self.handle);
            self.open = false;
        }
    }

    pub fn close(self: *File) Error!void {
        if (!self.open) return;
        try check(c.gsd_close(&self.handle));
        self.open = false;
    }

    pub fn header(self: *const File) Header {
        return .{
            .application = cStringFromArray(&self.handle.header.application),
            .schema = cStringFromArray(&self.handle.header.schema),
            .schema_version = self.handle.header.schema_version,
            .file_version = self.handle.header.gsd_version,
        };
    }

    pub fn frameCount(self: *File) u64 {
        return c.gsd_get_nframes(&self.handle);
    }

    pub fn findChunk(self: *File, frame: u64, name: [:0]const u8) Error!?Chunk {
        const entry = c.gsd_find_chunk(&self.handle, frame, name.ptr) orelse return null;
        return .{
            .entry = entry,
            .frame = entry.*.frame,
            .rows = entry.*.N,
            .columns = entry.*.M,
            .dtype = try Dtype.fromC(entry.*.type),
        };
    }

    pub fn requireChunk(self: *File, frame: u64, name: [:0]const u8) Error!Chunk {
        return (try self.findChunk(frame, name)) orelse error.ChunkNotFound;
    }

    pub fn readChunkBytes(self: *File, chunk: Chunk, destination: []u8) Error!void {
        if (destination.len != try chunk.byteLen()) return error.BufferSizeMismatch;
        try check(c.gsd_read_chunk(&self.handle, destination.ptr, chunk.entry));
    }

    pub fn readChunkInto(
        self: *File,
        comptime T: type,
        frame: u64,
        name: [:0]const u8,
        destination: []T,
    ) Error!Chunk {
        const chunk = try self.requireChunk(frame, name);
        if (chunk.dtype != Dtype.of(T)) return error.DtypeMismatch;
        if (destination.len != try chunk.elementCount()) return error.BufferSizeMismatch;
        try check(c.gsd_read_chunk(&self.handle, destination.ptr, chunk.entry));
        return chunk;
    }

    pub fn writeChunk(
        self: *File,
        comptime T: type,
        name: [:0]const u8,
        rows: u64,
        columns: u32,
        values: []const T,
    ) Error!void {
        if (columns == 0) return error.InvalidShape;
        const row_count = std.math.cast(usize, rows) orelse return error.SizeOverflow;
        const expected = std.math.mul(usize, row_count, columns) catch return error.SizeOverflow;
        if (values.len != expected) return error.BufferSizeMismatch;
        try check(c.gsd_write_chunk(
            &self.handle,
            name.ptr,
            @intFromEnum(Dtype.of(T)),
            rows,
            columns,
            0,
            if (values.len == 0) null else values.ptr,
        ));
    }

    pub fn endFrame(self: *File) Error!void {
        try check(c.gsd_end_frame(&self.handle));
    }

    pub fn flush(self: *File) Error!void {
        try check(c.gsd_flush(&self.handle));
    }
};

pub fn makeVersion(major: u16, minor: u16) u32 {
    return c.gsd_make_version(major, minor);
}

fn cOpenMode(mode: OpenMode) c.enum_gsd_open_flag {
    return switch (mode) {
        .read_only => c.GSD_OPEN_READONLY,
        .read_write => c.GSD_OPEN_READWRITE,
        .append => c.GSD_OPEN_APPEND,
    };
}

fn cStringFromArray(array: anytype) []const u8 {
    const bytes: [*]const u8 = @ptrCast(array);
    var length: usize = 0;
    while (length < array.len and bytes[length] != 0) : (length += 1) {}
    return bytes[0..length];
}

fn check(code: c_int) Error!void {
    if (code == c.GSD_SUCCESS) return;
    return switch (code) {
        c.GSD_ERROR_IO => error.Io,
        c.GSD_ERROR_INVALID_ARGUMENT => error.InvalidArgument,
        c.GSD_ERROR_NOT_A_GSD_FILE => error.NotGsd,
        c.GSD_ERROR_INVALID_GSD_FILE_VERSION => error.UnsupportedVersion,
        c.GSD_ERROR_FILE_CORRUPT => error.CorruptFile,
        c.GSD_ERROR_MEMORY_ALLOCATION_FAILED => error.OutOfMemory,
        c.GSD_ERROR_NAMELIST_FULL => error.NameListFull,
        c.GSD_ERROR_FILE_MUST_BE_WRITABLE => error.NotWritable,
        c.GSD_ERROR_FILE_MUST_BE_READABLE => error.NotReadable,
        else => error.UnknownGsdError,
    };
}
