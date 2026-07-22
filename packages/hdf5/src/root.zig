//! Owning Zig handles and caller-buffer I/O for the HDF5 C API.

const std = @import("std");
const c = @import("hdf5_c");

pub const Error = error{
    Hdf5Failure,
    InvalidRank,
    InvalidShape,
    InvalidSelection,
    SelectionOutOfBounds,
    BufferSizeMismatch,
    DtypeMismatch,
    UnsupportedDtype,
    SizeOverflow,
    ObjectNotFound,
    AttributeNotFound,
};

/// Numeric HDF5 datatype classes supported by this wrapper.
pub const Dtype = enum {
    u8,
    u16,
    u32,
    u64,
    i8,
    i16,
    i32,
    i64,
    f32,
    f64,

    /// Return the wrapper datatype for a supported Zig scalar type.
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
            else => @compileError("unsupported HDF5 element type: " ++ @typeName(T)),
        };
    }
};

/// `H5Fcreate` behavior: fail on an existing path or truncate it.
pub const CreateMode = enum { exclusive, truncate };
/// `H5Fopen` access mode.
pub const OpenMode = enum { read_only, read_write };

/// Dataset creation properties. A non-null shape enables HDF5 chunked layout.
pub const DatasetOptions = struct {
    chunk_shape: ?[]const u64 = null,
};

/// Owning HDF5 file identifier returned by `H5Fcreate` or `H5Fopen`.
pub const File = struct {
    allocator: std.mem.Allocator,
    id: c.hid_t,
    open: bool,

    /// Create an HDF5 file with `H5Fcreate` and default property lists.
    pub fn create(
        allocator: std.mem.Allocator,
        path: []const u8,
        mode: CreateMode,
    ) !File {
        try initialize();
        const path_z = try allocator.dupeSentinel(u8, path, 0);
        defer allocator.free(path_z);
        const flags: c_uint = switch (mode) {
            .exclusive => c.zig_h5f_acc_excl(),
            .truncate => c.zig_h5f_acc_trunc(),
        };
        const id = c.H5Fcreate(path_z.ptr, flags, c.zig_h5p_default(), c.zig_h5p_default());
        if (id < 0) return error.Hdf5Failure;
        return .{ .allocator = allocator, .id = id, .open = true };
    }

    /// Open an existing HDF5 file without reading any dataset payload.
    pub fn openPath(
        allocator: std.mem.Allocator,
        path: []const u8,
        mode: OpenMode,
    ) !File {
        try initialize();
        const path_z = try allocator.dupeSentinel(u8, path, 0);
        defer allocator.free(path_z);
        const flags: c_uint = switch (mode) {
            .read_only => c.zig_h5f_acc_rdonly(),
            .read_write => c.zig_h5f_acc_rdwr(),
        };
        const id = c.H5Fopen(path_z.ptr, flags, c.zig_h5p_default());
        if (id < 0) return error.Hdf5Failure;
        return .{ .allocator = allocator, .id = id, .open = true };
    }

    /// Close the file identifier, ignoring errors for deferred cleanup.
    pub fn deinit(self: *File) void {
        if (self.open) {
            _ = c.H5Fclose(self.id);
            self.open = false;
        }
    }

    /// Close the file identifier and report an `H5Fclose` failure.
    pub fn close(self: *File) Error!void {
        if (!self.open) return;
        try requireNonnegative(c.H5Fclose(self.id));
        self.open = false;
    }

    /// Flush file data and metadata globally with `H5Fflush`.
    pub fn flush(self: *File) Error!void {
        try requireNonnegative(c.H5Fflush(self.id, c.zig_h5f_scope_global()));
    }

    /// Test whether a link exists at `path` using `H5Lexists`.
    pub fn objectExists(self: *File, path: []const u8) !bool {
        const path_z = try self.allocator.dupeSentinel(u8, path, 0);
        defer self.allocator.free(path_z);
        const result = c.H5Lexists(self.id, path_z.ptr, c.zig_h5p_default());
        if (result < 0) return error.Hdf5Failure;
        return result > 0;
    }

    /// Create a group and any missing intermediate groups with `H5Gcreate2`.
    pub fn createGroup(self: *File, path: []const u8) !Group {
        const path_z = try self.allocator.dupeSentinel(u8, path, 0);
        defer self.allocator.free(path_z);
        const link_properties = c.H5Pcreate(c.zig_h5p_link_create());
        if (link_properties < 0) return error.Hdf5Failure;
        defer _ = c.H5Pclose(link_properties);
        try requireNonnegative(c.H5Pset_create_intermediate_group(link_properties, 1));
        const id = c.H5Gcreate2(
            self.id,
            path_z.ptr,
            link_properties,
            c.zig_h5p_default(),
            c.zig_h5p_default(),
        );
        if (id < 0) return error.Hdf5Failure;
        return .{ .id = id, .open = true };
    }

    /// Open an existing group with `H5Gopen2`.
    pub fn openGroup(self: *File, path: []const u8) !Group {
        const path_z = try self.allocator.dupeSentinel(u8, path, 0);
        defer self.allocator.free(path_z);
        const id = c.H5Gopen2(self.id, path_z.ptr, c.zig_h5p_default());
        if (id < 0) return error.ObjectNotFound;
        return .{ .id = id, .open = true };
    }

    /// Create a fixed-shape numeric dataset. Parent groups must already exist.
    /// Supplying `chunk_shape` selects HDF5 chunked storage.
    pub fn createDataset(
        self: *File,
        comptime T: type,
        path: []const u8,
        shape: []const u64,
        options: DatasetOptions,
    ) !Dataset {
        try validateShape(shape);
        if (options.chunk_shape) |chunk| try validateChunk(shape, chunk);
        const path_z = try self.allocator.dupeSentinel(u8, path, 0);
        defer self.allocator.free(path_z);
        const space = c.H5Screate_simple(@intCast(shape.len), shape.ptr, null);
        if (space < 0) return error.Hdf5Failure;
        defer _ = c.H5Sclose(space);
        const properties = c.H5Pcreate(c.zig_h5p_dataset_create());
        if (properties < 0) return error.Hdf5Failure;
        defer _ = c.H5Pclose(properties);
        if (options.chunk_shape) |chunk| {
            try requireNonnegative(c.H5Pset_chunk(properties, @intCast(chunk.len), chunk.ptr));
        }
        const id = c.H5Dcreate2(
            self.id,
            path_z.ptr,
            fileType(T),
            space,
            c.zig_h5p_default(),
            properties,
            c.zig_h5p_default(),
        );
        if (id < 0) return error.Hdf5Failure;
        return .{ .allocator = self.allocator, .id = id, .open = true };
    }

    /// Open a dataset handle without reading its payload.
    pub fn openDataset(self: *File, path: []const u8) !Dataset {
        const path_z = try self.allocator.dupeSentinel(u8, path, 0);
        defer self.allocator.free(path_z);
        const id = c.H5Dopen2(self.id, path_z.ptr, c.zig_h5p_default());
        if (id < 0) return error.ObjectNotFound;
        return .{ .allocator = self.allocator, .id = id, .open = true };
    }

    /// Remove a link with `H5Ldelete`; open object handles remain valid.
    pub fn deleteLink(self: *File, path: []const u8) !void {
        const path_z = try self.allocator.dupeSentinel(u8, path, 0);
        defer self.allocator.free(path_z);
        try requireNonnegative(c.H5Ldelete(self.id, path_z.ptr, c.zig_h5p_default()));
    }

    /// Rename or move a link within this file using `H5Lmove`.
    pub fn moveLink(self: *File, source: []const u8, destination: []const u8) !void {
        const source_z = try self.allocator.dupeSentinel(u8, source, 0);
        defer self.allocator.free(source_z);
        const destination_z = try self.allocator.dupeSentinel(u8, destination, 0);
        defer self.allocator.free(destination_z);
        try requireNonnegative(c.H5Lmove(
            self.id,
            source_z.ptr,
            self.id,
            destination_z.ptr,
            c.zig_h5p_default(),
            c.zig_h5p_default(),
        ));
    }

    /// Create or update a scalar numeric attribute on the file.
    pub fn writeAttribute(self: *File, comptime T: type, name: []const u8, value: T) !void {
        try writeNumericAttribute(self.allocator, self.id, T, name, value);
    }

    /// Test for a file attribute with `H5Aexists`.
    pub fn attributeExists(self: *File, name: []const u8) !bool {
        return attributeExistsAt(self.allocator, self.id, name);
    }

    /// Read a scalar numeric file attribute into native type `T`.
    pub fn readAttribute(self: *File, comptime T: type, name: []const u8) !T {
        return readNumericAttribute(self.allocator, self.id, T, name);
    }

    /// Replace a file attribute with a fixed-length UTF-8 string.
    pub fn writeStringAttribute(self: *File, name: []const u8, value: []const u8) !void {
        try writeStringAttributeAt(self.allocator, self.id, name, value);
    }

    /// Allocate and read a fixed-length string attribute; caller frees it.
    pub fn readStringAttributeAlloc(self: *File, allocator: std.mem.Allocator, name: []const u8) ![]u8 {
        return readStringAttributeAt(self.allocator, allocator, self.id, name);
    }
};

/// Owning HDF5 group identifier.
pub const Group = struct {
    id: c.hid_t,
    open: bool,

    /// Close the group identifier with `H5Gclose`.
    pub fn deinit(self: *Group) void {
        if (self.open) {
            _ = c.H5Gclose(self.id);
            self.open = false;
        }
    }
};

/// Owning HDF5 dataset identifier with caller-buffer I/O.
pub const Dataset = struct {
    allocator: std.mem.Allocator,
    id: c.hid_t,
    open: bool,

    /// Close the dataset identifier with `H5Dclose`.
    pub fn deinit(self: *Dataset) void {
        if (self.open) {
            _ = c.H5Dclose(self.id);
            self.open = false;
        }
    }

    /// Return the dataset dataspace rank.
    pub fn rank(self: *Dataset) Error!usize {
        const space = c.H5Dget_space(self.id);
        if (space < 0) return error.Hdf5Failure;
        defer _ = c.H5Sclose(space);
        const value = c.H5Sget_simple_extent_ndims(space);
        if (value < 0) return error.Hdf5Failure;
        return @intCast(value);
    }

    /// Allocate the current dataspace extents; caller frees the returned slice.
    pub fn shapeAlloc(self: *Dataset, allocator: std.mem.Allocator) ![]u64 {
        const space = c.H5Dget_space(self.id);
        if (space < 0) return error.Hdf5Failure;
        defer _ = c.H5Sclose(space);
        const rank_value = c.H5Sget_simple_extent_ndims(space);
        if (rank_value < 0) return error.Hdf5Failure;
        const shape = try allocator.alloc(u64, @intCast(rank_value));
        errdefer allocator.free(shape);
        if (c.H5Sget_simple_extent_dims(space, shape.ptr, null) < 0) return error.Hdf5Failure;
        return shape;
    }

    /// Inspect the dataset's integer/float class, sign, and byte width.
    pub fn dtype(self: *Dataset) Error!Dtype {
        const datatype = c.H5Dget_type(self.id);
        if (datatype < 0) return error.Hdf5Failure;
        defer _ = c.H5Tclose(datatype);
        const class = c.H5Tget_class(datatype);
        const size = c.H5Tget_size(datatype);
        if (class == c.H5T_FLOAT) return switch (size) {
            4 => .f32,
            8 => .f64,
            else => error.UnsupportedDtype,
        };
        if (class != c.H5T_INTEGER) return error.UnsupportedDtype;
        const sign = c.H5Tget_sign(datatype);
        return switch (sign) {
            c.H5T_SGN_NONE => switch (size) {
                1 => .u8,
                2 => .u16,
                4 => .u32,
                8 => .u64,
                else => error.UnsupportedDtype,
            },
            c.H5T_SGN_2 => switch (size) {
                1 => .i8,
                2 => .i16,
                4 => .i32,
                8 => .i64,
                else => error.UnsupportedDtype,
            },
            else => error.UnsupportedDtype,
        };
    }

    /// Allocate the chunk dimensions, or return null for non-chunked layout.
    pub fn chunkShapeAlloc(self: *Dataset, allocator: std.mem.Allocator) !?[]u64 {
        const properties = c.H5Dget_create_plist(self.id);
        if (properties < 0) return error.Hdf5Failure;
        defer _ = c.H5Pclose(properties);
        if (c.H5Pget_layout(properties) != c.H5D_CHUNKED) return null;
        const rank_value = try self.rank();
        const result = try allocator.alloc(u64, rank_value);
        errdefer allocator.free(result);
        if (c.H5Pget_chunk(properties, @intCast(rank_value), result.ptr) < 0) return error.Hdf5Failure;
        return result;
    }

    /// Write the complete dataset from an exactly sized, type-matched buffer.
    pub fn writeAll(self: *Dataset, comptime T: type, values: []const T) !void {
        const shape = try self.shapeAlloc(self.allocator);
        defer self.allocator.free(shape);
        if (values.len != try elementCount(shape)) return error.BufferSizeMismatch;
        if (try self.dtype() != Dtype.of(T)) return error.DtypeMismatch;
        try requireNonnegative(c.H5Dwrite(
            self.id,
            nativeType(T),
            c.zig_h5s_all(),
            c.zig_h5s_all(),
            c.zig_h5p_default(),
            values.ptr,
        ));
    }

    /// Read the complete dataset into an exactly sized, type-matched buffer.
    pub fn readAll(self: *Dataset, comptime T: type, destination: []T) !void {
        const shape = try self.shapeAlloc(self.allocator);
        defer self.allocator.free(shape);
        if (destination.len != try elementCount(shape)) return error.BufferSizeMismatch;
        if (try self.dtype() != Dtype.of(T)) return error.DtypeMismatch;
        try requireNonnegative(c.H5Dread(
            self.id,
            nativeType(T),
            c.zig_h5s_all(),
            c.zig_h5s_all(),
            c.zig_h5p_default(),
            destination.ptr,
        ));
    }

    /// Write a contiguous hyperslab selected by per-axis offset and count.
    pub fn writeHyperslab(
        self: *Dataset,
        comptime T: type,
        offset: []const u64,
        count: []const u64,
        values: []const T,
    ) !void {
        try self.transferHyperslab(T, .write, offset, count, @constCast(values).ptr, values.len);
    }

    /// Read a contiguous hyperslab selected by per-axis offset and count.
    pub fn readHyperslab(
        self: *Dataset,
        comptime T: type,
        offset: []const u64,
        count: []const u64,
        destination: []T,
    ) !void {
        try self.transferHyperslab(T, .read, offset, count, destination.ptr, destination.len);
    }

    /// Create or update a scalar numeric attribute on the dataset.
    pub fn writeAttribute(self: *Dataset, comptime T: type, name: []const u8, value: T) !void {
        try writeNumericAttribute(self.allocator, self.id, T, name, value);
    }

    /// Test for a dataset attribute with `H5Aexists`.
    pub fn attributeExists(self: *Dataset, name: []const u8) !bool {
        return attributeExistsAt(self.allocator, self.id, name);
    }

    /// Read a scalar numeric dataset attribute into native type `T`.
    pub fn readAttribute(self: *Dataset, comptime T: type, name: []const u8) !T {
        return readNumericAttribute(self.allocator, self.id, T, name);
    }

    /// Replace a dataset attribute with a fixed-length UTF-8 string.
    pub fn writeStringAttribute(self: *Dataset, name: []const u8, value: []const u8) !void {
        try writeStringAttributeAt(self.allocator, self.id, name, value);
    }

    /// Allocate and read a fixed-length string attribute; caller frees it.
    pub fn readStringAttributeAlloc(self: *Dataset, allocator: std.mem.Allocator, name: []const u8) ![]u8 {
        return readStringAttributeAt(self.allocator, allocator, self.id, name);
    }

    const Direction = enum { read, write };

    fn transferHyperslab(
        self: *Dataset,
        comptime T: type,
        direction: Direction,
        offset: []const u64,
        count: []const u64,
        buffer: [*]T,
        buffer_len: usize,
    ) !void {
        const shape = try self.shapeAlloc(self.allocator);
        defer self.allocator.free(shape);
        if (offset.len != shape.len or count.len != shape.len or shape.len == 0) return error.InvalidSelection;
        for (shape, offset, count) |extent, start, length| {
            if (length == 0) return error.InvalidSelection;
            const stop = std.math.add(u64, start, length) catch return error.SelectionOutOfBounds;
            if (stop > extent) return error.SelectionOutOfBounds;
        }
        if (buffer_len != try elementCount(count)) return error.BufferSizeMismatch;
        if (try self.dtype() != Dtype.of(T)) return error.DtypeMismatch;

        const file_space = c.H5Dget_space(self.id);
        if (file_space < 0) return error.Hdf5Failure;
        defer _ = c.H5Sclose(file_space);
        try requireNonnegative(c.H5Sselect_hyperslab(
            file_space,
            c.H5S_SELECT_SET,
            offset.ptr,
            null,
            count.ptr,
            null,
        ));
        const memory_space = c.H5Screate_simple(@intCast(count.len), count.ptr, null);
        if (memory_space < 0) return error.Hdf5Failure;
        defer _ = c.H5Sclose(memory_space);
        const status = switch (direction) {
            .read => c.H5Dread(self.id, nativeType(T), memory_space, file_space, c.zig_h5p_default(), buffer),
            .write => c.H5Dwrite(self.id, nativeType(T), memory_space, file_space, c.zig_h5p_default(), buffer),
        };
        try requireNonnegative(status);
    }
};

/// Query `H5is_library_threadsafe`; this wrapper adds no implicit locking.
pub fn isLibraryThreadSafe() Error!bool {
    try initialize();
    var result: c.hbool_t = false;
    try requireNonnegative(c.H5is_library_threadsafe(&result));
    return result;
}

fn initialize() Error!void {
    try requireNonnegative(c.H5open());
    try requireNonnegative(c.zig_h5_disable_error_printing());
}

fn validateShape(shape: []const u64) Error!void {
    if (shape.len == 0 or shape.len > c.H5S_MAX_RANK) return error.InvalidRank;
    for (shape) |extent| if (extent == 0) return error.InvalidShape;
    _ = try elementCount(shape);
}

fn validateChunk(shape: []const u64, chunk: []const u64) Error!void {
    if (chunk.len != shape.len) return error.InvalidShape;
    for (shape, chunk) |extent, chunk_extent| {
        if (chunk_extent == 0 or chunk_extent > extent) return error.InvalidShape;
    }
}

fn elementCount(shape: []const u64) Error!usize {
    var result: usize = 1;
    for (shape) |extent| {
        const value = std.math.cast(usize, extent) orelse return error.SizeOverflow;
        result = std.math.mul(usize, result, value) catch return error.SizeOverflow;
    }
    return result;
}

fn nativeType(comptime T: type) c.hid_t {
    return switch (T) {
        u8 => c.zig_h5_native_u8(),
        u16 => c.zig_h5_native_u16(),
        u32 => c.zig_h5_native_u32(),
        u64 => c.zig_h5_native_u64(),
        i8 => c.zig_h5_native_i8(),
        i16 => c.zig_h5_native_i16(),
        i32 => c.zig_h5_native_i32(),
        i64 => c.zig_h5_native_i64(),
        f32 => c.zig_h5_native_f32(),
        f64 => c.zig_h5_native_f64(),
        else => @compileError("unsupported HDF5 element type: " ++ @typeName(T)),
    };
}

fn fileType(comptime T: type) c.hid_t {
    return switch (T) {
        u8 => c.zig_h5_file_u8(),
        u16 => c.zig_h5_file_u16(),
        u32 => c.zig_h5_file_u32(),
        u64 => c.zig_h5_file_u64(),
        i8 => c.zig_h5_file_i8(),
        i16 => c.zig_h5_file_i16(),
        i32 => c.zig_h5_file_i32(),
        i64 => c.zig_h5_file_i64(),
        f32 => c.zig_h5_file_f32(),
        f64 => c.zig_h5_file_f64(),
        else => @compileError("unsupported HDF5 element type: " ++ @typeName(T)),
    };
}

fn writeNumericAttribute(
    allocator: std.mem.Allocator,
    location: c.hid_t,
    comptime T: type,
    name: []const u8,
    value: T,
) !void {
    const name_z = try allocator.dupeSentinel(u8, name, 0);
    defer allocator.free(name_z);
    const exists = c.H5Aexists(location, name_z.ptr);
    if (exists < 0) return error.Hdf5Failure;
    const attribute = if (exists > 0)
        c.H5Aopen(location, name_z.ptr, c.zig_h5p_default())
    else blk: {
        const space = c.H5Screate(c.H5S_SCALAR);
        if (space < 0) return error.Hdf5Failure;
        defer _ = c.H5Sclose(space);
        break :blk c.H5Acreate2(
            location,
            name_z.ptr,
            fileType(T),
            space,
            c.zig_h5p_default(),
            c.zig_h5p_default(),
        );
    };
    if (attribute < 0) return error.Hdf5Failure;
    defer _ = c.H5Aclose(attribute);
    var copy = value;
    try requireNonnegative(c.H5Awrite(attribute, nativeType(T), &copy));
}

fn readNumericAttribute(
    allocator: std.mem.Allocator,
    location: c.hid_t,
    comptime T: type,
    name: []const u8,
) !T {
    const name_z = try allocator.dupeSentinel(u8, name, 0);
    defer allocator.free(name_z);
    const attribute = c.H5Aopen(location, name_z.ptr, c.zig_h5p_default());
    if (attribute < 0) return error.AttributeNotFound;
    defer _ = c.H5Aclose(attribute);
    var value: T = undefined;
    try requireNonnegative(c.H5Aread(attribute, nativeType(T), &value));
    return value;
}

fn writeStringAttributeAt(
    allocator: std.mem.Allocator,
    location: c.hid_t,
    name: []const u8,
    value: []const u8,
) !void {
    if (value.len == 0) return error.InvalidShape;
    const name_z = try allocator.dupeSentinel(u8, name, 0);
    defer allocator.free(name_z);
    const exists = c.H5Aexists(location, name_z.ptr);
    if (exists < 0) return error.Hdf5Failure;
    if (exists > 0) try requireNonnegative(c.H5Adelete(location, name_z.ptr));
    const datatype = c.H5Tcopy(c.zig_h5_c_s1());
    if (datatype < 0) return error.Hdf5Failure;
    defer _ = c.H5Tclose(datatype);
    try requireNonnegative(c.H5Tset_size(datatype, value.len));
    try requireNonnegative(c.H5Tset_cset(datatype, c.H5T_CSET_UTF8));
    const space = c.H5Screate(c.H5S_SCALAR);
    if (space < 0) return error.Hdf5Failure;
    defer _ = c.H5Sclose(space);
    const attribute = c.H5Acreate2(
        location,
        name_z.ptr,
        datatype,
        space,
        c.zig_h5p_default(),
        c.zig_h5p_default(),
    );
    if (attribute < 0) return error.Hdf5Failure;
    defer _ = c.H5Aclose(attribute);
    try requireNonnegative(c.H5Awrite(attribute, datatype, value.ptr));
}

fn readStringAttributeAt(
    name_allocator: std.mem.Allocator,
    result_allocator: std.mem.Allocator,
    location: c.hid_t,
    name: []const u8,
) ![]u8 {
    const name_z = try name_allocator.dupeSentinel(u8, name, 0);
    defer name_allocator.free(name_z);
    const attribute = c.H5Aopen(location, name_z.ptr, c.zig_h5p_default());
    if (attribute < 0) return error.AttributeNotFound;
    defer _ = c.H5Aclose(attribute);
    const datatype = c.H5Aget_type(attribute);
    if (datatype < 0) return error.Hdf5Failure;
    defer _ = c.H5Tclose(datatype);
    if (c.H5Tget_class(datatype) != c.H5T_STRING or c.H5Tis_variable_str(datatype) > 0) {
        return error.UnsupportedDtype;
    }
    const size = c.H5Tget_size(datatype);
    const result = try result_allocator.alloc(u8, size);
    errdefer result_allocator.free(result);
    try requireNonnegative(c.H5Aread(attribute, datatype, result.ptr));
    return result;
}

fn attributeExistsAt(
    allocator: std.mem.Allocator,
    location: c.hid_t,
    name: []const u8,
) !bool {
    const name_z = try allocator.dupeSentinel(u8, name, 0);
    defer allocator.free(name_z);
    const result = c.H5Aexists(location, name_z.ptr);
    if (result < 0) return error.Hdf5Failure;
    return result > 0;
}

fn requireNonnegative(value: anytype) Error!void {
    if (value < 0) return error.Hdf5Failure;
}
