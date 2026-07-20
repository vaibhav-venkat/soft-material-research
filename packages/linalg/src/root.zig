const std = @import("std");
const build_options = @import("linalg_build_options");

const driver = if (std.mem.eql(u8, build_options.backend, "accelerate"))
    @import("backend/accelerate.zig")
else
    @import("backend/cuda.zig");

pub const Backend = enum { accelerate, cuda };
pub const compiled_backend: Backend = if (std.mem.eql(u8, build_options.backend, "accelerate"))
    .accelerate
else
    .cuda;

pub const Transpose = enum { none, transpose, conjugate_transpose };

pub fn Complex(comptime T: type) type {
    if (T != f32 and T != f64) @compileError("Complex components must be f32 or f64");
    return extern struct {
        re: T,
        im: T,

        const Self = @This();

        pub fn init(re: T, im: T) Self {
            return .{ .re = re, .im = im };
        }

        pub fn conjugate(self: Self) Self {
            return .{ .re = self.re, .im = -self.im };
        }

        pub fn fromStd(value: std.math.Complex(T)) Self {
            return .{ .re = value.re, .im = value.im };
        }

        pub fn toStd(self: Self) std.math.Complex(T) {
            return .init(self.re, self.im);
        }
    };
}

pub const Complex32 = Complex(f32);
pub const Complex64 = Complex(f64);

pub fn Real(comptime T: type) type {
    validateScalar(T);
    return if (T == f32 or T == Complex32) f32 else f64;
}

fn validateScalar(comptime T: type) void {
    if (T != f32 and T != f64 and T != Complex32 and T != Complex64) {
        @compileError("linalg supports f32, f64, Complex(f32), and Complex(f64)");
    }
}

pub const ContextOptions = struct {
    device_ordinal: u32 = 0,
};

pub const BackendInfo = struct {
    backend: Backend,
    device_ordinal: u32,
};

pub const Context = struct {
    allocator: std.mem.Allocator,
    inner: driver.Context,
    device_ordinal: u32,

    pub fn init(allocator: std.mem.Allocator, options: ContextOptions) !Context {
        if (compiled_backend == .accelerate and options.device_ordinal != 0) {
            return error.InvalidDeviceOrdinal;
        }
        return .{
            .allocator = allocator,
            .inner = driver.Context.init(),
            .device_ordinal = options.device_ordinal,
        };
    }

    pub fn deinit(self: *Context) void {
        self.inner.synchronize();
        self.inner.deinit();
    }

    pub fn synchronize(self: *Context) !void {
        self.inner.synchronize();
    }

    pub fn recordEvent(_: *Context) !Event {
        return .{ .inner = .{} };
    }

    pub fn backendInfo(self: *const Context) BackendInfo {
        return .{ .backend = compiled_backend, .device_ordinal = self.device_ordinal };
    }
};

pub const Event = struct {
    inner: driver.Event,

    pub fn query(self: *const Event) !bool {
        return self.inner.query();
    }

    pub fn wait(self: *Event) !void {
        self.inner.wait();
    }

    pub fn deinit(self: *Event) void {
        self.inner.deinit();
    }
};

pub fn VectorView(comptime T: type) type {
    validateScalar(T);
    return struct {
        context: *Context,
        buffer: *driver.Buffer(T),
        offset: usize,
        len: usize,
        stride: usize,
    };
}

pub fn ConstVectorView(comptime T: type) type {
    validateScalar(T);
    return struct {
        context: *const Context,
        buffer: *const driver.Buffer(T),
        offset: usize,
        len: usize,
        stride: usize,
    };
}

pub fn MatrixView(comptime T: type) type {
    validateScalar(T);
    return struct {
        context: *Context,
        buffer: *driver.Buffer(T),
        offset: usize,
        rows: usize,
        cols: usize,
        leading_dimension: usize,
    };
}

pub fn ConstMatrixView(comptime T: type) type {
    validateScalar(T);
    return struct {
        context: *const Context,
        buffer: *const driver.Buffer(T),
        offset: usize,
        rows: usize,
        cols: usize,
        leading_dimension: usize,
    };
}

pub fn Vector(comptime T: type) type {
    validateScalar(T);
    return struct {
        const Self = @This();

        context: *Context,
        buffer: driver.Buffer(T),
        len: usize,

        pub fn init(context: *Context, len: usize) !Self {
            return .{
                .context = context,
                .buffer = try driver.allocate(T, context.allocator, len),
                .len = len,
            };
        }

        pub fn fromHost(context: *Context, values: []const T) !Self {
            var result = try Self.init(context, values.len);
            errdefer result.deinit();
            result.copyFromHost(values);
            return result;
        }

        pub fn deinit(self: *Self) void {
            self.context.inner.synchronize();
            driver.release(T, self.context.allocator, &self.buffer);
            self.len = 0;
        }

        pub fn view(self: *Self) VectorView(T) {
            return .{ .context = self.context, .buffer = &self.buffer, .offset = 0, .len = self.len, .stride = 1 };
        }

        pub fn constView(self: *const Self) ConstVectorView(T) {
            return .{ .context = self.context, .buffer = &self.buffer, .offset = 0, .len = self.len, .stride = 1 };
        }

        pub fn copyFromHost(self: *Self, values: []const T) void {
            std.debug.assert(values.len == self.len);
            driver.copyFromHost(T, &self.buffer, 0, 1, values);
        }

        pub fn copyToHost(self: *const Self, destination: []T) void {
            std.debug.assert(destination.len == self.len);
            driver.copyToHost(T, &self.buffer, 0, 1, destination);
        }
    };
}

pub fn Matrix(comptime T: type) type {
    validateScalar(T);
    return struct {
        const Self = @This();

        context: *Context,
        buffer: driver.Buffer(T),
        rows: usize,
        cols: usize,
        leading_dimension: usize,

        pub fn init(context: *Context, rows: usize, cols: usize) !Self {
            return .{
                .context = context,
                .buffer = try driver.allocate(T, context.allocator, try std.math.mul(usize, rows, cols)),
                .rows = rows,
                .cols = cols,
                .leading_dimension = rows,
            };
        }

        pub fn fromHost(context: *Context, rows: usize, cols: usize, values: []const T) !Self {
            if (values.len != try std.math.mul(usize, rows, cols)) return error.DimensionMismatch;
            var result = try Self.init(context, rows, cols);
            errdefer result.deinit();
            result.copyFromHost(values);
            return result;
        }

        pub fn deinit(self: *Self) void {
            self.context.inner.synchronize();
            driver.release(T, self.context.allocator, &self.buffer);
            self.rows = 0;
            self.cols = 0;
            self.leading_dimension = 0;
        }

        pub fn view(self: *Self) MatrixView(T) {
            return .{
                .context = self.context,
                .buffer = &self.buffer,
                .offset = 0,
                .rows = self.rows,
                .cols = self.cols,
                .leading_dimension = self.leading_dimension,
            };
        }

        pub fn constView(self: *const Self) ConstMatrixView(T) {
            return .{
                .context = self.context,
                .buffer = &self.buffer,
                .offset = 0,
                .rows = self.rows,
                .cols = self.cols,
                .leading_dimension = self.leading_dimension,
            };
        }

        pub fn subview(self: *Self, row_index: usize, col_index: usize, rows: usize, cols: usize) !MatrixView(T) {
            if (row_index > self.rows or rows > self.rows - row_index or col_index > self.cols or cols > self.cols - col_index) {
                return error.DimensionMismatch;
            }
            return .{
                .context = self.context,
                .buffer = &self.buffer,
                .offset = row_index + col_index * self.leading_dimension,
                .rows = rows,
                .cols = cols,
                .leading_dimension = self.leading_dimension,
            };
        }

        pub fn row(self: *Self, index: usize) !VectorView(T) {
            if (index >= self.rows) return error.DimensionMismatch;
            return .{ .context = self.context, .buffer = &self.buffer, .offset = index, .len = self.cols, .stride = self.leading_dimension };
        }

        pub fn column(self: *Self, index: usize) !VectorView(T) {
            if (index >= self.cols) return error.DimensionMismatch;
            return .{ .context = self.context, .buffer = &self.buffer, .offset = index * self.leading_dimension, .len = self.rows, .stride = 1 };
        }

        pub fn copyFromHost(self: *Self, values: []const T) void {
            std.debug.assert(values.len == self.rows * self.cols);
            driver.copyFromHost(T, &self.buffer, 0, 1, values);
        }

        pub fn copyToHost(self: *const Self, destination: []T) void {
            std.debug.assert(destination.len == self.rows * self.cols);
            driver.copyToHost(T, &self.buffer, 0, 1, destination);
        }
    };
}

test "complex representation and conversions" {
    const value = Complex32.init(2.5, -3.0);
    try std.testing.expectEqual(@sizeOf([2]f32), @sizeOf(Complex32));
    try std.testing.expectEqual(@alignOf(f32), @alignOf(Complex32));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(Complex32, "re"));
    try std.testing.expectEqual(@sizeOf(f32), @offsetOf(Complex32, "im"));
    try std.testing.expectEqual(value, Complex32.fromStd(value.toStd()));
    try std.testing.expectEqual(Complex32.init(2.5, 3.0), value.conjugate());
}

test "backend-owned vector and matrix round trips" {
    var context = try Context.init(std.testing.allocator, .{});
    defer context.deinit();

    const vector_values = [_]Complex64{
        .init(1.0, 2.0),
        .init(-3.0, 4.0),
    };
    var vector = try Vector(Complex64).fromHost(&context, &vector_values);
    defer vector.deinit();
    var vector_output: [2]Complex64 = undefined;
    vector.copyToHost(&vector_output);
    try std.testing.expectEqualSlices(Complex64, &vector_values, &vector_output);

    const matrix_values = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var matrix = try Matrix(f64).fromHost(&context, 3, 2, &matrix_values);
    defer matrix.deinit();
    const row = try matrix.row(1);
    const column = try matrix.column(1);
    const submatrix = try matrix.subview(1, 0, 2, 2);
    try std.testing.expectEqual(@as(usize, 3), row.stride);
    try std.testing.expectEqual(@as(usize, 3), column.len);
    try std.testing.expectEqual(@as(usize, 1), submatrix.offset);
    try std.testing.expectEqual(@as(usize, 3), submatrix.leading_dimension);

    var matrix_output: [6]f64 = undefined;
    matrix.copyToHost(&matrix_output);
    try std.testing.expectEqualSlices(f64, &matrix_values, &matrix_output);
}
