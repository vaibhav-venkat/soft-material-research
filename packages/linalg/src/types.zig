const std = @import("std");
const build_options = @import("linalg_build_options");
pub const scalar = @import("scalar.zig");

pub const driver = if (std.mem.eql(u8, build_options.backend, "accelerate"))
    @import("backend/accelerate.zig")
else
    @import("backend/cuda.zig");

pub const Backend = enum { accelerate, cuda };
pub const compiled_backend: Backend = if (std.mem.eql(u8, build_options.backend, "accelerate"))
    .accelerate
else
    .cuda;

pub const Transpose = enum { none, transpose, conjugate_transpose };

pub const Complex = scalar.Complex;
pub const Complex32 = scalar.Complex32;
pub const Complex64 = scalar.Complex64;
pub const Real = scalar.Real;

pub fn validateScalar(comptime T: type) void {
    scalar.validate(T);
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
        const Self = @This();

        context: *Context,
        buffer: *driver.Buffer(T),
        offset: usize,
        len: usize,
        stride: usize,

        pub fn asConst(self: Self) ConstVectorView(T) {
            return .{
                .context = self.context,
                .buffer = self.buffer,
                .offset = self.offset,
                .len = self.len,
                .stride = self.stride,
            };
        }
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
        const Self = @This();

        context: *Context,
        buffer: *driver.Buffer(T),
        offset: usize,
        rows: usize,
        cols: usize,
        leading_dimension: usize,

        pub fn asConst(self: Self) ConstMatrixView(T) {
            return .{
                .context = self.context,
                .buffer = self.buffer,
                .offset = self.offset,
                .rows = self.rows,
                .cols = self.cols,
                .leading_dimension = self.leading_dimension,
            };
        }
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
