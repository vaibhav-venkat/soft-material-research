const std = @import("std");
const build_options = @import("linalg_build_options");
const scalar = @import("scalar.zig");

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

pub const Complex = scalar.Complex;
pub const Complex32 = scalar.Complex32;
pub const Complex64 = scalar.Complex64;
pub const Real = scalar.Real;

fn validateScalar(comptime T: type) void {
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

fn validateVectorPair(comptime T: type, x: ConstVectorView(T), y: anytype) !void {
    if (x.context != y.context) return error.ContextMismatch;
    if (x.len != y.len) return error.DimensionMismatch;
    if ((x.len > 1 and x.stride == 0) or (y.len > 1 and y.stride == 0)) return error.InvalidStride;
}

pub fn copy(comptime T: type, x: ConstVectorView(T), y: VectorView(T)) !void {
    try validateVectorPair(T, x, y);
    try driver.blasCopy(T, x.buffer, x.offset, x.stride, y.buffer, y.offset, y.stride, x.len);
}

pub fn swap(comptime T: type, x: VectorView(T), y: VectorView(T)) !void {
    try validateVectorPair(T, x.asConst(), y);
    try driver.blasSwap(T, x.buffer, x.offset, x.stride, y.buffer, y.offset, y.stride, x.len);
}

pub fn scale(comptime T: type, alpha: T, x: VectorView(T)) !void {
    if (x.len > 1 and x.stride == 0) return error.InvalidStride;
    try driver.blasScale(T, alpha, x.buffer, x.offset, x.stride, x.len);
}

pub fn scaleReal(comptime T: type, alpha: Real(T), x: VectorView(T)) !void {
    if (comptime !scalar.isComplex(T)) @compileError("scaleReal requires a complex vector");
    if (x.len > 1 and x.stride == 0) return error.InvalidStride;
    try driver.blasScaleReal(T, alpha, x.buffer, x.offset, x.stride, x.len);
}

pub fn axpy(comptime T: type, alpha: T, x: ConstVectorView(T), y: VectorView(T)) !void {
    try validateVectorPair(T, x, y);
    try driver.blasAxpy(T, alpha, x.buffer, x.offset, x.stride, y.buffer, y.offset, y.stride, x.len);
}

pub fn dot(comptime T: type, x: ConstVectorView(T), y: ConstVectorView(T)) !T {
    if (comptime scalar.isComplex(T)) @compileError("complex dot products must use dotu or dotc explicitly");
    return dotu(T, x, y);
}

pub fn dotu(comptime T: type, x: ConstVectorView(T), y: ConstVectorView(T)) !T {
    try validateVectorPair(T, x, y);
    return driver.blasDot(T, false, x.buffer, x.offset, x.stride, y.buffer, y.offset, y.stride, x.len);
}

pub fn dotc(comptime T: type, x: ConstVectorView(T), y: ConstVectorView(T)) !T {
    try validateVectorPair(T, x, y);
    return driver.blasDot(T, true, x.buffer, x.offset, x.stride, y.buffer, y.offset, y.stride, x.len);
}

pub fn norm2(comptime T: type, x: ConstVectorView(T)) !Real(T) {
    if (x.len > 1 and x.stride == 0) return error.InvalidStride;
    return driver.blasNorm2(T, x.buffer, x.offset, x.stride, x.len);
}

pub fn absSum(comptime T: type, x: ConstVectorView(T)) !Real(T) {
    if (x.len > 1 and x.stride == 0) return error.InvalidStride;
    return driver.blasAbsSum(T, x.buffer, x.offset, x.stride, x.len);
}

pub fn indexAbsMax(comptime T: type, x: ConstVectorView(T)) !usize {
    if (x.len > 1 and x.stride == 0) return error.InvalidStride;
    return driver.blasIndexAbsMax(T, x.buffer, x.offset, x.stride, x.len);
}

fn scalarZero(comptime T: type) T {
    return if (comptime scalar.isComplex(T)) T.init(0, 0) else 0;
}

fn scalarOne(comptime T: type) T {
    return if (comptime scalar.isComplex(T)) T.init(1, 0) else 1;
}

fn operatedRows(comptime T: type, matrix: ConstMatrixView(T), operation: Transpose) usize {
    return if (operation == .none) matrix.rows else matrix.cols;
}

fn operatedCols(comptime T: type, matrix: ConstMatrixView(T), operation: Transpose) usize {
    return if (operation == .none) matrix.cols else matrix.rows;
}

fn validateMatrixView(comptime T: type, matrix: anytype) !void {
    _ = T;
    if (matrix.rows == 0 or matrix.cols == 0) return error.EmptyInput;
    if (matrix.leading_dimension < matrix.rows) return error.InvalidStride;
}

pub fn gemvInto(
    comptime T: type,
    operation: Transpose,
    alpha: T,
    a: ConstMatrixView(T),
    x: ConstVectorView(T),
    beta: T,
    y: VectorView(T),
) !void {
    try validateMatrixView(T, a);
    if (a.context != x.context or a.context != y.context) return error.ContextMismatch;
    if (x.len != operatedCols(T, a, operation) or y.len != operatedRows(T, a, operation)) return error.DimensionMismatch;
    if (a.buffer == x.buffer or a.buffer == y.buffer or x.buffer == y.buffer) return error.OutputAliased;
    try driver.blasGemv(
        T,
        @intFromEnum(operation),
        alpha,
        a.buffer,
        a.offset,
        a.rows,
        a.cols,
        a.leading_dimension,
        x.buffer,
        x.offset,
        x.stride,
        beta,
        y.buffer,
        y.offset,
        y.stride,
    );
}

pub fn matvec(comptime T: type, operation: Transpose, a: ConstMatrixView(T), x: ConstVectorView(T)) !Vector(T) {
    const output_len = operatedRows(T, a, operation);
    var result = try Vector(T).init(@constCast(a.context), output_len);
    errdefer result.deinit();
    try gemvInto(T, operation, scalarOne(T), a, x, scalarZero(T), result.view());
    return result;
}

fn gerIntoInternal(
    comptime T: type,
    conjugate_y: bool,
    alpha: T,
    x: ConstVectorView(T),
    y: ConstVectorView(T),
    a: MatrixView(T),
) !void {
    try validateMatrixView(T, a);
    if (x.context != y.context or x.context != a.context) return error.ContextMismatch;
    if (a.rows != x.len or a.cols != y.len) return error.DimensionMismatch;
    if (a.buffer == x.buffer or a.buffer == y.buffer) return error.OutputAliased;
    try driver.blasGer(T, conjugate_y, alpha, x.buffer, x.offset, x.stride, y.buffer, y.offset, y.stride, a.buffer, a.offset, a.rows, a.cols, a.leading_dimension);
}

pub fn gerInto(comptime T: type, alpha: T, x: ConstVectorView(T), y: ConstVectorView(T), a: MatrixView(T)) !void {
    if (comptime scalar.isComplex(T)) @compileError("complex outer products must use geruInto or gercInto explicitly");
    try gerIntoInternal(T, false, alpha, x, y, a);
}

pub fn geruInto(comptime T: type, alpha: T, x: ConstVectorView(T), y: ConstVectorView(T), a: MatrixView(T)) !void {
    if (comptime !scalar.isComplex(T)) @compileError("geruInto requires complex inputs");
    try gerIntoInternal(T, false, alpha, x, y, a);
}

pub fn gercInto(comptime T: type, alpha: T, x: ConstVectorView(T), y: ConstVectorView(T), a: MatrixView(T)) !void {
    if (comptime !scalar.isComplex(T)) @compileError("gercInto requires complex inputs");
    try gerIntoInternal(T, true, alpha, x, y, a);
}

pub fn gemmInto(
    comptime T: type,
    operation_a: Transpose,
    operation_b: Transpose,
    alpha: T,
    a: ConstMatrixView(T),
    b: ConstMatrixView(T),
    beta: T,
    output: MatrixView(T),
) !void {
    try validateMatrixView(T, a);
    try validateMatrixView(T, b);
    try validateMatrixView(T, output);
    if (a.context != b.context or a.context != output.context) return error.ContextMismatch;
    const m = operatedRows(T, a, operation_a);
    const k = operatedCols(T, a, operation_a);
    const b_rows = operatedRows(T, b, operation_b);
    const n = operatedCols(T, b, operation_b);
    if (k != b_rows or output.rows != m or output.cols != n) return error.DimensionMismatch;
    if (output.buffer == a.buffer or output.buffer == b.buffer) return error.OutputAliased;
    try driver.blasGemm(
        T,
        @intFromEnum(operation_a),
        @intFromEnum(operation_b),
        m,
        n,
        k,
        alpha,
        a.buffer,
        a.offset,
        a.leading_dimension,
        b.buffer,
        b.offset,
        b.leading_dimension,
        beta,
        output.buffer,
        output.offset,
        output.leading_dimension,
    );
}

pub fn matmul(comptime T: type, operation_a: Transpose, operation_b: Transpose, a: ConstMatrixView(T), b: ConstMatrixView(T)) !Matrix(T) {
    const rows = operatedRows(T, a, operation_a);
    const cols = operatedCols(T, b, operation_b);
    var result = try Matrix(T).init(@constCast(a.context), rows, cols);
    errdefer result.deinit();
    try gemmInto(T, operation_a, operation_b, scalarOne(T), a, b, scalarZero(T), result.view());
    return result;
}

fn expectComplexApprox(comptime T: type, expected: T, actual: T, tolerance: Real(T)) !void {
    try std.testing.expectApproxEqAbs(expected.re, actual.re, tolerance);
    try std.testing.expectApproxEqAbs(expected.im, actual.im, tolerance);
}

fn testRealLevelOne(comptime T: type, tolerance: T) !void {
    var context = try Context.init(std.testing.allocator, .{});
    defer context.deinit();
    const x_values = [_]T{ 1, -2, 3 };
    const y_values = [_]T{ 4, 5, 6 };
    var x = try Vector(T).fromHost(&context, &x_values);
    defer x.deinit();
    var y = try Vector(T).fromHost(&context, &y_values);
    defer y.deinit();

    try std.testing.expectApproxEqAbs(@as(T, 12), try dot(T, x.constView(), y.constView()), tolerance);
    try std.testing.expectApproxEqAbs(@sqrt(@as(T, 14)), try norm2(T, x.constView()), tolerance);
    try std.testing.expectApproxEqAbs(@as(T, 6), try absSum(T, x.constView()), tolerance);
    try std.testing.expectEqual(@as(usize, 2), try indexAbsMax(T, x.constView()));

    try scale(T, 2, y.view());
    try axpy(T, -1, x.constView(), y.view());
    var output: [3]T = undefined;
    y.copyToHost(&output);
    try std.testing.expectEqualSlices(T, &[_]T{ 7, 12, 9 }, &output);

    try copy(T, x.constView(), y.view());
    try swap(T, x.view(), y.view());
    x.copyToHost(&output);
    try std.testing.expectEqualSlices(T, &x_values, &output);
}

fn testComplexLevelOne(comptime T: type, tolerance: Real(T)) !void {
    var context = try Context.init(std.testing.allocator, .{});
    defer context.deinit();
    const x_values = [_]T{ .init(1, 2), .init(-3, 1) };
    const y_values = [_]T{ .init(2, -1), .init(1, 4) };
    var x = try Vector(T).fromHost(&context, &x_values);
    defer x.deinit();
    var y = try Vector(T).fromHost(&context, &y_values);
    defer y.deinit();

    try expectComplexApprox(T, T.init(-3, -8), try dotu(T, x.constView(), y.constView()), tolerance);
    try expectComplexApprox(T, T.init(1, -18), try dotc(T, x.constView(), y.constView()), tolerance);
    try std.testing.expectApproxEqAbs(@sqrt(@as(Real(T), 15)), try norm2(T, x.constView()), tolerance);
    try std.testing.expectApproxEqAbs(@as(Real(T), 7), try absSum(T, x.constView()), tolerance);
    try std.testing.expectEqual(@as(usize, 1), try indexAbsMax(T, x.constView()));

    try scaleReal(T, 2, x.view());
    try scale(T, T.init(0, 0.5), x.view());
    try axpy(T, T.init(1, 0), y.constView(), x.view());
    var output: [2]T = undefined;
    x.copyToHost(&output);
    try expectComplexApprox(T, T.init(0, 0), output[0], tolerance);
    try expectComplexApprox(T, T.init(0, 1), output[1], tolerance);
}

fn testRealMatrixBlas(comptime T: type, tolerance: T) !void {
    var context = try Context.init(std.testing.allocator, .{});
    defer context.deinit();
    const a_values = [_]T{ 1, 4, 2, 5, 3, 6 };
    var a = try Matrix(T).fromHost(&context, 2, 3, &a_values);
    defer a.deinit();
    var x = try Vector(T).fromHost(&context, &[_]T{ 1, 2, 1 });
    defer x.deinit();
    var y = try Vector(T).fromHost(&context, &[_]T{ 1, 1 });
    defer y.deinit();
    try gemvInto(T, .none, 2, a.constView(), x.constView(), 0.5, y.view());
    var y_output: [2]T = undefined;
    y.copyToHost(&y_output);
    try std.testing.expectApproxEqAbs(@as(T, 16.5), y_output[0], tolerance);
    try std.testing.expectApproxEqAbs(@as(T, 40.5), y_output[1], tolerance);

    var trans_x = try Vector(T).fromHost(&context, &[_]T{ 1, 2 });
    defer trans_x.deinit();
    var trans_y = try matvec(T, .transpose, a.constView(), trans_x.constView());
    defer trans_y.deinit();
    var trans_output: [3]T = undefined;
    trans_y.copyToHost(&trans_output);
    try std.testing.expectEqualSlices(T, &[_]T{ 9, 12, 15 }, &trans_output);

    const b_values = [_]T{ 1, 0, 1, 2, 1, 0 };
    var b = try Matrix(T).fromHost(&context, 3, 2, &b_values);
    defer b.deinit();
    var product = try matmul(T, .none, .none, a.constView(), b.constView());
    defer product.deinit();
    var product_output: [4]T = undefined;
    product.copyToHost(&product_output);
    try std.testing.expectEqualSlices(T, &[_]T{ 4, 10, 4, 13 }, &product_output);

    var outer_x = try Vector(T).fromHost(&context, &[_]T{ 1, 2 });
    defer outer_x.deinit();
    var outer_y = try Vector(T).fromHost(&context, &[_]T{ 3, 4, 5 });
    defer outer_y.deinit();
    var outer = try Matrix(T).fromHost(&context, 2, 3, &[_]T{ 0, 0, 0, 0, 0, 0 });
    defer outer.deinit();
    try gerInto(T, 1, outer_x.constView(), outer_y.constView(), outer.view());
    var outer_output: [6]T = undefined;
    outer.copyToHost(&outer_output);
    try std.testing.expectEqualSlices(T, &[_]T{ 3, 6, 4, 8, 5, 10 }, &outer_output);
}

fn testComplexMatrixBlas(comptime T: type, tolerance: Real(T)) !void {
    var context = try Context.init(std.testing.allocator, .{});
    defer context.deinit();
    const a_values = [_]T{ .init(1, 1), .init(2, 0), .init(0, 1), .init(3, -1) };
    var a = try Matrix(T).fromHost(&context, 2, 2, &a_values);
    defer a.deinit();
    var x = try Vector(T).fromHost(&context, &[_]T{ .init(1, 0), .init(0, 1) });
    defer x.deinit();
    var y = try matvec(T, .conjugate_transpose, a.constView(), x.constView());
    defer y.deinit();
    var y_output: [2]T = undefined;
    y.copyToHost(&y_output);
    try expectComplexApprox(T, T.init(1, 1), y_output[0], tolerance);
    try expectComplexApprox(T, T.init(-1, 2), y_output[1], tolerance);

    var identity = try Matrix(T).fromHost(&context, 2, 2, &[_]T{ .init(1, 0), .init(0, 0), .init(0, 0), .init(1, 0) });
    defer identity.deinit();
    var product = try matmul(T, .none, .none, a.constView(), identity.constView());
    defer product.deinit();
    var product_output: [4]T = undefined;
    product.copyToHost(&product_output);
    for (a_values, product_output) |expected, actual| try expectComplexApprox(T, expected, actual, tolerance);

    var outer_x = try Vector(T).fromHost(&context, &[_]T{.init(1, 1)});
    defer outer_x.deinit();
    var outer_y = try Vector(T).fromHost(&context, &[_]T{.init(2, 1)});
    defer outer_y.deinit();
    var outer_u = try Matrix(T).fromHost(&context, 1, 1, &[_]T{.init(0, 0)});
    defer outer_u.deinit();
    var outer_c = try Matrix(T).fromHost(&context, 1, 1, &[_]T{.init(0, 0)});
    defer outer_c.deinit();
    try geruInto(T, T.init(1, 0), outer_x.constView(), outer_y.constView(), outer_u.view());
    try gercInto(T, T.init(1, 0), outer_x.constView(), outer_y.constView(), outer_c.view());
    var value: [1]T = undefined;
    outer_u.copyToHost(&value);
    try expectComplexApprox(T, T.init(1, 3), value[0], tolerance);
    outer_c.copyToHost(&value);
    try expectComplexApprox(T, T.init(3, 1), value[0], tolerance);
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

test "real Level 1 BLAS" {
    try testRealLevelOne(f32, 1.0e-5);
    try testRealLevelOne(f64, 1.0e-12);
}

test "complex Level 1 BLAS" {
    try testComplexLevelOne(Complex32, 1.0e-5);
    try testComplexLevelOne(Complex64, 1.0e-12);
}

test "real matrix BLAS" {
    try testRealMatrixBlas(f32, 1.0e-5);
    try testRealMatrixBlas(f64, 1.0e-12);
}

test "complex matrix BLAS" {
    try testComplexMatrixBlas(Complex32, 1.0e-5);
    try testComplexMatrixBlas(Complex64, 1.0e-12);
}
