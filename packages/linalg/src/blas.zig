const api = @import("types.zig");

const scalar = api.scalar;
const driver = api.driver;
const Transpose = api.Transpose;
const Real = api.Real;
const Vector = api.Vector;
const VectorView = api.VectorView;
const ConstVectorView = api.ConstVectorView;
const Matrix = api.Matrix;
const MatrixView = api.MatrixView;
const ConstMatrixView = api.ConstMatrixView;

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

pub fn validateMatrixView(comptime T: type, matrix: anytype) !void {
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

pub fn vecMean(comptime T: type, x: ConstVectorView(T)) !Real(T) {
    if (x.len == 0) return error.EmptyVector;
    if (x.len > 1 and x.stride == 0) return error.InvalidStride;
    var sum: T = 0.0;
    for (x.buffer.values) |val| {
        sum += val;
    }
    return sum / @as(T, @floatFromInt(x.len));
}
