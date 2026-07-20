const api = @import("types.zig");
const blas = @import("blas.zig");

const driver = api.driver;
const Real = api.Real;
const Vector = api.Vector;
const Matrix = api.Matrix;
const ConstMatrixView = api.ConstMatrixView;
const validateScalar = api.validateScalar;

pub const SvdMode = enum { values_only, thin };

pub const SvdOptions = struct {
    mode: SvdMode = .thin,
};

pub fn SvdResult(comptime T: type) type {
    validateScalar(T);
    return struct {
        const Self = @This();

        u: ?Matrix(T),
        singular_values: Vector(Real(T)),
        vh: ?Matrix(T),

        pub fn deinit(self: *Self) void {
            if (self.u) |*matrix| matrix.deinit();
            self.singular_values.deinit();
            if (self.vh) |*matrix| matrix.deinit();
            self.u = null;
            self.vh = null;
        }
    };
}

pub fn svd(comptime T: type, input: ConstMatrixView(T), options: SvdOptions) !SvdResult(T) {
    try blas.validateMatrixView(T, input);
    if (input.rows < input.cols) return error.WideMatrixUnsupported;

    const context = @constCast(input.context);
    var singular_values = try Vector(Real(T)).init(context, input.cols);
    errdefer singular_values.deinit();
    var u: ?Matrix(T) = null;
    errdefer if (u) |*matrix| matrix.deinit();
    var vh: ?Matrix(T) = null;
    errdefer if (vh) |*matrix| matrix.deinit();
    if (options.mode == .thin) {
        u = try Matrix(T).init(context, input.rows, input.cols);
        vh = try Matrix(T).init(context, input.cols, input.cols);
    }

    try driver.lapackSvd(
        T,
        context.allocator,
        options.mode == .thin,
        input.buffer,
        input.offset,
        input.rows,
        input.cols,
        input.leading_dimension,
        if (u) |*matrix| &matrix.buffer else null,
        &singular_values.buffer,
        if (vh) |*matrix| &matrix.buffer else null,
    );
    return .{ .u = u, .singular_values = singular_values, .vh = vh };
}

pub fn LeastSquaresOptions(comptime T: type) type {
    validateScalar(T);
    return struct {
        rcond: ?Real(T) = null,
    };
}

pub fn LeastSquaresResult(comptime T: type) type {
    validateScalar(T);
    return struct {
        const Self = @This();

        solution: Matrix(T),
        singular_values: Vector(Real(T)),
        rank: usize,

        pub fn deinit(self: *Self) void {
            self.solution.deinit();
            self.singular_values.deinit();
            self.rank = 0;
        }
    };
}

pub fn leastSquares(
    comptime T: type,
    coefficients: ConstMatrixView(T),
    right_hand_sides: ConstMatrixView(T),
    options: LeastSquaresOptions(T),
) !LeastSquaresResult(T) {
    try blas.validateMatrixView(T, coefficients);
    try blas.validateMatrixView(T, right_hand_sides);
    if (coefficients.context != right_hand_sides.context) return error.ContextMismatch;
    if (coefficients.rows < coefficients.cols) return error.WideMatrixUnsupported;
    if (right_hand_sides.rows != coefficients.rows) return error.DimensionMismatch;

    const context = @constCast(coefficients.context);
    var solution = try Matrix(T).init(context, coefficients.cols, right_hand_sides.cols);
    errdefer solution.deinit();
    var singular_values = try Vector(Real(T)).init(context, coefficients.cols);
    errdefer singular_values.deinit();
    const rank = try driver.lapackLeastSquares(
        T,
        context.allocator,
        coefficients.buffer,
        coefficients.offset,
        coefficients.rows,
        coefficients.cols,
        coefficients.leading_dimension,
        right_hand_sides.buffer,
        right_hand_sides.offset,
        right_hand_sides.cols,
        right_hand_sides.leading_dimension,
        options.rcond orelse -1,
        &solution.buffer,
        &singular_values.buffer,
    );
    return .{ .solution = solution, .singular_values = singular_values, .rank = rank };
}
