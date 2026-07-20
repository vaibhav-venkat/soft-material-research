const std = @import("std");

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
    validate(T);
    return if (T == f32 or T == Complex32) f32 else f64;
}

pub fn validate(comptime T: type) void {
    if (T != f32 and T != f64 and T != Complex32 and T != Complex64) {
        @compileError("linalg supports f32, f64, Complex(f32), and Complex(f64)");
    }
}

pub fn isComplex(comptime T: type) bool {
    validate(T);
    return T == Complex32 or T == Complex64;
}
