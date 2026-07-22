const std = @import("std");

pub fn Complex(comptime T: type) type {
    switch (T) {
        f32, f64 => {},
        else => @compileError("Complex components must be f32 or f64"),
    }
    return extern struct {
        re: T align(2 * @alignOf(T)),
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
    return switch (T) {
        f32, Complex32 => f32,
        else => f64,
    };
}

pub fn validate(comptime T: type) void {
    switch (T) {
        f32, f64, Complex32, Complex64 => {},
        else => @compileError("linalg supports f32, f64, Complex(f32), and Complex(f64)"),
    }
}

pub fn isComplex(comptime T: type) bool {
    validate(T);
    return switch (T) {
        Complex32, Complex64 => true,
        else => false,
    };
}
