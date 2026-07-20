const std = @import("std");
const scalar = @import("../scalar.zig");

pub const c = @import("accelerate_c");

pub const Context = struct {
    pub fn init(_: u32) !Context {
        return .{};
    }

    pub fn deinit(_: *Context) void {}
    pub fn synchronize(_: *Context) !void {}

    pub fn recordEvent(_: *Context) !Event {
        return .{};
    }
};

pub const Event = struct {
    pub fn query(_: *const Event) !bool {
        return true;
    }

    pub fn wait(_: *Event) !void {}
    pub fn deinit(_: *Event) void {}
};

pub fn Buffer(comptime T: type) type {
    return struct {
        values: []T,
    };
}

pub fn allocate(comptime T: type, allocator: std.mem.Allocator, _: *Context, len: usize) !Buffer(T) {
    return .{ .values = try allocator.alloc(T, len) };
}

pub fn release(comptime T: type, allocator: std.mem.Allocator, buffer: *Buffer(T)) void {
    allocator.free(buffer.values);
    buffer.values = &.{};
}

pub fn copyFromHost(
    comptime T: type,
    buffer: *Buffer(T),
    offset: usize,
    stride: usize,
    source: []const T,
) !void {
    for (source, 0..) |value, index| buffer.values[offset + index * stride] = value;
}

pub fn copyToHost(
    comptime T: type,
    buffer: *const Buffer(T),
    offset: usize,
    stride: usize,
    destination: []T,
) !void {
    for (destination, 0..) |*value, index| value.* = buffer.values[offset + index * stride];
}

pub fn asCInt(value: usize) !c_int {
    return std.math.cast(c_int, value) orelse error.DimensionTooLarge;
}

pub fn constPointer(comptime T: type, buffer: *const Buffer(T), offset: usize) *const T {
    return &buffer.values[offset];
}

pub fn mutablePointer(comptime T: type, buffer: *Buffer(T), offset: usize) *T {
    return &buffer.values[offset];
}
