//! Small numerical helpers shared by particle properties.

const std = @import("std");
const coordinates = @import("../data_structures/coordinates.zig");

pub fn transformCylindrical(
    positions: coordinates.CartesianPositions.Slice,
    cylindrical: coordinates.CylindricalCoordinates.Slice,
) !void {
    if (positions.len != cylindrical.len) return error.BufferSizeMismatch;
    const lane_count = std.simd.suggestVectorLength(f32) orelse 4;
    const Vector = @Vector(lane_count, f32);
    var particle: usize = 0;
    const input_x = positions.items(.x);
    const input_y = positions.items(.y);
    const input_z = positions.items(.z);
    const output_x = cylindrical.items(.x);
    const output_theta = cylindrical.items(.theta);
    const output_r = cylindrical.items(.r);
    while (particle + lane_count <= positions.len) : (particle += lane_count) {
        const x: Vector = input_x[particle..][0..lane_count].*;
        const y: Vector = input_y[particle..][0..lane_count].*;
        const z: Vector = input_z[particle..][0..lane_count].*;
        const radius = @sqrt(y * y + z * z);
        output_x[particle..][0..lane_count].* = x;
        output_theta[particle..][0..lane_count].* = atan2Simd(lane_count, y, z);
        output_r[particle..][0..lane_count].* = radius;
    }
    while (particle < positions.len) : (particle += 1) {
        output_x[particle] = input_x[particle];
        output_theta[particle] = normalizedAtan2(input_y[particle], input_z[particle]);
        output_r[particle] = @sqrt(input_y[particle] * input_y[particle] + input_z[particle] * input_z[particle]);
    }
}

fn atan2Simd(comptime n: comptime_int, y: @Vector(n, f32), x: @Vector(n, f32)) @Vector(n, f32) {
    const V = @Vector(n, f32);
    const zero: V = @splat(0.0);
    const one: V = @splat(1.0);
    const pi: V = @splat(std.math.pi);
    const two_pi: V = @splat(2.0 * std.math.pi);
    const half_pi: V = @splat(std.math.pi / 2.0);
    const ax = @abs(x);
    const ay = @abs(y);
    const swap = ay > ax;
    const lo = @select(f32, swap, ax, ay);
    const hi = @select(f32, swap, ay, ax);
    const is_zero = hi == zero;
    const r = lo / @select(f32, is_zero, one, hi);
    const r2 = r * r;
    var p: V = @splat(0.02084517);
    p = @mulAdd(V, p, r2, @as(V, @splat(-0.08515649)));
    p = @mulAdd(V, p, r2, @as(V, @splat(0.18015942)));
    p = @mulAdd(V, p, r2, @as(V, @splat(-0.33030482)));
    p = @mulAdd(V, p, r2, @as(V, @splat(0.99986633)));
    var angle = r * p;
    angle = @select(f32, swap, half_pi - angle, angle);
    angle = @select(f32, x < zero, pi - angle, angle);
    angle = @select(f32, y < zero, -angle, angle);
    angle = @select(f32, angle < zero, angle + two_pi, angle);
    return @select(f32, is_zero, zero, angle);
}

fn normalizedAtan2(y: f32, z: f32) f32 {
    if (y == 0 and z == 0) return 0;
    const angle = std.math.atan2(y, z);
    return if (angle < 0) angle + 2.0 * std.math.pi else angle;
}

pub fn activeDirection(quaternion: coordinates.Quaternion) !coordinates.CartesianVector {
    const norm_squared = quaternion.w * quaternion.w + quaternion.x * quaternion.x +
        quaternion.y * quaternion.y + quaternion.z * quaternion.z;
    if (!std.math.isFinite(norm_squared) or norm_squared <= 0) return error.InvalidQuaternion;
    const inverse_norm = 1.0 / @sqrt(norm_squared);
    const w = quaternion.w * inverse_norm;
    const x = quaternion.x * inverse_norm;
    const y = quaternion.y * inverse_norm;
    const z = quaternion.z * inverse_norm;
    return .{
        .x = 1.0 - 2.0 * (y * y + z * z),
        .y = 2.0 * (x * y + w * z),
        .z = 2.0 * (x * z - w * y),
    };
}

/// Python-compatible component order: `(x, radial, azimuthal)`.
pub fn cylindricalVector(vector: coordinates.CartesianVector, theta: f32) coordinates.CylindricalVector {
    const sin_theta = @sin(theta);
    const cos_theta = @cos(theta);
    return .{
        .x = vector.x,
        .r = vector.y * sin_theta + vector.z * cos_theta,
        .theta = vector.y * cos_theta - vector.z * sin_theta,
    };
}

pub fn gaussianWeight(distance_squared: f32, width: f32) !f32 {
    if (!std.math.isFinite(width) or width <= 0) return error.InvalidGaussianWidth;
    if (!std.math.isFinite(distance_squared) or distance_squared < 0) return error.InvalidDistance;
    const normalization = std.math.pow(f32, 2.0 * std.math.pi, 1.5) * width * width * width;
    return @exp(-0.5 * distance_squared / (width * width)) / normalization;
}

pub fn sixfoldMisorientation(left: f32, right: f32) f32 {
    const period: f32 = std.math.pi / 3.0;
    const difference = @mod(left - right, period);
    return @min(difference, period - difference);
}
