//! Shared structure-of-arrays particle storage.

const std = @import("std");

pub const CartesianPosition = struct {
    x: f32,
    y: f32,
    z: f32,
};

pub const CylindricalCoordinate = struct {
    x: f32,
    theta: f32,
    r: f32,
};

pub const Quaternion = struct {
    w: f32,
    x: f32,
    y: f32,
    z: f32,
};

pub const CartesianVector = CartesianPosition;

pub const CylindricalVector = struct {
    x: f32,
    r: f32,
    theta: f32,
};

pub const CartesianPositions = std.MultiArrayList(CartesianPosition);
pub const CylindricalCoordinates = std.MultiArrayList(CylindricalCoordinate);
pub const Quaternions = std.MultiArrayList(Quaternion);
pub const CartesianVectors = std.MultiArrayList(CartesianVector);
pub const CylindricalVectors = std.MultiArrayList(CylindricalVector);
