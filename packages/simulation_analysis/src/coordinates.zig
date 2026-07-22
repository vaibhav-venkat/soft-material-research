//! Shared SoA coordinate storage types.

const std = @import("std");

/// One Cartesian particle position.
pub const CartesianPosition = struct {
    x: f32,
    y: f32,
    z: f32,
};

/// One cylindrical particle position around the x axis.
pub const CylindricalCoordinate = struct {
    x: f32,
    theta: f32,
    r: f32,
};

pub const CartesianPositions = std.MultiArrayList(CartesianPosition);
pub const CylindricalCoordinates = std.MultiArrayList(CylindricalCoordinate);
