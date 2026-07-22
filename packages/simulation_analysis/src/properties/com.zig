//! Center-of-mass property interfaces.
//!
//! Implementations write one cylindrical SoA sample per input frame. They do
//! not allocate; callers own and size all input and output buffers.

const std = @import("std");
const coordinates_module = @import("../coordinates.zig");

/// One frame-level cylindrical vector.
pub const CylindricalFrameValue = struct {
    x: f32,
    theta: f32,
    r: f32,
};

/// SoA storage for frame-level cylindrical values.
pub const CylindricalFrameValues = std.MultiArrayList(CylindricalFrameValue);

pub const Error = error{
    NotImplemented,
    BufferSizeMismatch,
    InvalidParticleCount,
};

/// Calculate one unwrapped cylindrical center of mass per frame.
///
/// `coordinates` is ordered by frame and then particle within each SoA field.
/// `destination.len` must equal `coordinates.len / particle_count`.
pub fn com_unwrapped(
    coordinates: coordinates_module.CylindricalCoordinates.Slice,
    particle_count: usize,
    destination: CylindricalFrameValues.Slice,
) Error!void {
    _ = coordinates;
    _ = particle_count;
    _ = destination;
    return error.NotImplemented;
}

/// Calculate one unwrapped cylindrical center-of-mass velocity per frame.
///
/// `steps` contains one simulation step per frame. `coordinates` uses the same
/// frame-major logical ordering as `com_unwrapped`; the result is SoA.
pub fn com_velocity_unwrapped(
    coordinates: coordinates_module.CylindricalCoordinates.Slice,
    steps: []const u64,
    particle_count: usize,
    destination: CylindricalFrameValues.Slice,
) Error!void {
    _ = coordinates;
    _ = steps;
    _ = particle_count;
    _ = destination;
    return error.NotImplemented;
}
