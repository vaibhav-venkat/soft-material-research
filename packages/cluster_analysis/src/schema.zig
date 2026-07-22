//! Central validation contracts for structural cluster analysis.

const std = @import("std");
const safetensors = @import("safetensors");
const Options = @import("options.zig").Options;

pub const Geometry = struct {
    axial_period: f64,
    circumference: f64,

    pub fn surfaceArea(self: Geometry) f64 {
        return self.axial_period * self.circumference;
    }
};

pub const FrameSchema = struct {
    frame_count: usize,
    particle_count: usize,
};

/// Future cluster kernels consume projected `(x, radius * theta)` points.
pub const StructuralFrame = struct {
    points: []const [2]f64,
    psi6: []const [2]f64,
    eligible: []const bool,
    periods: [2]f64,
};

pub fn validateOptions(value: Options) !void {
    if (value.frame_stop) |stop| {
        if (stop <= value.frame_start) return error.InvalidFrameRange;
    }
    if (!std.math.isFinite(value.psi6_minimum) or
        value.psi6_minimum < 0.0 or value.psi6_minimum > 1.0)
    {
        return error.InvalidPsi6Threshold;
    }
    if (!std.math.isFinite(value.misorientation_degrees) or
        value.misorientation_degrees < 0.0 or value.misorientation_degrees > 30.0)
    {
        return error.InvalidMisorientation;
    }
    if (!std.math.isFinite(value.neighbor_radius_diameters) or
        value.neighbor_radius_diameters <= 0.0)
    {
        return error.InvalidNeighborRadius;
    }
    if (value.minimum_particles < 2) return error.InvalidMinimumParticles;
    if (!std.math.isFinite(value.particle_diameter) or value.particle_diameter <= 0.0) {
        return error.InvalidParticleDiameter;
    }
}

pub fn readGeometry(reader: *const safetensors.Reader) !Geometry {
    const geometry = Geometry{
        .axial_period = try readPositiveScalar(try reader.tensor("lx")),
        .circumference = try readPositiveScalar(try reader.tensor("circumference")),
    };
    if (!std.math.isFinite(geometry.surfaceArea())) return error.InvalidSurfaceArea;
    return geometry;
}

pub fn inspectFrameSchema(reader: *const safetensors.Reader) !FrameSchema {
    const coordinates = try reader.tensor("coords");
    const psi_real = try reader.tensor("psi_real");
    const psi_imaginary = try reader.tensor("psi_imag");
    const eligible = try reader.tensor("hexatic_shell_mask");
    const steps = try reader.tensor("step");
    if (coordinates.shape.len != 3 or coordinates.shape[0] == 0 or
        coordinates.shape[1] == 0 or coordinates.shape[2] != 3)
    {
        return error.InvalidCoordinateShape;
    }
    const frames = coordinates.shape[0];
    const particles = coordinates.shape[1];
    try requireFrameParticleShape(psi_real, frames, particles);
    try requireFrameParticleShape(psi_imaginary, frames, particles);
    try requireFrameParticleShape(eligible, frames, particles);
    if (steps.shape.len != 1 or steps.shape[0] != frames) return error.InvalidStepShape;
    if (!isFloat(coordinates.dtype) or !isFloat(psi_real.dtype) or
        !isFloat(psi_imaginary.dtype)) return error.InvalidFloatDtype;
    if (eligible.dtype != .bool) return error.InvalidMaskDtype;
    if (steps.dtype != .i32 and steps.dtype != .i64) return error.InvalidStepDtype;
    return .{ .frame_count = frames, .particle_count = particles };
}

pub fn requireCompatibleFrames(reference: FrameSchema, candidate: FrameSchema) !void {
    if (candidate.particle_count != reference.particle_count) {
        return error.InconsistentParticleCount;
    }
}

pub fn validateStructuralFrame(frame: StructuralFrame) !void {
    if (frame.points.len == 0) return error.NoParticles;
    if (frame.psi6.len != frame.points.len or frame.eligible.len != frame.points.len) {
        return error.DimensionMismatch;
    }
    for (frame.periods) |period| {
        if (!std.math.isFinite(period) or period <= 0.0) return error.InvalidPeriod;
    }
    for (frame.points) |point| {
        for (point) |value| if (!std.math.isFinite(value)) return error.NonFiniteCoordinate;
    }
    for (frame.psi6) |value| {
        for (value) |component| if (!std.math.isFinite(component)) return error.NonFinitePsi6;
    }
}

fn requireFrameParticleShape(tensor: safetensors.TensorView, frames: usize, particles: usize) !void {
    if (tensor.shape.len != 2 or tensor.shape[0] != frames or tensor.shape[1] != particles) {
        return error.InvalidFrameParticleShape;
    }
}

fn readPositiveScalar(tensor: safetensors.TensorView) !f64 {
    if (tensor.elementCount() != 1) return error.InvalidScalarShape;
    const value = switch (tensor.dtype) {
        .f32 => @as(f64, @floatCast((try tensor.values(f32))[0])),
        .f64 => (try tensor.values(f64))[0],
        else => return error.InvalidFloatDtype,
    };
    if (!std.math.isFinite(value) or value <= 0.0) return error.InvalidGeometry;
    return value;
}

fn isFloat(dtype: safetensors.Dtype) bool {
    return dtype == .f32 or dtype == .f64;
}
