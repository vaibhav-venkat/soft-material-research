//! Public analysis controls.

pub const Options = struct {
    frame_start: usize = 0,
    frame_stop: ?usize = null,
    timestep: f64 = 1.0,
    max_lag: ?usize = null,
    device_ordinal: u32 = 0,
};
