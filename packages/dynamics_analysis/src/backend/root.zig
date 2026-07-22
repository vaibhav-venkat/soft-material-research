//! Numerical backend declarations.

pub const linalg = @import("linalg");
pub const Context = linalg.Context;
pub const laggedPearson = @import("linalg_correlation.zig").laggedPearson;
