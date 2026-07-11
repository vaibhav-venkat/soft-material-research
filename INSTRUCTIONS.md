# 1. Consolidate the existing Rust core
2. 
There are two substantial implementations of mechanical deposition:
CPU loops in [`mechanics/mod.rs`](/Users/vaibhavvenkat/structures_and_functions/rust/rho_fitting_core/src/mechanics/mod.rs)

GPU/Burn deposition in [`coarse_grain_burn/mod.rs`](/Users/vaibhavvenkat/structures_and_functions/rust/rho_fitting_core/src/coarse_grain_burn/mod.rs)


Keep a single typed MechanicalFieldSet domain model and share validation, moment construction, conservation checks, and output conversion. 

The CPU version should be a reference/fallback backend, not a second independently evolving implementation. It should also not be preferred.

Also replace dynamic-rank ArrayD for J_Q and targets internally with fixed-rank types. ArrayD is useful at the PyO3 boundary, but it forfeits shape guarantees inside Rust.

Useful domain types:
```rust
struct CylindricalGrid {
    x: Array1<f64>,
    theta: Array1<f64>,
    r: Array1<f64>,
    lx: f64,
    theta_period: f64,
}

struct MechanicalFields {
    rho: Array4<f64>,
    p: Array5<f64>,
    q: Array6<f64>,
    a: Array6<f64>,
    // fixed-rank current tensors
}

enum PhysicalComponent {
    Axial,
    Azimuthal,
    Radial,
}
```

And also other types when necessary. At the end of this, there shouldn't be two separate/divergent paths like mechanics/mod.rs and coarse-grain-burn/mod.rs which require changing both when making a simple edit. Minimize the leakage of any errors by consolidating these types of things.

Most importantly, encode the physical order (x, e_theta, e_r) centrally to avoid any mismatch within the conventions between Python and Rust. Do this also for other frame types. 

The idea is to emphasize as much types as possible while keeping and improving readability.
