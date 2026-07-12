mod domain;

pub(crate) use domain::{
    build_targets, relative_mass_error, CurrentQField, CylindricalGrid, MechanicalFieldSet,
    MechanicalFrame, MechanicalInputViews, PhysicalComponent, TENSOR_COMPONENTS,
};
