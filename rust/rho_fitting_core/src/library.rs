use crate::{CoreError, CoreResult};

pub fn validate_known_terms(names: &[String], known: &[&str]) -> CoreResult<()> {
    for name in names {
        if !known.iter().any(|known_name| known_name == name) {
            return Err(CoreError::InvalidInput(format!("unknown term {name}")));
        }
    }
    Ok(())
}
