//! Integration of Salva with other physics engines.

#[cfg(feature = "nphysics")]
pub mod nphysics;
#[cfg(feature = "rapier")]
pub mod rapier;
