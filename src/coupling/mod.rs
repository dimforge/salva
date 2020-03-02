//! Two-way coupling between nphysics bodies and salva fluids.

#[cfg(feature = "nphysics")]
pub use self::collider_coupling_manager::{
    ColliderCouplingManager, ColliderCouplingSet, CouplingMethod,
};
pub use self::coupling_manager::CouplingManager;

#[cfg(feature = "nphysics")]
mod collider_coupling_manager;
mod coupling_manager;
