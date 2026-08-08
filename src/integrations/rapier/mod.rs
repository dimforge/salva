//! Two-way coupling with the Rapier physics engine.

pub use crate::solver::DfsphParameters;
pub use fluids_pipeline::{
    ColliderCouplingManager, ColliderCouplingSet, ColliderSampling, FluidsPipeline,
};

mod fluids_pipeline;

#[cfg(feature = "rapier-testbed")]
mod testbed_plugin;
#[cfg(feature = "rapier-testbed")]
pub use testbed_plugin::{FluidsRenderingMode, FluidsTestbedPlugin};
