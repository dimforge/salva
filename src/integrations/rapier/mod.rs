//! Two-way coupling with the Rapier physics engine.

pub use fluids_pipeline::{
    ColliderCouplingManager, ColliderCouplingSet, ColliderSampling, FluidsPipeline,
};
#[cfg(feature = "rapier-testbed")]
pub use testbed_plugin::{FluidsRenderingMode, FluidsTestbedPlugin};

mod fluids_pipeline;
#[cfg(feature = "rapier-testbed")]
mod testbed_plugin;
#[cfg(feature = "rapier-harness")]
mod harness_plugin;