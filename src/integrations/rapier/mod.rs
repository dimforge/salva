//! Two-way coupling with the Rapier physics engine.

pub use fluids_pipeline::{
    ColliderCouplingManager, ColliderCouplingSet, ColliderSampling, FluidsPipeline,
};

pub use harness_plugin::FluidsHarnessPlugin;

mod fluids_pipeline;
mod harness_plugin;

#[cfg(feature = "rapier-testbed")]
mod testbed_plugin;
#[cfg(feature = "rapier-testbed")]
pub use testbed_plugin::{FluidsRenderingMode, FluidsTestbedPlugin};
