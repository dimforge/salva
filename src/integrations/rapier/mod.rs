pub use fluids_pipeline::{
    ColliderCouplingManager, ColliderCouplingSet, ColliderSampling, FluidsPipeline,
};
#[cfg(feature = "rapier-testbed")]
pub use testbed_plugin::{FluidsRenderingMode, FluidsTestbedPlugin};

mod fluids_pipeline;
#[cfg(feature = "rapier-testbed")]
mod testbed_plugin;
