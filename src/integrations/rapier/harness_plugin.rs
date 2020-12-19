use super::FluidsPipeline;
use crate::object::{Boundary, BoundaryHandle, Fluid, FluidHandle};
use na::{Point3, Vector3};
use rapier::math::{Point, Vector};
use rapier_testbed::{PhysicsState, HarnessPlugin};
use std::collections::HashMap;

/// A user-defined callback executed at each frame.
pub type FluidCallback = Box<dyn FnMut(&mut PhysicsState, &mut FluidsPipeline, f32)>;

/// A plugin for rendering fluids with the Rapier testbed.
pub struct FluidsHarnessPlugin {
    callbacks: Vec<FluidCallback>,
    step_time: f64,
    fluids_pipeline: FluidsPipeline,
}

impl FluidsHarnessPlugin {
    /// Initializes the plugin.
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
            step_time: 0.0,
            fluids_pipeline: FluidsPipeline::new(0.025, 2.0),
        }
    }

    /// Initialize the plugin with external FluidsPipeline
    /// This allows us to set the particle_radius, and smoothing factor
    // pub fn new_with_pipeline(fluids_pipeline: FluidsPipeline) -> Self {
    //     Self {
    //         callbacks: Vec::new(),
    //         step_time: 0.0,
    //         fluids_pipeline
    //     }
    // }

    /// Adds a callback to be executed at each frame.
    pub fn add_callback(
        &mut self,
        f: impl FnMut(&mut PhysicsState, &mut FluidsPipeline, f32) + 'static,
    ) {
        self.callbacks.push(Box::new(f))
    }

    /// Sets the fluids pipeline used by the testbed.
    pub fn set_pipeline(&mut self, fluids_pipeline: FluidsPipeline) {
        self.fluids_pipeline = fluids_pipeline;
        self.fluids_pipeline.liquid_world.counters.enable();
    }

}

impl HarnessPlugin for FluidsHarnessPlugin {
    fn run_callbacks(&mut self, physics: &mut PhysicsState, t: f32) {
        for f in &mut self.callbacks {
            f(physics, &mut self.fluids_pipeline, t)
        }
    }

    fn step(&mut self, physics: &mut PhysicsState) {
        let step_time = instant::now();
        let dt = physics.integration_parameters.dt();
        self.fluids_pipeline.step(
            &physics.gravity,
            dt,
            &physics.colliders,
            &mut physics.bodies,
        );

        self.step_time = instant::now() - step_time;
    }

    fn profiling_string(&self) -> String {
        format!("Fluids: {:.2}ms", self.step_time)
    }
}
