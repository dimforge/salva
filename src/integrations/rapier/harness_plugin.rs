use super::FluidsPipeline;
use rapier_testbed::harness::RunState;
use rapier_testbed::physics::PhysicsEvents;
use rapier_testbed::{HarnessPlugin, PhysicsState};

/// A user-defined callback executed at each frame.
pub type FluidCallback =
    Box<dyn FnMut(&mut PhysicsState, &PhysicsEvents, &mut FluidsPipeline, &RunState)>;

/// A plugin for rendering fluids with the Rapier harness.
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

    /// Adds a callback to be executed at each frame.
    pub fn add_callback(
        &mut self,
        f: impl FnMut(&mut PhysicsState, &PhysicsEvents, &mut FluidsPipeline, &RunState) + 'static,
    ) {
        self.callbacks.push(Box::new(f))
    }

    /// Sets the fluids pipeline used by the harness.
    pub fn set_pipeline(&mut self, fluids_pipeline: FluidsPipeline) {
        self.fluids_pipeline = fluids_pipeline;
        self.fluids_pipeline.liquid_world.counters.enable();
    }
}

impl HarnessPlugin for FluidsHarnessPlugin {
    fn run_callbacks(
        &mut self,
        physics: &mut PhysicsState,
        physics_events: &PhysicsEvents,
        run_state: &RunState,
    ) {
        for f in &mut self.callbacks {
            f(
                physics,
                physics_events,
                &mut self.fluids_pipeline,
                run_state,
            )
        }
    }

    fn step(&mut self, physics: &mut PhysicsState, _run_state: &RunState) {
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
