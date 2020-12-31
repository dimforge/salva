use super::FluidsPipeline;
use crate::object::{Boundary, BoundaryHandle, Fluid, FluidHandle};
use kiss3d::window::Window;
use na::{Point3, Vector3};
use rapier::math::{Point, Vector};
use rapier_testbed::harness::RunState;
use rapier_testbed::objects::node::GraphicsNode;
use rapier_testbed::{PhysicsState, TestbedPlugin};
use std::collections::HashMap;

/// How the fluids should be rendered by the testbed.
#[derive(Copy, Clone, Debug)]
pub enum FluidsRenderingMode {
    /// Use a red taint the closer to `max` the velocity is.
    VelocityColor {
        /// Fluids with a velocity smaller than this will not have any red taint.
        min: f32,
        /// Fluids with a velocity greater than this will be completely red.
        max: f32,
    },
    /// Use a plain color.
    StaticColor,
}

/// A user-defined callback executed at each frame.
pub type FluidCallback =
    Box<dyn FnMut(&mut Window, &mut PhysicsState, &mut FluidsPipeline, &RunState)>;

/// A plugin for rendering fluids with the Rapier testbed.
pub struct FluidsTestbedPlugin {
    callbacks: Vec<FluidCallback>,
    step_time: f64,
    fluids_pipeline: FluidsPipeline,
    f2sn: HashMap<FluidHandle, FluidNode>,
    boundary2sn: HashMap<BoundaryHandle, FluidNode>,
    f2color: HashMap<FluidHandle, Point3<f32>>,
    fluid_rendering_mode: FluidsRenderingMode,
    render_boundary_particles: bool,
    ground_color: Point3<f32>,
}

impl FluidsTestbedPlugin {
    /// Initializes the plugin.
    pub fn new() -> Self {
        Self {
            step_time: 0.0,
            callbacks: Vec::new(),
            fluids_pipeline: FluidsPipeline::new(0.025, 2.0),
            f2sn: HashMap::new(),
            boundary2sn: HashMap::new(),
            f2color: HashMap::new(),
            fluid_rendering_mode: FluidsRenderingMode::StaticColor,
            render_boundary_particles: false,
            ground_color: Point3::new(0.5, 0.5, 0.5),
        }
    }

    /// Adds a callback to be executed at each frame.
    pub fn add_callback(
        &mut self,
        f: impl FnMut(&mut Window, &mut PhysicsState, &mut FluidsPipeline, &RunState) + 'static,
    ) {
        self.callbacks.push(Box::new(f))
    }

    /// Sets the fluids pipeline used by the testbed.
    pub fn set_pipeline(&mut self, fluids_pipeline: FluidsPipeline) {
        self.fluids_pipeline = fluids_pipeline;
        self.fluids_pipeline.liquid_world.counters.enable();
    }

    /// Sets the color used to render the specified fluid.
    pub fn set_fluid_color(&mut self, fluid: FluidHandle, color: Point3<f32>) {
        let _ = self.f2color.insert(fluid, color);

        if let Some(n) = self.f2sn.get_mut(&fluid) {
            n.set_color(color)
        }
    }

    /// Sets the way fluids are rendered.
    pub fn set_fluid_rendering_mode(&mut self, mode: FluidsRenderingMode) {
        self.fluid_rendering_mode = mode;
    }

    /// Enables the rendering of boundary particles.
    pub fn enable_boundary_particles_rendering(&mut self, enabled: bool) {
        self.render_boundary_particles = enabled;

        for sn in self.boundary2sn.values_mut() {
            sn.scene_node_mut().set_visible(enabled);
        }
    }
}

impl TestbedPlugin for FluidsTestbedPlugin {
    fn init_graphics(&mut self, window: &mut Window, gen_color: &mut dyn FnMut() -> Point3<f32>) {
        let particle_radius = self.fluids_pipeline.liquid_world.particle_radius();

        for (handle, fluid) in self.fluids_pipeline.liquid_world.fluids().iter() {
            let color = *self.f2color.entry(handle).or_insert_with(|| gen_color());
            let node = FluidNode::new(particle_radius, &fluid.positions, color, window);
            let _ = self.f2sn.insert(handle, node);
        }

        for (handle, boundary) in self.fluids_pipeline.liquid_world.boundaries().iter() {
            let color = self.ground_color;
            let node = FluidNode::new(particle_radius, &boundary.positions, color, window);
            let _ = self.boundary2sn.insert(handle, node);
        }
    }

    fn clear_graphics(&mut self, window: &mut Window) {
        for sn in self.f2sn.values_mut().chain(self.boundary2sn.values_mut()) {
            let node = sn.scene_node_mut();
            #[cfg(feature = "dim2")]
            window.remove_planar_node(node);
            #[cfg(feature = "dim3")]
            window.remove_node(node);
        }

        self.f2sn.clear();
        self.boundary2sn.clear();
    }

    fn run_callbacks(
        &mut self,
        window: &mut Window,
        physics: &mut PhysicsState,
        run_state: &RunState,
    ) {
        for f in &mut self.callbacks {
            f(window, physics, &mut self.fluids_pipeline, run_state)
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

    fn draw(&mut self) {
        for (i, fluid) in self.fluids_pipeline.liquid_world.fluids().iter() {
            if let Some(node) = self.f2sn.get_mut(&i) {
                node.update_with_fluid(fluid, self.fluid_rendering_mode)
            }
        }

        if self.render_boundary_particles {
            for (i, boundary) in self.fluids_pipeline.liquid_world.boundaries().iter() {
                if let Some(node) = self.boundary2sn.get_mut(&i) {
                    node.update_with_boundary(boundary)
                }
            }
        }
    }

    fn profiling_string(&self) -> String {
        format!("Fluids: {:.2}ms", self.step_time)
    }
}

struct FluidNode {
    radius: f32,
    color: Point3<f32>,
    base_color: Point3<f32>,
    gfx: GraphicsNode,
    balls_gfx: Vec<GraphicsNode>,
}

impl FluidNode {
    pub fn new(
        radius: f32,
        centers: &[Point<f32>],
        color: Point3<f32>,
        window: &mut Window,
    ) -> FluidNode {
        #[cfg(feature = "dim2")]
        let mut gfx = window.add_planar_group();
        #[cfg(feature = "dim3")]
        let mut gfx = window.add_group();

        let mut balls_gfx = Vec::new();

        for c in centers {
            #[cfg(feature = "dim2")]
            let mut ball_gfx = gfx.add_circle(radius);
            #[cfg(feature = "dim3")]
            let mut ball_gfx = gfx.add_sphere(radius);
            let c: Vector<f64> = na::convert_unchecked(c.coords);
            let c: Vector<f32> = na::convert(c);
            ball_gfx.set_local_translation(c.into());
            balls_gfx.push(ball_gfx);
        }

        let mut res = FluidNode {
            radius,
            color,
            base_color: color,
            gfx,
            balls_gfx,
        };

        res.gfx.set_color(color.x, color.y, color.z);
        res
    }

    pub fn set_color(&mut self, color: Point3<f32>) {
        self.gfx.set_color(color.x, color.y, color.z);
        self.color = color;
        self.base_color = color;
    }

    fn update(
        &mut self,
        centers: &[Point<f32>],
        velocities: &[Vector<f32>],
        mode: FluidsRenderingMode,
    ) {
        if centers.len() > self.balls_gfx.len() {
            for _ in 0..centers.len() - self.balls_gfx.len() {
                #[cfg(feature = "dim2")]
                let ball_gfx = self.gfx.add_circle(self.radius);
                #[cfg(feature = "dim3")]
                let ball_gfx = self.gfx.add_sphere(self.radius);
                self.balls_gfx.push(ball_gfx);
            }
        }

        for ball_gfx in &mut self.balls_gfx[centers.len()..] {
            ball_gfx.set_visible(false);
        }

        for (i, (pt, ball)) in centers.iter().zip(self.balls_gfx.iter_mut()).enumerate() {
            ball.set_visible(true);
            let c: Vector<f64> = na::convert_unchecked(pt.coords);
            let c: Vector<f32> = na::convert(c);
            ball.set_local_translation(c.into());

            let color = match mode {
                FluidsRenderingMode::StaticColor => self.base_color,
                FluidsRenderingMode::VelocityColor { min, max } => {
                    let start = self.base_color.coords;
                    let end = Vector3::new(1.0, 0.0, 0.0);
                    let vel: Vector<f64> = na::convert_unchecked(velocities[i]);
                    let vel: Vector<f32> = na::convert(vel);
                    let t = (vel.norm() - min) / (max - min);
                    start.lerp(&end, na::clamp(t, 0.0, 1.0)).into()
                }
            };

            ball.set_color(color.x, color.y, color.z);
        }
    }

    pub fn update_with_boundary(&mut self, boundary: &Boundary) {
        self.update(&boundary.positions, &[], FluidsRenderingMode::StaticColor)
    }

    pub fn update_with_fluid(&mut self, fluid: &Fluid, rendering_mode: FluidsRenderingMode) {
        self.update(&fluid.positions, &fluid.velocities, rendering_mode)
    }

    pub fn scene_node_mut(&mut self) -> &mut GraphicsNode {
        &mut self.gfx
    }
}
