use crate::math::{Isometry, Point, Real, Rotation, Translation, Vector};
use crate::object::{BoundaryHandle, FluidHandle};
use bevy::math::Quat;
use bevy::prelude::{Assets, Commands, Mesh, Query, Transform};
use bevy_egui::{egui::ComboBox, egui::Window, EguiContexts};
#[cfg(feature = "dim3")]
use na::Quaternion;
use na::{Point3, Vector3};
use parry::shape::SharedShape;
use rapier_testbed::{
    harness::Harness, objects::node::EntityWithGraphics, BevyMaterial, GraphicsManager,
    PhysicsState, TestbedPlugin,
};

use crate::integrations::rapier::FluidsPipeline;
use std::collections::HashMap;

//FIXME: handle this with macros, or use bevy-inspectable-egui
pub const FLUIDS_RENDERING_MAP: [(&str, FluidsRenderingMode); 3] = [
    ("Static", FluidsRenderingMode::StaticColor),
    (
        "Velocity Color",
        FluidsRenderingMode::VelocityColor {
            min: 0.0,
            max: 50.0,
        },
    ),
    // (
    //     "Velocity Color & Opacity",
    //     FluidsRenderingMode::VelocityColorOpacity {
    //         min: 0.0,
    //         max: 50.0,
    //     },
    // ),
    (
        "Velocity Arrows",
        FluidsRenderingMode::VelocityArrows {
            min: 0.0,
            max: 50.0,
        },
    ),
    // ("Acceleration Arrows", FluidsRenderingMode::AccelerationArrows),
];

/// How the fluids should be rendered by the testbed.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FluidsRenderingMode {
    /// Use a plain color.
    StaticColor,
    /// Use a red taint the closer to `max` the velocity is.
    VelocityColor {
        /// Fluids with a velocity smaller than this will not have any red taint.
        min: f32,
        /// Fluids with a velocity greater than this will be completely red.
        max: f32,
    },
    // /// Use a red taint the closer to `max` the velocity is, with opacity, low velocity is more transparent
    // VelocityColorOpacity {
    //     /// Fluids with a velocity smaller than this will not have any red taint.
    //     min: f32,
    //     /// Fluids with a velocity greater than this will be completely red.
    //     max: f32,
    // },
    /// Show particles as arrows indicating the velocity
    VelocityArrows {
        /// Fluids with a velocity smaller than this will not have any red taint.
        min: f32,
        /// Fluids with a velocity greater than this will be completely red.
        max: f32,
    },
}

/// A user-defined callback executed at each frame.
pub type FluidCallback = Box<dyn FnMut(&mut Harness, &mut FluidsPipeline)>;

/// A plugin for rendering fluids with the Rapier testbed.
pub struct FluidsTestbedPlugin {
    /// Whether to render the boundary particles
    pub render_boundary_particles: bool,
    /// Rendering mode of fluid particles
    pub fluids_rendering_mode: FluidsRenderingMode,
    callbacks: Vec<FluidCallback>,
    step_time: f64,
    fluids_pipeline: FluidsPipeline,
    f2sn: HashMap<FluidHandle, Vec<EntityWithGraphics>>,
    boundary2sn: HashMap<BoundaryHandle, Vec<EntityWithGraphics>>,
    f2color: HashMap<FluidHandle, Point3<f32>>,
    ground_color: Point3<f32>,
    default_fluid_color: Point3<f32>,
    queue_graphics_reset: bool,
}

impl FluidsTestbedPlugin {
    /// Initializes the plugin.
    pub fn new() -> Self {
        Self {
            render_boundary_particles: false,
            fluids_rendering_mode: FluidsRenderingMode::StaticColor,
            step_time: 0.0,
            callbacks: Vec::new(),
            fluids_pipeline: FluidsPipeline::new(0.025, 2.0),
            f2sn: HashMap::new(),
            boundary2sn: HashMap::new(),
            f2color: HashMap::new(),
            ground_color: Point3::new(0.5, 0.5, 0.5),
            default_fluid_color: Point3::new(0.0, 0.0, 0.5),
            queue_graphics_reset: false,
        }
    }

    /// Adds a callback to be executed at each frame.
    pub fn add_callback(&mut self, f: impl FnMut(&mut Harness, &mut FluidsPipeline) + 'static) {
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
    }

    /// Sets the way fluids are rendered.
    pub fn set_fluid_rendering_mode(&mut self, mode: FluidsRenderingMode) {
        self.fluids_rendering_mode = mode;
    }

    /// Enables the rendering of boundary particles.
    pub fn enable_boundary_particles_rendering(&mut self, enabled: bool) {
        self.render_boundary_particles = enabled;
    }

    // TODO: pass velocity & acceleration vectors in
    fn add_particle_graphics(
        &self,
        particle: &Point<Real>,
        particle_radius: f32,
        graphics: &mut GraphicsManager,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<BevyMaterial>,
        _components: &mut Query<&mut Transform>,
        _harness: &mut Harness,
        color: &Point3<f32>,
        force_shape: Option<SharedShape>,
    ) -> Vec<EntityWithGraphics> {
        let shape = if let Some(shape) = force_shape {
            shape
        } else {
            match self.fluids_rendering_mode {
                #[cfg(feature = "dim3")]
                FluidsRenderingMode::VelocityArrows { .. } => {
                    SharedShape::cone(particle_radius, particle_radius / 4.)
                }
                // #[cfg(feature = "dim2")]
                //FIXME: This doesn't work, it is caused by either not being in prefab_meshes, or the shape_type not being supported.. somewhere
                // FluidsRenderingMode::VelocityArrows { .. } => SharedShape::triangle(
                //     Point::new(0., particle_radius),
                //     Point::new(particle_radius * 0.4, -particle_radius * 0.8),
                //     Point::new(-particle_radius * 0.4, -particle_radius * 0.8),
                // ),
                _ => SharedShape::ball(particle_radius),
            }
        };

        let mut shapes = Vec::new();
        let isometry =
            Isometry::from_parts(Translation::from(particle.coords), Rotation::identity());
        graphics.add_shape(
            commands,
            meshes,
            materials,
            None,
            &*shape,
            false,
            &isometry,
            &Isometry::identity(),
            *color,
            &mut shapes,
        );
        shapes
    }

    fn lerp_velocity(
        velocity: Vector<f32>,
        start: Vector3<f32>,
        min: f32,
        max: f32,
    ) -> Vector3<f32> {
        let end = Vector3::new(1.0, 0.0, 0.0);
        let vel: Vector<Real> = na::convert_unchecked(velocity);
        let vel: Vector<Real> = na::convert(vel);
        let t = (vel.norm() - min) / (max - min);
        start.lerp(&end, na::clamp(t, 0.0, 1.0))
    }
}

impl TestbedPlugin for FluidsTestbedPlugin {
    fn init_plugin(&mut self) {
        // TODO: decide if anything needs to be changed
    }

    fn init_graphics(
        &mut self,
        graphics: &mut GraphicsManager,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<BevyMaterial>,
        components: &mut Query<&mut Transform>,
        harness: &mut Harness,
    ) {
        for (handle, fluid) in self.fluids_pipeline.liquid_world.fluids().iter() {
            let _ = self
                .f2sn
                .insert(handle, Vec::with_capacity(fluid.positions.len()));

            let color = *self
                .f2color
                .entry(handle)
                .or_insert_with(|| self.default_fluid_color);

            for particle in &fluid.positions {
                let ent = self.add_particle_graphics(
                    particle,
                    fluid.particle_radius(),
                    graphics,
                    commands,
                    meshes,
                    materials,
                    components,
                    harness,
                    &color,
                    None,
                );
                if let Some(entities) = self.f2sn.get_mut(&handle) {
                    entities.extend(ent);
                }
            }
        }

        let particle_radius = self.fluids_pipeline.liquid_world.particle_radius();

        // FIXME: There is currently no way to get the collider pose from this function
        if self.render_boundary_particles {
            for (handle, boundary) in self.fluids_pipeline.liquid_world.boundaries().iter() {
                let _ = self
                    .boundary2sn
                    .insert(handle, Vec::with_capacity(boundary.num_particles()));
                let color = self.ground_color;

                for (_, cce) in &self.fluids_pipeline.coupling.entries {
                    if cce.boundary == handle {
                        match &cce.sampling_method {
                            crate::integrations::rapier::ColliderSampling::StaticSampling(particles) => {
                                    for particle in particles {
                                        let ent = self.add_particle_graphics(
                                            particle,
                                            particle_radius,
                                            graphics,
                                            commands,
                                            meshes,
                                            materials,
                                            components,
                                            harness,
                                            &color,
                                            Some(SharedShape::ball(particle_radius))
                                        );
                                    if let Some(entities) = self.boundary2sn.get_mut(&handle) {
                                        entities.extend(ent);
                                    }
                                }
                            },
                            crate::integrations::rapier::ColliderSampling::DynamicContactSampling => {
                                // TODO: ???
                            },
                        }
                    }
                }
            }
        }
    }

    fn clear_graphics(&mut self, _graphics: &mut GraphicsManager, commands: &mut Commands) {
        for (handle, _) in self.fluids_pipeline.liquid_world.fluids().iter() {
            if let Some(entities) = self.f2sn.get_mut(&handle) {
                for entity in entities {
                    entity.despawn(commands);
                }
            }
        }

        for (handle, _) in self.fluids_pipeline.liquid_world.boundaries().iter() {
            if let Some(entities) = self.boundary2sn.get_mut(&handle) {
                for entity in entities {
                    entity.despawn(commands);
                }
            }
        }

        self.f2sn.clear();
        self.boundary2sn.clear();
    }

    fn run_callbacks(&mut self, harness: &mut Harness) {
        // FIXME: salva should be able to keep a list of indices that were added & removed in this step
        // at the moment we just clear & initialize the grahics when fluids_lengths changes
        let fluid_lengths: Vec<(FluidHandle, usize)> = self
            .fluids_pipeline
            .liquid_world
            .fluids()
            .iter()
            .map(|(h, f)| (h, f.positions.len()))
            .collect();

        for f in &mut self.callbacks {
            f(harness, &mut self.fluids_pipeline)
        }

        for (h, fl) in fluid_lengths {
            if let Some(fluid) = self.fluids_pipeline.liquid_world.fluids().get(h) {
                if fluid.positions.len() != fl || fluid.num_deleted_particles() > 0 {
                    self.queue_graphics_reset = true;
                }
            }
        }
    }

    fn step(&mut self, physics: &mut PhysicsState) {
        let step_time = instant::now();
        let dt = physics.integration_parameters.dt;
        self.fluids_pipeline.step(
            &physics.gravity,
            dt,
            &physics.colliders,
            &mut physics.bodies,
        );

        self.step_time = instant::now() - step_time;
    }

    fn draw(
        &mut self,
        graphics: &mut GraphicsManager,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<BevyMaterial>,
        components: &mut Query<&mut Transform>,
        harness: &mut Harness,
    ) {
        if self.queue_graphics_reset {
            self.clear_graphics(graphics, commands);
            self.init_graphics(graphics, commands, meshes, materials, components, harness);
            self.queue_graphics_reset = false;
        }

        let (mut min, mut max) = (f32::MAX, f32::MIN);
        for (handle, fluid) in self.fluids_pipeline.liquid_world.fluids().iter() {
            if let Some(entities) = self.f2sn.get_mut(&handle) {
                for (idx, particle) in fluid.positions.iter().enumerate() {
                    let velocity = Vector::from(fluid.velocities[idx]);
                    let magnitude = velocity.magnitude();
                    min = min.min(magnitude);
                    max = max.max(magnitude);
                    if let Some(entity) = entities.get_mut(idx) {
                        if let Ok(mut pos) =
                            components.get_mut(entity.entity)
                        {
                            {
                                pos.translation.x = particle.x;
                                pos.translation.y = particle.y;
                                #[cfg(feature = "dim3")]
                                {
                                    pos.translation.z = particle.z;

                                    if let FluidsRenderingMode::VelocityArrows { .. } =
                                        self.fluids_rendering_mode
                                    {
                                        let cone_paxis: Quaternion<Real> =
                                            Quaternion::from_vector(-Vector3::y().to_homogeneous());
                                        let vr = Quaternion::from_vector(
                                            velocity.normalize().to_homogeneous(),
                                        );
                                        let rotation = (vr - cone_paxis).normalize();

                                        pos.rotation = Quat::from_xyzw(
                                            rotation.i, rotation.j, rotation.k, rotation.w,
                                        );
                                    }
                                }
                                #[cfg(feature = "dim2")]
                                {
                                    let norm = velocity.normalize();
                                    let hyp = (norm.x * norm.x + norm.y * norm.y).sqrt();
                                    let angle = 2. * (norm.y / (norm.x + hyp)).atan();
                                    pos.rotation = Quat::from_rotation_z(angle);
                                }
                            }
                        }

                        if let Some(color) = self.f2color.get(&handle) {
                            match self.fluids_rendering_mode {
                                FluidsRenderingMode::VelocityColor { min, max } => {
                                    let lerp = Self::lerp_velocity(
                                        fluid.velocities[idx],
                                        color.coords,
                                        min,
                                        max,
                                    );
                                    entity
                                        .set_color(materials, Point3::new(lerp.x, lerp.y, lerp.z));
                                }
                                FluidsRenderingMode::VelocityArrows { min, max } => {
                                    let lerp = Self::lerp_velocity(
                                        fluid.velocities[idx],
                                        color.coords,
                                        min,
                                        max,
                                    );
                                    entity
                                        .set_color(materials, Point3::new(lerp.x, lerp.y, lerp.z));
                                }
                                // FIXME: rapier needs to be updated to respect opacity
                                // FluidsRenderingMode::VelocityColorOpacity { min, max } => {
                                //     let lerp = lerp_velocity(
                                //         fluid.velocities[idx],
                                //         color.coords,
                                //         min,
                                //         max,
                                //     );
                                //     entity.opacity = lerp.magnitude();
                                //     entity.set_color(
                                //         _materials,
                                //         Point3::new(lerp.x, lerp.y, lerp.z),
                                //     );
                                // }
                                // FluidsRenderingMode::VelocityArrows => {}
                                _ => {
                                    entity.opacity = 1.0;
                                    entity.set_color(materials, *color);
                                }
                            }
                        }
                        entity.update(&harness.physics.colliders, components, &graphics.gfx_shift)
                    }
                }
            }
        }
    }

    fn update_ui(
        &mut self,
        ui_context: &EguiContexts,
        harness: &mut Harness,
        graphics: &mut GraphicsManager,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<BevyMaterial>,
        components: &mut Query<&mut Transform>,
    ) {
        fn get_rendering_mode_index(rendering_mode: FluidsRenderingMode) -> usize {
            FLUIDS_RENDERING_MAP
                .iter()
                .enumerate()
                .find(|(_, mode)| rendering_mode == mode.1)
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }

        let _ = Window::new("Fluid Parameters")
            .min_height(200.0)
            .show(ui_context.ctx(), |ui| {
                let mut changed = false;

                let _ = ComboBox::from_label("Rendering Mode")
                    .width(150.0)
                    .selected_text(
                        FLUIDS_RENDERING_MAP[get_rendering_mode_index(self.fluids_rendering_mode)]
                            .0,
                    )
                    .show_ui(ui, |ui| {
                        for (_, (name, mode)) in FLUIDS_RENDERING_MAP.iter().enumerate() {
                            changed = ui
                                .selectable_value(&mut self.fluids_rendering_mode, *mode, *name)
                                .changed()
                                || changed;
                        }
                    });

                if changed {
                    // FIXME: not too sure what to do here for color
                    // let fluid_handle = self
                    //     .fluids_pipeline
                    //     .liquid_world
                    //     .fluids()
                    //     .iter()
                    //     .next()
                    //     .unwrap()
                    //     .0;
                    self.clear_graphics(graphics, commands);
                    // let color = self.f2color[&fluid_handle].clone();
                    self.init_graphics(graphics, commands, meshes, materials, components, harness)
                }
            });
    }

    fn profiling_string(&self) -> String {
        format!("Fluids: {:.2}ms", self.step_time)
    }
}
