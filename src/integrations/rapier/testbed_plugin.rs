use super::FluidsPipeline;
use crate::math::{Isometry, Point, Rotation, Translation, Vector};
use crate::object::{BoundaryHandle, FluidHandle};
#[cfg(feature = "dim3")]
use bevy::math::Quat;
use bevy::prelude::{Assets, Commands, Mesh, Query, StandardMaterial, Transform};
use bevy_egui::egui::ComboBox;
use bevy_egui::{egui::Window, EguiContext};
#[cfg(feature = "dim3")]
use na::Quaternion;
use na::{Point3, Vector3};
use parry::shape::SharedShape;
use rapier_testbed::{
    harness::Harness, objects::node::EntityWithGraphics, GraphicsManager, PhysicsState,
    TestbedPlugin,
};
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
    pub rendering_mode: FluidsRenderingMode,
    callbacks: Vec<FluidCallback>,
    step_time: f64,
    fluids_pipeline: FluidsPipeline,
    f2sn: HashMap<FluidHandle, Vec<EntityWithGraphics>>,
    boundary2sn: HashMap<BoundaryHandle, Vec<EntityWithGraphics>>,
    f2color: HashMap<FluidHandle, Point3<f32>>,
    ground_color: Point3<f32>,
}

impl FluidsTestbedPlugin {
    /// Initializes the plugin.
    pub fn new() -> Self {
        Self {
            render_boundary_particles: false,
            rendering_mode: FluidsRenderingMode::StaticColor,
            step_time: 0.0,
            callbacks: Vec::new(),
            fluids_pipeline: FluidsPipeline::new(0.025, 2.0),
            f2sn: HashMap::new(),
            boundary2sn: HashMap::new(),
            f2color: HashMap::new(),
            ground_color: Point3::new(0.5, 0.5, 0.5),
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

        // if let Some(n) = self.f2sn.get_mut(&fluid) {
        //     // n.set_color(color)
        // }
    }

    /// Sets the way fluids are rendered.
    pub fn set_fluid_rendering_mode(&mut self, mode: FluidsRenderingMode) {
        self.rendering_mode = mode;
    }

    /// Enables the rendering of boundary particles.
    pub fn enable_boundary_particles_rendering(&mut self, enabled: bool) {
        self.render_boundary_particles = enabled;
    }

    // TODO: pass velocity & acceleration vectors in
    fn add_particle_graphics(
        &self,
        particle: &Point<f32>,
        particle_radius: f32,
        graphics: &mut GraphicsManager,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<StandardMaterial>,
        _components: &mut Query<(&mut Transform,)>,
        _harness: &mut Harness,
        color: &Point3<f32>,
    ) -> Vec<EntityWithGraphics> {
        // println!("rm: {:?}", self.fluid_rendering_mode);
        let shape = match self.rendering_mode {
            #[cfg(feature = "dim3")]
            FluidsRenderingMode::VelocityArrows { .. } => {
                SharedShape::cone(particle_radius, particle_radius / 4.)
            }
            #[cfg(feature = "dim2")]
            //FIXME: use actual trig
            FluidsRenderingMode::VelocityArrows { .. } => SharedShape::triangle(
                Point::new(0., particle_radius),
                Point::new(particle_radius * 0.4, -particle_radius * 0.8),
                Point::new(-particle_radius * 0.4, -particle_radius * 0.8),
            ),

            _ => SharedShape::ball(particle_radius),
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
}

impl TestbedPlugin for FluidsTestbedPlugin {
    fn init_graphics(
        &mut self,
        graphics: &mut GraphicsManager,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<StandardMaterial>,
        components: &mut Query<(&mut Transform,)>,
        harness: &mut Harness,
        gen_color: &mut dyn FnMut() -> Point3<f32>,
    ) {
        //hack to get _some_ particle radius
        let mut particle_radius = None;
        for (handle, fluid) in self.fluids_pipeline.liquid_world.fluids().iter() {
            let _ = self
                .f2sn
                .insert(handle, Vec::with_capacity(fluid.positions.len()));

            let color = *self.f2color.entry(handle).or_insert_with(|| gen_color());

            particle_radius = Some(fluid.particle_radius());
            for particle in &fluid.positions {
                let ent = self.add_particle_graphics(
                    particle,
                    particle_radius.unwrap(),
                    graphics,
                    commands,
                    meshes,
                    materials,
                    components,
                    harness,
                    &color,
                );
                if let Some(entities) = self.f2sn.get_mut(&handle) {
                    entities.extend(ent);
                }
            }
        }

        if self.render_boundary_particles {
            // hack, ugly hack
            if let Some(particle_radius) = particle_radius {
                for (handle, _) in self.fluids_pipeline.liquid_world.boundaries().iter() {
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
                                        );
                                        if let Some(entities) = self.boundary2sn.get_mut(&handle) {
                                            entities.extend(ent);
                                        }
                                    }
                                },
                                crate::integrations::rapier::ColliderSampling::DynamicContactSampling => {
                                    // TODO
                                },
                            }
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
        self.f2sn.clear();
        self.boundary2sn.clear();
    }

    fn run_callbacks(&mut self, harness: &mut Harness) {
        for f in &mut self.callbacks {
            f(harness, &mut self.fluids_pipeline)
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
        _graphics: &mut GraphicsManager,
        _commands: &mut Commands,
        _meshes: &mut Assets<Mesh>,
        _materials: &mut Assets<StandardMaterial>,
        components: &mut Query<(&mut Transform,)>,
        harness: &mut Harness,
    ) {
        fn lerp_velocity(
            velocity: Vector<f32>,
            start: Vector3<f32>,
            min: f32,
            max: f32,
        ) -> Vector3<f32> {
            let end = Vector3::new(1.0, 0.0, 0.0);
            let vel: Vector<f32> = na::convert_unchecked(velocity);
            let vel: Vector<f32> = na::convert(vel);
            let t = (vel.norm() - min) / (max - min);
            start.lerp(&end, na::clamp(t, 0.0, 1.0))
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
                            components.get_component_mut::<Transform>(entity.entity)
                        {
                            {
                                pos.translation.x = particle.x;
                                pos.translation.y = particle.y;
                                #[cfg(feature = "dim3")]
                                {
                                    // FIXME: this is not working, converting from Vector3 -> Quaternion -> bevy::Quat, perhaps it's the handedness?
                                    let rotation = Quaternion::from_vector(
                                        (velocity * -1.).normalize().to_homogeneous(),
                                    );
                                    pos.translation.z = particle.z;
                                    pos.rotation = Quat::from_xyzw(
                                        rotation.i, rotation.j, rotation.k, rotation.w,
                                    );
                                }
                                #[cfg(feature = "dim2")]
                                {
                                    // FIXME: ???
                                    // pos.rotation = Quat::from_rotation_z(co_pos.rotation.angle());
                                }
                            }
                        }

                        if let Some(color) = self.f2color.get(&handle) {
                            match self.rendering_mode {
                                FluidsRenderingMode::VelocityColor { min, max } => {
                                    let lerp = lerp_velocity(
                                        fluid.velocities[idx],
                                        color.coords,
                                        min,
                                        max,
                                    );
                                    entity
                                        .set_color(_materials, Point3::new(lerp.x, lerp.y, lerp.z));
                                }
                                FluidsRenderingMode::VelocityArrows { min, max } => {
                                    let lerp = lerp_velocity(
                                        fluid.velocities[idx],
                                        color.coords,
                                        min,
                                        max,
                                    );
                                    entity
                                        .set_color(_materials, Point3::new(lerp.x, lerp.y, lerp.z));
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
                                    entity.set_color(_materials, *color);
                                }
                            }
                        }
                        entity.update(&harness.physics.colliders, components)
                    }
                }
            }
        }
    }

    fn update_ui(
        &mut self,
        ui_context: &EguiContext,
        harness: &mut Harness,
        graphics: &mut GraphicsManager,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<StandardMaterial>,
        components: &mut Query<(&mut Transform,)>,
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
                        FLUIDS_RENDERING_MAP[get_rendering_mode_index(self.rendering_mode)].0,
                    )
                    .show_ui(ui, |ui| {
                        for (_, (name, mode)) in FLUIDS_RENDERING_MAP.iter().enumerate() {
                            changed = ui
                                .selectable_value(&mut self.rendering_mode, *mode, name)
                                .changed()
                                || changed;
                        }
                    });
                println!("changedL {}, {:?}", changed, self.rendering_mode);

                if changed {
                    println!("{:?}", self.rendering_mode);
                    // FIXME: not too sure what to do here for color
                    let fluid_handle = self
                        .fluids_pipeline
                        .liquid_world
                        .fluids()
                        .iter()
                        .next()
                        .unwrap()
                        .0;
                    self.clear_graphics(graphics, commands);
                    let color = self.f2color[&fluid_handle].clone();
                    self.init_graphics(
                        graphics,
                        commands,
                        meshes,
                        materials,
                        components,
                        harness,
                        &mut || color,
                    )
                }
            });
    }

    fn profiling_string(&self) -> String {
        format!("Fluids: {:.2}ms", self.step_time)
    }
}
