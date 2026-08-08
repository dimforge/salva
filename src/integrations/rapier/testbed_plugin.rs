use crate::math::{Real, Vector};
use crate::object::{BoundaryHandle, FluidHandle};
#[cfg(feature = "dim2")]
use kiss3d::prelude::{InstanceData2d as InstanceData, Mat2, SceneNode2d as SceneNode, Vec2};
#[cfg(feature = "dim3")]
use kiss3d::prelude::{InstanceData3d as InstanceData, Mat3, SceneNode3d as SceneNode, Vec3};
use kiss3d::{color::Color, window::Window};
use na::Vector3;
use rapier::pipeline::PhysicsWorld;
use rapier_testbed::{egui, settings::ExampleSettings, TestbedViewer};

use crate::integrations::rapier::{DfsphParameters, FluidsPipeline};
use std::collections::HashMap;

pub const FLUIDS_RENDERING_MAP: [(&str, FluidsRenderingMode); 3] = [
    ("Static", FluidsRenderingMode::StaticColor),
    (
        "Velocity Color",
        FluidsRenderingMode::VelocityColor {
            min: 0.0,
            max: 50.0,
        },
    ),
    (
        "Velocity Arrows",
        FluidsRenderingMode::VelocityArrows {
            min: 0.0,
            max: 50.0,
        },
    ),
];

const SETTINGS_RENDER_BOUNDARIES: &str = "Fluid render boundaries";
const SETTINGS_RENDERING_MODE: &str = "Fluid rendering mode";
const SETTINGS_MAX_PRESSURE_ITER: &str = "Fluid max pressure iter";
const SETTINGS_MAX_DIVERGENCE_ITER: &str = "Fluid max divergence iter";
const SETTINGS_MAX_DENSITY_ERROR: &str = "Fluid max density error";
const SETTINGS_MAX_DIVERGENCE_ERROR: &str = "Fluid max divergence error";
const SETTINGS_BOUNDARY_FORCE_COEFFICIENT: &str = "Fluid boundary force";

/// How the fluids should be rendered by the testbed.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FluidsRenderingMode {
    /// Use a plain color.
    StaticColor,
    /// Use a red taint the closer to `max` the velocity is.
    VelocityColor {
        /// Fluids with a velocity smaller than this will not have any red taint.
        min: Real,
        /// Fluids with a velocity greater than this will be completely red.
        max: Real,
    },
    /// Show particles as arrows indicating the velocity.
    VelocityArrows {
        /// Fluids with a velocity smaller than this will not have any red taint.
        min: Real,
        /// Fluids with a velocity greater than this will be completely red.
        max: Real,
    },
}

/// Fluid simulation and rendering support for the Rapier testbed.
///
/// The testbed no longer has a plugin system: the example owns its render
/// loop and calls this helper explicitly, e.g.:
///
/// ```ignore
/// while viewer.render_frame(&mut world).await {
///     plugin.update_from_settings(viewer.example_settings_mut());
///     plugin.draw(viewer);
///     if viewer.simulating() {
///         world.step();
///         plugin.step(&mut world);
///     }
/// }
/// ```
pub struct FluidsTestbedPlugin {
    /// Whether to render the boundary particles.
    pub render_boundary_particles: bool,
    /// Rendering mode of fluid particles.
    pub fluids_rendering_mode: FluidsRenderingMode,
    fluids_pipeline: FluidsPipeline,
    dfsph_parameters: DfsphParameters,
    f2color: HashMap<FluidHandle, Vector3<Real>>,
    boundary2color: HashMap<BoundaryHandle, Vector3<Real>>,
    default_fluid_color: Vector3<Real>,
    // Scene nodes holding one instance (a low-resolution sphere) per particle.
    // They are (re)created lazily on the first `draw` after each example
    // (re)start, since the testbed clears the whole kiss3d scene between
    // examples.
    fluids_node: Option<SceneNode>,
    boundaries_node: Option<SceneNode>,
}

impl FluidsTestbedPlugin {
    /// Initializes the plugin.
    pub fn new() -> Self {
        Self {
            render_boundary_particles: false,
            fluids_rendering_mode: FluidsRenderingMode::StaticColor,
            fluids_pipeline: FluidsPipeline::new(0.025, 2.0),
            dfsph_parameters: DfsphParameters::default(),
            f2color: HashMap::new(),
            boundary2color: HashMap::new(),
            default_fluid_color: Vector3::new(0.0, 0.0, 0.5),
            fluids_node: None,
            boundaries_node: None,
        }
    }

    /// Sets the fluids pipeline used by the testbed.
    pub fn set_pipeline(&mut self, fluids_pipeline: FluidsPipeline) {
        self.fluids_pipeline = fluids_pipeline;
        self.fluids_pipeline.liquid_world.counters.enable();
        self.refresh_dfsph_parameters();
    }

    /// The fluids pipeline managed by this plugin.
    pub fn pipeline(&self) -> &FluidsPipeline {
        &self.fluids_pipeline
    }

    /// Mutable access to the fluids pipeline managed by this plugin.
    pub fn pipeline_mut(&mut self) -> &mut FluidsPipeline {
        &mut self.fluids_pipeline
    }

    /// Sets the color used to render the specified fluid.
    pub fn set_fluid_color(&mut self, fluid: FluidHandle, color: Vector3<Real>) {
        let _ = self.f2color.insert(fluid, color);
    }

    /// Sets the color used to render the specified boundary.
    pub fn set_boundary_color(&mut self, boundary: BoundaryHandle, color: Vector3<Real>) {
        let _ = self.boundary2color.insert(boundary, color);
    }

    /// Sets the way fluids are rendered.
    pub fn set_fluid_rendering_mode(&mut self, mode: FluidsRenderingMode) {
        self.fluids_rendering_mode = mode;
    }

    /// Enables the rendering of boundary particles.
    pub fn enable_boundary_particles_rendering(&mut self, enabled: bool) {
        self.render_boundary_particles = enabled;
    }

    /// Steps the fluids simulation, coupling it with the rigid-bodies of `world`.
    ///
    /// Call this right after `world.step()`.
    pub fn step(&mut self, world: &mut PhysicsWorld) {
        let dt = world.integration_parameters.dt;
        self.fluids_pipeline
            .step(&world.gravity, dt, &world.colliders, &mut world.bodies);
    }

    /// Renders the fluid (and optionally boundary) particles as instanced
    /// low-resolution spheres (circles in 2D).
    ///
    /// Call this once per frame.
    pub fn draw(&mut self, viewer: &mut TestbedViewer) {
        let draw_velocities = matches!(
            self.fluids_rendering_mode,
            FluidsRenderingMode::VelocityArrows { .. }
        );

        // Collect one instance per particle (and the velocity arrows, drawn
        // separately in immediate mode).
        let mut fluid_instances = Vec::new();
        let mut arrows = Vec::new();

        for (handle, fluid) in self.fluids_pipeline.liquid_world.fluids().iter() {
            let diameter = Self::particle_diameter(fluid.particle_radius());

            for (point, velocity) in fluid.positions.iter().zip(fluid.velocities.iter()) {
                let color = self.fluid_color(handle, velocity);
                fluid_instances.push(Self::particle_instance(point, color, diameter));

                if draw_velocities && velocity.norm_squared() > na::zero() {
                    arrows.push((*point, *velocity, color));
                }
            }
        }

        let mut boundary_instances = Vec::new();

        if self.render_boundary_particles {
            let default_color = Vector3::repeat(na::convert::<_, Real>(0.5));
            let diameter =
                Self::particle_diameter(self.fluids_pipeline.liquid_world.particle_radius());

            for (handle, boundary) in self.fluids_pipeline.liquid_world.boundaries().iter() {
                let color =
                    Self::color(*self.boundary2color.get(&handle).unwrap_or(&default_color));

                for point in &boundary.positions {
                    boundary_instances.push(Self::particle_instance(point, color, diameter));
                }
            }
        }

        let window = viewer.window_mut();
        for (point, velocity, color) in arrows {
            Self::draw_velocity(window, &point, &velocity, color);
        }

        let scene = viewer.graphics_mut().scene_mut();

        if self.fluids_node.is_none() {
            self.fluids_node = Some(Self::particles_node(scene));
        }
        let _ = self
            .fluids_node
            .as_mut()
            .unwrap()
            .set_instances(&fluid_instances);

        // An empty instance list renders nothing, so this also hides the
        // boundaries when their rendering is disabled.
        if self.boundaries_node.is_none() && !boundary_instances.is_empty() {
            self.boundaries_node = Some(Self::particles_node(scene));
        }
        if let Some(boundaries_node) = &mut self.boundaries_node {
            let _ = boundaries_node.set_instances(&boundary_instances);
        }
    }

    /// A string summarizing the fluids simulation timings.
    pub fn profiling_string(&self) -> String {
        format!(
            "Fluids: {:.2}ms",
            self.fluids_pipeline.liquid_world.counters.step_time.time()
        )
    }

    fn color(color: Vector3<Real>) -> Color {
        Color::new(color.x as f32, color.y as f32, color.z as f32, 1.0)
    }

    fn fluid_color(&self, handle: FluidHandle, velocity: &Vector<Real>) -> Color {
        let base = *self
            .f2color
            .get(&handle)
            .unwrap_or(&self.default_fluid_color);

        match self.fluids_rendering_mode {
            FluidsRenderingMode::StaticColor => Self::color(base),
            FluidsRenderingMode::VelocityColor { min, max }
            | FluidsRenderingMode::VelocityArrows { min, max } => {
                Self::color(Self::velocity_tinted_color(base, velocity, min, max))
            }
        }
    }

    fn velocity_tinted_color(
        base: Vector3<Real>,
        velocity: &Vector<Real>,
        min: Real,
        max: Real,
    ) -> Vector3<Real> {
        let range = max - min;

        if range <= na::zero() {
            return base;
        }

        let factor = ((velocity.norm() - min) / range)
            .max(na::zero())
            .min(na::one());
        let inv_factor = na::one::<Real>() - factor;

        Vector3::new(
            base.x * inv_factor + factor,
            base.y * inv_factor,
            base.z * inv_factor,
        )
    }

    fn particle_diameter(radius: Real) -> f32 {
        radius as f32 * 2.0
    }

    fn refresh_dfsph_parameters(&mut self) {
        if let Some(parameters) = self.fluids_pipeline.dfsph_parameters() {
            self.dfsph_parameters = parameters;
        }
    }

    fn rendering_mode_index(&self) -> usize {
        FLUIDS_RENDERING_MAP
            .iter()
            .position(|(_, mode)| *mode == self.fluids_rendering_mode)
            .unwrap_or(0)
    }

    fn get_clamped_u32_setting(
        settings: &mut ExampleSettings,
        key: &'static str,
        default: u32,
        min: u32,
        max: u32,
    ) -> u32 {
        let value = settings
            .get_or_set_u32(key, default.clamp(min, max), min..=max)
            .clamp(min, max);
        settings.set_u32(key, value, min..=max);
        value
    }

    /// Registers and reads the fluid-related settings of the testbed's settings panel.
    ///
    /// Call this once per frame with `viewer.example_settings_mut()`.
    pub fn update_from_settings(&mut self, settings: &mut ExampleSettings) {
        self.render_boundary_particles =
            settings.get_or_set_bool(SETTINGS_RENDER_BOUNDARIES, self.render_boundary_particles);

        let rendering_options = FLUIDS_RENDERING_MAP
            .iter()
            .map(|(name, _)| (*name).to_string())
            .collect();
        let rendering_mode = settings.get_or_set_string(
            SETTINGS_RENDERING_MODE,
            self.rendering_mode_index(),
            rendering_options,
        );

        if let Some((_, mode)) = FLUIDS_RENDERING_MAP.get(rendering_mode) {
            self.fluids_rendering_mode = *mode;
        }

        let max_pressure_iter = Self::get_clamped_u32_setting(
            settings,
            SETTINGS_MAX_PRESSURE_ITER,
            self.dfsph_parameters.max_pressure_iter as u32,
            1,
            80,
        ) as usize;
        let max_divergence_iter = Self::get_clamped_u32_setting(
            settings,
            SETTINGS_MAX_DIVERGENCE_ITER,
            self.dfsph_parameters.max_divergence_iter as u32,
            1,
            80,
        ) as usize;
        let max_density_error = settings.get_or_set_f32(
            SETTINGS_MAX_DENSITY_ERROR,
            self.dfsph_parameters.max_density_error as f32,
            0.0..=0.5,
        );
        let max_divergence_error = settings.get_or_set_f32(
            SETTINGS_MAX_DIVERGENCE_ERROR,
            self.dfsph_parameters.max_divergence_error as f32,
            0.0..=2.0,
        );

        let boundary_force = settings.get_or_set_f32(
            SETTINGS_BOUNDARY_FORCE_COEFFICIENT,
            self.fluids_pipeline.liquid_world.boundary_force_coefficient as f32,
            0.0..=1.0,
        );
        self.fluids_pipeline.liquid_world.boundary_force_coefficient =
            na::convert::<_, Real>(boundary_force);

        let dfsph_parameters = DfsphParameters {
            min_pressure_iter: self.dfsph_parameters.min_pressure_iter,
            max_pressure_iter,
            max_density_error: na::convert::<_, Real>(max_density_error),
            min_divergence_iter: self.dfsph_parameters.min_divergence_iter,
            max_divergence_iter,
            max_divergence_error: na::convert::<_, Real>(max_divergence_error),
        };

        if dfsph_parameters != self.dfsph_parameters {
            self.dfsph_parameters = dfsph_parameters;
            let _ = self.fluids_pipeline.set_dfsph_parameters(dfsph_parameters);
        }
    }

    /// Draws an egui window with the fluid rendering options.
    ///
    /// Call this once per frame with `viewer.egui_context()`. This is an
    /// alternative to [`Self::update_from_settings`] for examples that prefer
    /// a dedicated window over the testbed's settings panel.
    pub fn update_ui(&mut self, ui_context: &egui::Context) {
        let _ = egui::Window::new("Fluids").show(ui_context, |ui| {
            let _ = ui.checkbox(
                &mut self.render_boundary_particles,
                "Render boundary particles",
            );

            let selected = FLUIDS_RENDERING_MAP
                .iter()
                .find_map(|(name, mode)| (*mode == self.fluids_rendering_mode).then_some(*name))
                .unwrap_or("Custom");

            let _ = egui::ComboBox::from_label("Rendering mode")
                .selected_text(selected)
                .show_ui(ui, |ui| {
                    for (name, mode) in FLUIDS_RENDERING_MAP {
                        let _ = ui.selectable_value(&mut self.fluids_rendering_mode, mode, name);
                    }
                });
        });
    }

    #[cfg(feature = "dim2")]
    fn point(point: &Vector<Real>) -> Vec2 {
        Vec2::new(point.x as f32, point.y as f32)
    }

    #[cfg(feature = "dim3")]
    fn point(point: &Vector<Real>) -> Vec3 {
        Vec3::new(point.x as f32, point.y as f32, point.z as f32)
    }

    /// Creates the unit-diameter node whose instances render the particles.
    #[cfg(feature = "dim2")]
    fn particles_node(scene: &mut SceneNode) -> SceneNode {
        let mut node = scene.add_circle_with_subdiv(0.5, 12);
        let _ = node.set_instances(&[]);
        node
    }

    /// Creates the unit-diameter node whose instances render the particles.
    #[cfg(feature = "dim3")]
    fn particles_node(scene: &mut SceneNode) -> SceneNode {
        let mut node = scene.add_sphere_with_subdiv(0.5, 8, 4);
        let _ = node.set_instances(&[]);
        node
    }

    #[cfg(feature = "dim2")]
    fn particle_instance(point: &Vector<Real>, color: Color, diameter: f32) -> InstanceData {
        InstanceData {
            position: Self::point(point),
            deformation: Mat2::from_diagonal(Vec2::splat(diameter)),
            color: [color.r, color.g, color.b, color.a],
            ..Default::default()
        }
    }

    #[cfg(feature = "dim3")]
    fn particle_instance(point: &Vector<Real>, color: Color, diameter: f32) -> InstanceData {
        InstanceData {
            position: Self::point(point),
            deformation: Mat3::from_diagonal(Vec3::splat(diameter)),
            color,
            ..Default::default()
        }
    }

    #[cfg(feature = "dim2")]
    fn draw_velocity(
        window: &mut Window,
        point: &Vector<Real>,
        velocity: &Vector<Real>,
        color: Color,
    ) {
        let end = *point + *velocity * na::convert::<_, Real>(0.02);
        window.draw_line_2d(Self::point(point), Self::point(&end), color, 1.5);
    }

    #[cfg(feature = "dim3")]
    fn draw_velocity(
        window: &mut Window,
        point: &Vector<Real>,
        velocity: &Vector<Real>,
        color: Color,
    ) {
        let end = *point + *velocity * na::convert::<_, Real>(0.02);
        window.draw_line(Self::point(point), Self::point(&end), color, 1.5, false);
    }
}

impl Default for FluidsTestbedPlugin {
    fn default() -> Self {
        Self::new()
    }
}
