extern crate nalgebra as na;

use na::{Isometry2, Vector2, Vector3};
use rapier2d::dynamics::RigidBodyBuilder;
use rapier2d::geometry::ColliderBuilder;
use rapier2d::math::Vector;
use rapier2d::pipeline::PhysicsWorld;
use rapier_testbed2d::TestbedViewer;
use salva2d::integrations::rapier::{
    ColliderSampling, FluidsPipeline, FluidsRenderingMode, FluidsTestbedPlugin,
};
use salva2d::object::interaction_groups::InteractionGroups;
use salva2d::object::Boundary;
use salva2d::solver::{Akinci2013SurfaceTension, ArtificialViscosity};
use std::f32;

#[path = "./helper.rs"]
mod helper;

const PARTICLE_RADIUS: f32 = 0.0025;
const SMOOTHING_FACTOR: f32 = 2.0;

pub async fn run(viewer: &mut TestbedViewer) -> anyhow::Result<()> {
    /*
     * World
     */
    // We want to simulate a 1cm³ droplet. We use the spacial unit 1 = 1dm.
    // Therefore each particles must have a diameter of 0.005, and the gravity is -0.981 instead of -9.81.
    let mut world = PhysicsWorld::new();
    world.gravity = (Vector2::y() * -0.981).into();
    world.integration_parameters.dt = 1.0 / 200.0;

    let mut plugin = FluidsTestbedPlugin::new();
    let mut fluids_pipeline = FluidsPipeline::new(PARTICLE_RADIUS, SMOOTHING_FACTOR);

    // Initialize the fluid and give it some surface tension. This will make the fluid take a spherical shape.
    let surface_tension = Akinci2013SurfaceTension::new(1.0, 0.0);
    let viscosity = ArtificialViscosity::new(0.01, 0.0);
    let mut fluid = helper::cube_fluid(20, 20, PARTICLE_RADIUS, 1000.0);
    fluid.transform_by(&Isometry2::translation(0.0, 0.08));
    fluid.nonpressure_forces.push(Box::new(surface_tension));
    fluid.nonpressure_forces.push(Box::new(viscosity));
    let fluid_handle = fluids_pipeline.liquid_world.add_fluid(fluid);
    plugin.set_fluid_color(fluid_handle, Vector3::new(0.8, 0.7, 1.0));

    // Setup the ground.
    let ground_thickness = 0.02;
    let ground_half_width = 0.15;

    let ground_handle = world.bodies.insert(RigidBodyBuilder::fixed().build());
    let co = ColliderBuilder::cuboid(ground_half_width, ground_thickness).build();
    let co_handle = world
        .colliders
        .insert_with_parent(co, ground_handle, &mut world.bodies);
    let bo_handle = fluids_pipeline
        .liquid_world
        .add_boundary(Boundary::new(Vec::new(), InteractionGroups::default()));
    fluids_pipeline.coupling.register_coupling(
        bo_handle,
        co_handle,
        ColliderSampling::DynamicContactSampling,
    );

    /*
     * Set up the viewer and run the simulation.
     */
    plugin.set_pipeline(fluids_pipeline);
    plugin.set_fluid_rendering_mode(FluidsRenderingMode::VelocityColor { min: 0.0, max: 5.0 });
    viewer.set_world(&mut world);
    viewer.look_at(Vector::ZERO, 1500.0);

    while viewer.render_frame(&mut world).await {
        plugin.update_from_settings(viewer.example_settings_mut());
        plugin.draw(viewer);

        if viewer.simulating() {
            world.step();
            plugin.step(&mut world);
        }
    }

    Ok(())
}
