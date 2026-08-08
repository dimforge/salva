extern crate nalgebra as na;

use na::{Isometry2, Vector2, Vector3};
use rapier2d::dynamics::RigidBodyBuilder;
use rapier2d::geometry::ColliderBuilder;
use rapier2d::pipeline::PhysicsWorld;
use rapier_testbed2d::TestbedViewer;
use salva2d::integrations::rapier::{
    ColliderSampling, FluidsPipeline, FluidsRenderingMode, FluidsTestbedPlugin,
};
use salva2d::object::interaction_groups::InteractionGroups;
use salva2d::object::Boundary;
use salva2d::solver::{Becker2009Elasticity, XSPHViscosity};
use std::f32;

#[path = "./helper.rs"]
mod helper;

const PARTICLE_RADIUS: f32 = 0.1;
const SMOOTHING_FACTOR: f32 = 2.0;

pub async fn run(viewer: &mut TestbedViewer) -> anyhow::Result<()> {
    /*
     * World
     */
    let mut world = PhysicsWorld::new();
    world.gravity = (Vector2::y() * -9.81).into();
    world.integration_parameters.dt = 1.0 / 200.0;

    let mut plugin = FluidsTestbedPlugin::new();
    let mut fluids_pipeline = FluidsPipeline::new(PARTICLE_RADIUS, SMOOTHING_FACTOR);

    let ground_thickness = 0.2;
    let ground_half_width = 3.0;

    // Initialize the fluids and give them elasticity.
    let height = 0.4;
    let nparticlesx = 25;
    let nparticlesy = 15;

    // First fluid with high young modulus.
    let elasticity: Becker2009Elasticity = Becker2009Elasticity::new(500_000.0, 0.3, true);
    let viscosity = XSPHViscosity::new(0.5, 1.0);
    let mut fluid = helper::cube_fluid(nparticlesx, nparticlesy, PARTICLE_RADIUS, 1000.0);
    fluid.transform_by(&Isometry2::translation(
        0.0,
        ground_thickness + PARTICLE_RADIUS * nparticlesy as f32 + height,
    ));
    fluid.nonpressure_forces.push(Box::new(elasticity));
    fluid.nonpressure_forces.push(Box::new(viscosity.clone()));
    let fluid_handle = fluids_pipeline.liquid_world.add_fluid(fluid);
    plugin.set_fluid_color(fluid_handle, Vector3::new(0.8, 0.7, 1.0));

    // Second fluid with smaller young modulus.
    let elasticity: Becker2009Elasticity = Becker2009Elasticity::new(100_000.0, 0.3, true);
    let mut fluid = helper::cube_fluid(nparticlesx, nparticlesy, PARTICLE_RADIUS, 1000.0);
    fluid.transform_by(&Isometry2::translation(
        0.0,
        ground_thickness + PARTICLE_RADIUS * nparticlesy as f32 * 4.0 + height,
    ));
    fluid.nonpressure_forces.push(Box::new(elasticity));
    fluid.nonpressure_forces.push(Box::new(viscosity));
    let fluid_handle = fluids_pipeline.liquid_world.add_fluid(fluid);
    plugin.set_fluid_color(fluid_handle, Vector3::new(0.6, 0.8, 0.5));

    // Setup the ground.
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
    viewer.look_at(Vector2::new(0.0, 1.0).into(), 100.0);

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
