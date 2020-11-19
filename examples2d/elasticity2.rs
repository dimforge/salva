extern crate nalgebra as na;

use na::{Isometry2, Point2, Point3, Vector2};
use rapier2d::dynamics::{JointSet, RigidBodyBuilder, RigidBodySet};
use rapier2d::geometry::{ColliderBuilder, ColliderSet};
use rapier_testbed2d::Testbed;
use salva2d::integrations::rapier::{
    ColliderSampling, FluidsPipeline, FluidsRenderingMode, FluidsTestbedPlugin,
};
use salva2d::object::Boundary;
use salva2d::solver::{Becker2009Elasticity, XSPHViscosity};
use std::f32;

#[path = "./helper.rs"]
mod helper;

const PARTICLE_RADIUS: f32 = 0.1;
const SMOOTHING_FACTOR: f32 = 2.0;

pub fn init_world(testbed: &mut Testbed) {
    /*
     * World
     */
    let gravity = Vector2::y() * -9.81;
    let mut plugin = FluidsTestbedPlugin::new();
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let joints = JointSet::new();
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
    plugin.set_fluid_color(fluid_handle, Point3::new(0.8, 0.7, 1.0));

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
    plugin.set_fluid_color(fluid_handle, Point3::new(0.6, 0.8, 0.5));

    // Setup the ground.
    let ground_handle = bodies.insert(RigidBodyBuilder::new_static().build());
    let co = ColliderBuilder::cuboid(ground_half_width, ground_thickness).build();
    let co_handle = colliders.insert(co, ground_handle, &mut bodies);
    let bo_handle = fluids_pipeline
        .liquid_world
        .add_boundary(Boundary::new(Vec::new()));
    fluids_pipeline.coupling.register_coupling(
        bo_handle,
        co_handle,
        ColliderSampling::DynamicContactSampling,
    );

    /*
     * Set up the testbed.
     */
    plugin.set_pipeline(fluids_pipeline);
    plugin.set_fluid_rendering_mode(FluidsRenderingMode::VelocityColor { min: 0.0, max: 5.0 });
    testbed.add_plugin(plugin);
    testbed.set_world_with_gravity(bodies, colliders, joints, gravity);
    testbed.integration_parameters_mut().set_dt(1.0 / 200.0);
    testbed.look_at(Point2::new(0.0, 1.0), 100.0);
}
