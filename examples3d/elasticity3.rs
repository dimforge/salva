extern crate nalgebra as na;

use na::{Isometry3, Point3, Vector3};
use rapier3d::dynamics::{JointSet, RigidBodyBuilder, RigidBodySet};
use rapier3d::geometry::{ColliderBuilder, ColliderSet};
use rapier_testbed3d::{Testbed, TestbedApp};
use salva3d::integrations::rapier::{
    ColliderSampling, FluidsPipeline, FluidsRenderingMode, FluidsTestbedPlugin,
};
use salva3d::object::Boundary;
use salva3d::solver::{Becker2009Elasticity, XSPHViscosity};
use std::f32;

#[path = "./helper.rs"]
mod helper;

const PARTICLE_RADIUS: f32 = 0.025;
const SMOOTHING_FACTOR: f32 = 2.0;

pub fn init_world(testbed: &mut Testbed) {
    /*
     * World
     */
    let gravity = Vector3::y() * -9.81;
    let mut plugin = FluidsTestbedPlugin::new();
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let joints = JointSet::new();
    let mut fluids_pipeline = FluidsPipeline::new(PARTICLE_RADIUS, SMOOTHING_FACTOR);

    // Parameters of the ground.
    let ground_thickness = 0.2;
    let ground_half_width = 1.5;

    // Initialize the fluids and give them elasticity.
    let height = 0.4;
    let nparticles = 6;

    // First fluid with high young modulus.
    let elasticity: Becker2009Elasticity = Becker2009Elasticity::new(500_000.0, 0.3, true);
    let viscosity = XSPHViscosity::new(0.5, 1.0);
    let mut fluid = helper::cube_fluid(
        nparticles * 2,
        nparticles,
        nparticles * 2,
        PARTICLE_RADIUS,
        1000.0,
    );
    fluid.transform_by(&Isometry3::translation(
        0.0,
        ground_thickness + PARTICLE_RADIUS * nparticles as f32 + height,
        0.0,
    ));
    fluid.nonpressure_forces.push(Box::new(elasticity));
    fluid.nonpressure_forces.push(Box::new(viscosity.clone()));
    let fluid_handle = fluids_pipeline.liquid_world.add_fluid(fluid);
    plugin.set_fluid_color(fluid_handle, Point3::new(0.8, 0.7, 1.0));

    // Second fluid with smaller young modulus.
    let elasticity: Becker2009Elasticity = Becker2009Elasticity::new(100_000.0, 0.3, true);
    let mut fluid = helper::cube_fluid(
        nparticles * 2,
        nparticles,
        nparticles * 2,
        PARTICLE_RADIUS,
        1000.0,
    );
    fluid.transform_by(&Isometry3::translation(
        0.0,
        ground_thickness + PARTICLE_RADIUS * nparticles as f32 * 4.0 + height,
        0.0,
    ));
    fluid.nonpressure_forces.push(Box::new(elasticity));
    fluid.nonpressure_forces.push(Box::new(viscosity));
    let fluid_handle = fluids_pipeline.liquid_world.add_fluid(fluid);
    plugin.set_fluid_color(fluid_handle, Point3::new(0.6, 0.8, 0.5));

    // Setup the ground.
    let ground_handle = bodies.insert(RigidBodyBuilder::new_static().build());
    let co =
        ColliderBuilder::cuboid(ground_half_width, ground_thickness, ground_half_width).build();
    let co_handle = colliders.insert(co);
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
    testbed.set_body_wireframe(ground_handle, true);
    testbed.set_world_with_params(bodies, colliders, joints, gravity, ());
    testbed.integration_parameters_mut().dt = 1.0 / 200.0;
    testbed.look_at(Point3::new(1.5, 1.5, 1.5), Point3::origin());
}

fn main() {
    let testbed = TestbedApp::from_builders(0, vec![("Elasticity", init_world)]);
    testbed.run()
}
