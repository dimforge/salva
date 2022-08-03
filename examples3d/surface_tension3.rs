extern crate nalgebra as na;

use na::{Isometry3, Point3, Vector3};
use rapier3d::dynamics::{ImpulseJointSet, RigidBodyBuilder, RigidBodySet};
use rapier3d::geometry::{ColliderBuilder, ColliderSet};
use rapier_testbed3d::{Testbed, TestbedApp};
use salva3d::integrations::rapier::{
    ColliderSampling, FluidsPipeline, FluidsRenderingMode, FluidsTestbedPlugin,
};
use salva3d::object::Boundary;
use salva3d::solver::{Akinci2013SurfaceTension, ArtificialViscosity};
use std::f32;

#[path = "./helper.rs"]
mod helper;

const PARTICLE_RADIUS: f32 = 0.005;
const SMOOTHING_FACTOR: f32 = 2.0;

pub fn init_world(testbed: &mut Testbed) {
    /*
     * World
     */
    // We want to simulate a 1cm³ droplet. We use the spacial unit 1 = 1dm.
    // Therefore each particles must have a diameter of 0.005, and the gravity is -0.981 instead of -9.81.
    let gravity = Vector3::y() * -0.981;
    let mut plugin = FluidsTestbedPlugin::new();
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let joints = ImpulseJointSet::new();
    let mut fluids_pipeline = FluidsPipeline::new(PARTICLE_RADIUS, SMOOTHING_FACTOR);

    /*
     * Liquid world.
     */
    // Initialize the fluid and give it some surface tension. This will make the fluid take a spherical shape.
    let surface_tension = Akinci2013SurfaceTension::new(1.0, 0.0);
    let viscosity = ArtificialViscosity::new(0.01, 0.01);
    let mut fluid = helper::cube_fluid(7, 7, 7, PARTICLE_RADIUS, 1000.0);
    fluid.transform_by(&Isometry3::translation(0.0, 0.08, 0.0));
    fluid.nonpressure_forces.push(Box::new(surface_tension));
    fluid.nonpressure_forces.push(Box::new(viscosity));
    let fluid_handle = fluids_pipeline.liquid_world.add_fluid(fluid);
    plugin.set_fluid_color(fluid_handle, Point3::new(0.8, 0.7, 1.0));

    // Setup the ground.
    let ground_handle = bodies.insert(RigidBodyBuilder::fixed().build());

    let ground_thickness = 0.02;
    let ground_half_width = 0.15;

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
    testbed.set_world_with_params(bodies, colliders, joints, Default::default(), gravity, ());
    testbed.integration_parameters_mut().dt = 1.0 / 200.0;
    testbed.look_at(Point3::new(0.25, 0.25, 0.25), Point3::origin());
}

fn main() {
    let testbed = TestbedApp::from_builders(0, vec![("Surface tension", init_world)]);
    testbed.run()
}
