extern crate nalgebra as na;

use na::{Point2, Point3, Vector2};
use ncollide2d::shape::Cuboid;
use rapier2d::dynamics::{JointSet, RigidBodyBuilder, RigidBodySet};
use rapier2d::geometry::{ColliderBuilder, ColliderSet};
use rapier_testbed2d::Testbed;
use salva2d::integrations::rapier::{
    ColliderSampling, FluidsPipeline, FluidsRenderingMode, FluidsTestbedPlugin,
};
use salva2d::object::{Boundary, Fluid};
use salva2d::solver::ArtificialViscosity;
use std::f32;

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

    // Liquid.
    let mut points1 = Vec::new();
    let mut points2 = Vec::new();
    let ni = 25;
    let nj = 15;

    let shift2 = (nj as f32) * PARTICLE_RADIUS * 2.0;

    for i in 0..ni {
        for j in 0..nj {
            let x = (i as f32) * PARTICLE_RADIUS * 2.0 - ni as f32 * PARTICLE_RADIUS;
            let y = (j as f32 + 1.0) * PARTICLE_RADIUS * 2.0;
            points1.push(Point2::new(x, y));
            points2.push(Point2::new(x, y + shift2));
        }
    }

    let viscosity = ArtificialViscosity::new(0.5, 0.0);
    let mut fluid = Fluid::new(points1, PARTICLE_RADIUS, 1.0);
    fluid.nonpressure_forces.push(Box::new(viscosity.clone()));
    let fluid_handle = fluids_pipeline.liquid_world.add_fluid(fluid);
    plugin.set_fluid_color(fluid_handle, Point3::new(0.8, 0.7, 1.0));

    let mut fluid = Fluid::new(points2, PARTICLE_RADIUS, 1.0);
    fluid.nonpressure_forces.push(Box::new(viscosity.clone()));
    let fluid_handle = fluids_pipeline.liquid_world.add_fluid(fluid);
    plugin.set_fluid_color(fluid_handle, Point3::new(0.6, 0.8, 0.5));

    /*
     *
     * Ground cuboid.
     *
     */
    let ground_size = 25.0;

    let ground_handle = bodies.insert(RigidBodyBuilder::new_static().build());
    let co = ColliderBuilder::cuboid(ground_size, 1.0)
        .translation(0.0, -1.0)
        .build();
    let co_handle = colliders.insert(co, ground_handle, &mut bodies);
    let bo_handle = fluids_pipeline
        .liquid_world
        .add_boundary(Boundary::new(Vec::new()));
    fluids_pipeline.coupling.register_coupling(
        bo_handle,
        co_handle,
        ColliderSampling::DynamicContactSampling,
    );

    let co = ColliderBuilder::cuboid(1.0, ground_size)
        .translation(-5.0, 0.0)
        .rotation(0.1)
        .build();
    let co_handle = colliders.insert(co, ground_handle, &mut bodies);
    let bo_handle = fluids_pipeline
        .liquid_world
        .add_boundary(Boundary::new(Vec::new()));
    fluids_pipeline.coupling.register_coupling(
        bo_handle,
        co_handle,
        ColliderSampling::DynamicContactSampling,
    );

    let co = ColliderBuilder::cuboid(1.0, ground_size)
        .translation(5.0, 0.0)
        .rotation(-0.1)
        .build();
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
     * Create a dynamic box.
     */
    let rad = 0.4;
    let cuboid = Cuboid::new(Vector2::repeat(rad));
    let cuboid_sample =
        salva2d::sampling::shape_surface_ray_sample(&cuboid, PARTICLE_RADIUS).unwrap();

    // Build the rigid body.
    let rb = RigidBodyBuilder::new_dynamic()
        .translation(0.0, 10.0)
        .build();
    let rb_handle = bodies.insert(rb);
    testbed.set_body_color(rb_handle, Point3::new(0.3, 0.3, 0.7));

    // Build the collider.
    let co = ColliderBuilder::cuboid(rad, rad).density(0.9).build();
    let co_handle = colliders.insert(co, rb_handle, &mut bodies);
    let bo_handle = fluids_pipeline
        .liquid_world
        .add_boundary(Boundary::new(Vec::new()));
    fluids_pipeline.coupling.register_coupling(
        bo_handle,
        co_handle,
        ColliderSampling::StaticSampling(cuboid_sample),
    );

    /*
     * Set up the testbed.
     */
    plugin.set_pipeline(fluids_pipeline);
    plugin.set_fluid_rendering_mode(FluidsRenderingMode::VelocityColor { min: 0.0, max: 5.0 });
    testbed.add_plugin(plugin);
    testbed.set_world_with_gravity(bodies, colliders, joints, gravity);
    testbed.integration_parameters_mut().set_dt(1.0 / 100.0);
    //    testbed.enable_boundary_particles_rendering(true);
}
