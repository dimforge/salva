extern crate nalgebra as na;

use na::{DVector, Point2, Point3, Vector2};
use rapier2d::dynamics::{JointSet, RigidBodyBuilder, RigidBodySet};
use rapier2d::geometry::{Collider, ColliderBuilder, ColliderSet};
use rapier_testbed2d::Testbed;
use salva2d::integrations::rapier::{ColliderSampling, FluidsPipeline, FluidsTestbedPlugin};
use salva2d::object::{Boundary, Fluid};
use salva2d::solver::{ArtificialViscosity, Becker2009Elasticity, XSPHViscosity};
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
    let mut points3 = Vec::new();
    let ni = 25;
    let nj = 15;

    let shift2 = (nj as f32) * PARTICLE_RADIUS * 2.0;

    for i in 0..ni / 2 {
        for j in 0..nj {
            let x = (i as f32) * PARTICLE_RADIUS * 2.0 - ni as f32 * PARTICLE_RADIUS;
            let y = (j as f32 + 1.0) * PARTICLE_RADIUS * 2.0 + 0.5;
            points1.push(Point2::new(x, y));
            points2.push(Point2::new(x + ni as f32 * PARTICLE_RADIUS, y));
        }
    }

    for i in 0..ni {
        for j in 0..nj * 2 {
            let x = (i as f32) * PARTICLE_RADIUS * 2.0 - ni as f32 * PARTICLE_RADIUS;
            let y = (j as f32 + 1.0) * PARTICLE_RADIUS * 2.0 + 0.5;
            points3.push(Point2::new(x, y + shift2));
        }
    }

    let elasticity: Becker2009Elasticity = Becker2009Elasticity::new(1_000.0, 0.3, true);
    let viscosity = XSPHViscosity::new(0.5, 1.0);
    let mut fluid = Fluid::new(points1, PARTICLE_RADIUS, 1.0);
    fluid.nonpressure_forces.push(Box::new(elasticity));
    fluid.nonpressure_forces.push(Box::new(viscosity.clone()));
    let fluid_handle = fluids_pipeline.liquid_world.add_fluid(fluid);
    plugin.set_fluid_color(fluid_handle, Point3::new(0.8, 0.7, 1.0));

    let elasticity: Becker2009Elasticity = Becker2009Elasticity::new(1_000.0, 0.3, true);
    let viscosity = XSPHViscosity::new(0.5, 1.0);
    let mut fluid = Fluid::new(points2, PARTICLE_RADIUS, 1.0);
    fluid.nonpressure_forces.push(Box::new(elasticity));
    fluid.nonpressure_forces.push(Box::new(viscosity.clone()));
    let fluid_handle = fluids_pipeline.liquid_world.add_fluid(fluid);
    plugin.set_fluid_color(fluid_handle, Point3::new(1.0, 0.4, 0.6));

    let viscosity = ArtificialViscosity::new(0.5, 0.0);
    let mut fluid = Fluid::new(points3, PARTICLE_RADIUS, 1.0);
    fluid.nonpressure_forces.push(Box::new(viscosity.clone()));
    let fluid_handle = fluids_pipeline.liquid_world.add_fluid(fluid);
    plugin.set_fluid_color(fluid_handle, Point3::new(0.6, 0.8, 0.5));

    /*
     * Ground
     */
    let ground_size = Vector2::new(10.0, 1.0);
    let nsubdivs = 50;

    let heights = DVector::from_fn(nsubdivs + 1, |i, _| {
        if i == 0 || i == nsubdivs {
            20.0
        } else {
            (i as f32 * ground_size.x / (nsubdivs as f32)).cos() * 0.5
        }
    });

    let rigid_body = RigidBodyBuilder::new_static().build();
    let handle = bodies.insert(rigid_body);
    let collider = ColliderBuilder::heightfield(heights, ground_size).build();
    let co_handle = colliders.insert(collider, handle, &mut bodies);
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
    let mut build_rigid_body_with_coupling = |x, y, collider: Collider| {
        let samples =
            salva2d::sampling::shape_surface_ray_sample(collider.shape(), PARTICLE_RADIUS).unwrap();
        let rb = RigidBodyBuilder::new_dynamic().translation(x, y).build();
        let rb_handle = bodies.insert(rb);
        let co_handle = colliders.insert(collider, rb_handle, &mut bodies);
        let bo_handle = fluids_pipeline
            .liquid_world
            .add_boundary(Boundary::new(Vec::new()));
        fluids_pipeline.coupling.register_coupling(
            bo_handle,
            co_handle,
            ColliderSampling::StaticSampling(samples.clone()),
        );
        testbed.set_body_color(rb_handle, Point3::new(0.3, 0.3, 0.7));
    };

    let co1 = ColliderBuilder::cuboid(rad, rad).density(0.8).build();
    let co2 = ColliderBuilder::ball(rad).density(0.8).build();
    let co3 = ColliderBuilder::capsule_y(rad, rad).density(0.8).build();
    build_rigid_body_with_coupling(0.0, 10.0, co1);
    build_rigid_body_with_coupling(-2.0, 10.0, co2);
    build_rigid_body_with_coupling(2.0, 10.5, co3);

    /*
     * Set up the testbed.
     */
    plugin.set_pipeline(fluids_pipeline);
    testbed.add_plugin(plugin);
    testbed.set_world_with_gravity(bodies, colliders, joints, gravity);
    testbed.integration_parameters_mut().dt = 1.0 / 200.0;
    //    testbed.enable_boundary_particles_rendering(true);
}
