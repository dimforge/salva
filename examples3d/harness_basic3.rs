//! Headless run: builds a fluid cube over a walled ground and steps the
//! coupled rapier + salva simulation without any window or testbed.

extern crate nalgebra as na;

use na::Isometry3;
use rapier3d::dynamics::RigidBodyBuilder;
use rapier3d::geometry::{ColliderBuilder, SharedShape};
use rapier3d::math::{Pose, Vector};
use rapier3d::pipeline::PhysicsWorld;
use salva3d::integrations::rapier::{ColliderSampling, FluidsPipeline};
use salva3d::object::interaction_groups::InteractionGroups;
use salva3d::object::Boundary;
use salva3d::solver::ArtificialViscosity;
use std::f32;

#[path = "./helper.rs"]
mod helper;

const PARTICLE_RADIUS: f32 = 0.025;
const SMOOTHING_FACTOR: f32 = 2.0;

fn main() {
    /*
     * World
     */
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::Y * -9.81;
    world.integration_parameters.dt = 1.0 / 200.0;
    let mut fluids_pipeline = FluidsPipeline::new(PARTICLE_RADIUS, SMOOTHING_FACTOR);

    // Parameters of the ground.
    let ground_thickness = 0.2;
    let ground_half_width = 1.5;
    let ground_half_height = 0.7;

    // fluids.
    let nparticles = 15;
    let mut fluid = helper::cube_fluid(nparticles, nparticles, nparticles, PARTICLE_RADIUS, 1000.0);
    fluid.transform_by(&Isometry3::translation(
        0.0,
        ground_thickness + nparticles as f32 * PARTICLE_RADIUS,
        0.0,
    ));
    let viscosity = ArtificialViscosity::new(1.0, 0.0);
    fluid.nonpressure_forces.push(Box::new(viscosity));
    let _fluid_handle = fluids_pipeline.liquid_world.add_fluid(fluid);

    /*
     * Ground.
     */
    let ground_shape = SharedShape::cuboid(ground_half_width, ground_thickness, ground_half_width);
    let wall_shape = SharedShape::cuboid(ground_thickness, ground_half_height, ground_half_width);

    let ground_body = RigidBodyBuilder::fixed().build();
    let ground_handle = world.bodies.insert(ground_body);

    let wall_poses = [
        Pose::new(
            Vector::new(0.0, ground_half_height, ground_half_width),
            Vector::Y * (f32::consts::PI / 2.0),
        ),
        Pose::new(
            Vector::new(0.0, ground_half_height, -ground_half_width),
            Vector::Y * (f32::consts::PI / 2.0),
        ),
        Pose::from_translation(Vector::new(ground_half_width, ground_half_height, 0.0)),
        Pose::from_translation(Vector::new(-ground_half_width, ground_half_height, 0.0)),
    ];

    for pose in wall_poses.iter() {
        let samples =
            salva3d::sampling::shape_surface_ray_sample(&*wall_shape, PARTICLE_RADIUS).unwrap();
        let co = ColliderBuilder::new(wall_shape.clone())
            .position(*pose)
            .build();
        let co_handle = world
            .colliders
            .insert_with_parent(co, ground_handle, &mut world.bodies);
        let bo_handle = fluids_pipeline
            .liquid_world
            .add_boundary(Boundary::new(Vec::new(), InteractionGroups::default()));

        fluids_pipeline.coupling.register_coupling(
            bo_handle,
            co_handle,
            ColliderSampling::StaticSampling(samples),
        );
    }

    let samples =
        salva3d::sampling::shape_surface_ray_sample(&*ground_shape, PARTICLE_RADIUS).unwrap();
    let co = ColliderBuilder::new(ground_shape).build();
    let co_handle = world
        .colliders
        .insert_with_parent(co, ground_handle, &mut world.bodies);
    let bo_handle = fluids_pipeline
        .liquid_world
        .add_boundary(Boundary::new(Vec::new(), InteractionGroups::default()));

    fluids_pipeline.coupling.register_coupling(
        bo_handle,
        co_handle,
        ColliderSampling::StaticSampling(samples),
    );

    /*
     * Run the simulation.
     */
    let num_steps = 1000;
    for _ in 0..num_steps {
        world.step();
        fluids_pipeline.step(
            &world.gravity,
            world.integration_parameters.dt,
            &world.colliders,
            &mut world.bodies,
        );
    }
    println!("Ran {num_steps} coupled steps.");
}
