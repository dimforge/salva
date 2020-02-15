extern crate nalgebra as na;

use na::{Isometry3, Point3, Vector3};
use ncollide3d::shape::{Capsule, Cuboid, ShapeHandle};
use nphysics3d::force_generator::DefaultForceGeneratorSet;
use nphysics3d::joint::DefaultJointConstraintSet;
use nphysics3d::object::{
    BodyPartHandle, ColliderDesc, DefaultBodySet, DefaultColliderSet, Ground, RigidBodyDesc,
};
use nphysics3d::world::{DefaultGeometricalWorld, DefaultMechanicalWorld};
use nphysics_testbed3d::Testbed;
use salva3d::coupling::{ColliderCouplingSet, CouplingMethod};
use salva3d::object::{Boundary, Fluid};
use salva3d::solver::Becker2009Elasticity;
use salva3d::LiquidWorld;
use std::f32;

pub fn init_world(testbed: &mut Testbed) {
    /*
     * World
     */
    let mechanical_world = DefaultMechanicalWorld::new(Vector3::new(0.0, -9.81, 0.0));
    let geometrical_world = DefaultGeometricalWorld::new();
    let mut bodies = DefaultBodySet::new();
    let mut colliders = DefaultColliderSet::new();
    let joint_constraints = DefaultJointConstraintSet::new();
    let force_generators = DefaultForceGeneratorSet::new();

    // Parameters of the ground.
    let ground_thickness = 0.2;
    let ground_half_width = 1.5;
    let ground_half_height = 0.7;

    /*
     * Liquid world.
     */
    let particle_rad = 0.025;
    let mut liquid_world = LiquidWorld::new(particle_rad, 2.0);
    let mut coupling_manager = ColliderCouplingSet::new();

    // Liquid.
    let mut points1 = Vec::new();
    //    let mut points2 = Vec::new();
    let ni = 13;
    let nj = 13;
    let nk = 13;

    let half_extents = Vector3::new(ni as f32, nj as f32, nk as f32) * particle_rad;
    let shift1 = Vector3::new(
        ground_half_width - half_extents.x - ColliderDesc::<f32>::default_margin() * 2.0,
        ground_thickness,
        ground_half_width - half_extents.z - ColliderDesc::<f32>::default_margin() * 2.0,
    );
    let shift2 = Vector3::new(-shift1.x, ground_thickness, -shift1.z);

    for i in 0..ni {
        for j in 0..nj {
            for k in 0..nk {
                let x = (i as f32) * particle_rad * 2.0 - half_extents.x;
                let y = (j as f32) * particle_rad * 2.0;
                let z = (k as f32) * particle_rad * 2.0 - half_extents.z;
                points1.push(Point3::new(x, y, z) + shift1 + Vector3::repeat(particle_rad));
                points1.push(Point3::new(x, y, z) + shift2 + Vector3::repeat(particle_rad));
            }
        }
    }

    let elasticity: Becker2009Elasticity<_> = Becker2009Elasticity::new(100_000.0, 0.3, false);
    let mut fluid = Fluid::new(points1, particle_rad, 1000.0, 0.5, 1.0, 1.0);
    fluid.nonpressure_forces.push(Box::new(elasticity));
    let fluid_handle = liquid_world.add_fluid(fluid);
    testbed.set_fluid_color(fluid_handle, Point3::new(0.8, 0.7, 1.0));

    //    let fluid = Fluid::new(points2, particle_rad, 1000.0, 0.9, 1.0, 1.02);
    //    let fluid_handle = liquid_world.add_fluid(fluid);
    //    testbed.set_fluid_color(fluid_handle, Point3::new(0.8, 0.7, 1.0));

    //    testbed.set_fluid_color(fluid_handle, Point3::new(0.6, 0.8, 0.5));

    //    let fluid = Fluid::new(points2, particle_rad, 1.0, 0.1, 1.0, 0.5);
    //    let fluid_handle = liquid_world.add_fluid(fluid);
    //    testbed.set_fluid_color(fluid_handle, Point3::new(0.6, 0.8, 0.5));

    //    let shape = Cuboid::new(Vector3::repeat(0.2));
    //    let aabb = ncollide3d::bounding_volume::local_aabb(&shape);
    //    let samples = salva3d::sampling::volume_ray_sample(&shape, &aabb, particle_rad * 2.0);
    //    println!("Num samples: {}", samples.len());
    //    let mut fluid = Fluid::new(samples, particle_rad, 1000.0, 0.5, 1.0, 1.015);
    //    fluid.positions.iter_mut().for_each(|p| p.y += 2.0);
    //    let fluid_handle = liquid_world.add_fluid(fluid);
    //    testbed.set_fluid_color(fluid_handle, Point3::new(0.6, 0.8, 0.5));

    /*
     * Ground.
     */
    let ground_shape = ShapeHandle::new(Cuboid::new(Vector3::new(
        ground_half_width,
        ground_thickness,
        ground_half_width,
    )));
    let wall_shape = ShapeHandle::new(Cuboid::new(Vector3::new(
        ground_thickness,
        ground_half_height,
        ground_half_width,
    )));

    let ground_handle = bodies.insert(Ground::new());

    let wall_poses = [
        Isometry3::new(
            Vector3::new(
                0.0,
                ground_half_height,
                ground_half_width + ground_thickness,
            ),
            Vector3::y() * (f32::consts::PI / 2.0),
        ),
        Isometry3::new(
            Vector3::new(
                0.0,
                ground_half_height,
                -ground_half_width - ground_thickness,
            ),
            Vector3::y() * (f32::consts::PI / 2.0),
        ),
        Isometry3::translation(
            ground_half_width + ground_thickness,
            ground_half_height,
            0.0,
        ),
        Isometry3::translation(
            -ground_half_width - ground_thickness,
            ground_half_height,
            0.0,
        ),
    ];

    for pose in wall_poses.into_iter() {
        let co = ColliderDesc::new(wall_shape.clone())
            .position(*pose)
            .build(BodyPartHandle(ground_handle, 0));
        let co_handle = colliders.insert(co);
        let bo_handle = liquid_world.add_boundary(Boundary::new(Vec::new()));
        coupling_manager.register_coupling(
            bo_handle,
            co_handle,
            CouplingMethod::DynamicContactSampling,
        );
    }

    let co = ColliderDesc::new(ground_shape).build(BodyPartHandle(ground_handle, 0));
    let co_handle = colliders.insert(co);
    let bo_handle = liquid_world.add_boundary(Boundary::new(Vec::new()));
    coupling_manager.register_coupling(
        bo_handle,
        co_handle,
        CouplingMethod::DynamicContactSampling,
    );

    /*
    /*
     * Create a cuboid.
     */
    let rad = 0.2;
    let cuboid = ShapeHandle::new(Cuboid::new(Vector3::repeat(rad)));

    // Build the rigid body.
    let rb = RigidBodyDesc::new().translation(Vector3::y() * 7.0).build();
    let rb_handle = bodies.insert(rb);
    testbed.set_body_color(rb_handle, Point3::new(0.3, 0.3, 0.7));

    // Build the collider.
    let co = ColliderDesc::new(cuboid.clone())
        .density(1.0)
        .build(BodyPartHandle(rb_handle, 0));
    let co_handle = colliders.insert(co);
    let bo_handle = liquid_world.add_boundary(Boundary::new(Vec::new()));
    coupling_manager.register_coupling(
        bo_handle,
        co_handle,
        CouplingMethod::DynamicContactSampling,
    );
    */

    /*
     * Set up the testbed.
     */
    testbed.set_body_wireframe(ground_handle, true);
    testbed.set_ground_handle(Some(ground_handle));

    testbed.set_world(
        mechanical_world,
        geometrical_world,
        bodies,
        colliders,
        joint_constraints,
        force_generators,
    );
    testbed.set_liquid_world(liquid_world, coupling_manager);
}

fn main() {
    let testbed = Testbed::from_builders(0, vec![("Boxes", init_world)]);
    testbed.run()
}
