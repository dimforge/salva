extern crate nalgebra as na;

use na::{Isometry2, Point2, Point3, Vector2};
use ncollide2d::shape::{Cuboid, ShapeHandle};
use nphysics2d::force_generator::DefaultForceGeneratorSet;
use nphysics2d::joint::DefaultJointConstraintSet;
use nphysics2d::object::{
    BodyPartHandle, ColliderDesc, DefaultBodySet, DefaultColliderSet, Ground,
};
use nphysics2d::world::{DefaultGeometricalWorld, DefaultMechanicalWorld};
use nphysics_testbed2d::objects::FluidRenderingMode;
use nphysics_testbed2d::Testbed;
use salva2d::coupling::{ColliderCouplingSet, CouplingMethod};
use salva2d::object::Boundary;
use salva2d::solver::{Becker2009Elasticity, DFSPHSolver, XSPHViscosity};
use salva2d::LiquidWorld;
use std::f32;

#[path = "./helper.rs"]
mod helper;

pub fn init_world(testbed: &mut Testbed) {
    /*
     * World
     */
    let mechanical_world = DefaultMechanicalWorld::new(Vector2::new(0.0, -9.81));
    let geometrical_world = DefaultGeometricalWorld::new();
    let mut bodies = DefaultBodySet::new();
    let mut colliders = DefaultColliderSet::new();
    let joint_constraints = DefaultJointConstraintSet::new();
    let force_generators = DefaultForceGeneratorSet::new();

    let ground_thickness = 0.2;
    let ground_half_width = 3.0;

    /*
     * Liquid world.
     */
    let particle_rad = 0.025;
    let solver: DFSPHSolver<f32> = DFSPHSolver::new();
    let mut liquid_world = LiquidWorld::new(solver, particle_rad, 2.0);
    let mut coupling_manager = ColliderCouplingSet::new();

    // Initialize the fluids and give them elasticity.
    let height = 0.4;
    let nparticlesx = 25;
    let nparticlesy = 15;

    // First fluid with high young modulus.
    let elasticity = Becker2009Elasticity::<f32>::new(500_000.0, 0.3, true);
    let viscosity = XSPHViscosity::new(0.5, 1.0);
    let mut fluid = helper::cube_fluid(nparticlesx, nparticlesy, particle_rad, 1000.0);
    fluid.transform_by(&Isometry2::translation(
        0.0,
        ground_thickness + particle_rad * nparticlesy as f32 + height,
    ));
    fluid.nonpressure_forces.push(Box::new(elasticity));
    fluid.nonpressure_forces.push(Box::new(viscosity.clone()));
    let fluid_handle = liquid_world.add_fluid(fluid);
    testbed.set_fluid_color(fluid_handle, Point3::new(0.8, 0.7, 1.0));

    // Second fluid with smaller young modulus.
    let elasticity = Becker2009Elasticity::<f32>::new(100_000.0, 0.3, true);
    let mut fluid = helper::cube_fluid(nparticlesx, nparticlesy, particle_rad, 1000.0);
    fluid.transform_by(&Isometry2::translation(
        0.0,
        ground_thickness + particle_rad * nparticlesy as f32 * 4.0 + height,
    ));
    fluid.nonpressure_forces.push(Box::new(elasticity));
    fluid.nonpressure_forces.push(Box::new(viscosity));
    let fluid_handle = liquid_world.add_fluid(fluid);
    testbed.set_fluid_color(fluid_handle, Point3::new(0.6, 0.8, 0.5));

    // Setup the ground.
    let ground_handle = bodies.insert(Ground::new());

    let ground_shape = ShapeHandle::new(Cuboid::new(Vector2::new(
        ground_half_width,
        ground_thickness,
    )));

    let co = ColliderDesc::new(ground_shape)
        .margin(0.0)
        .build(BodyPartHandle(ground_handle, 0));
    let co_handle = colliders.insert(co);
    let bo_handle = liquid_world.add_boundary(Boundary::new(Vec::new()));

    coupling_manager.register_coupling(
        bo_handle,
        co_handle,
        CouplingMethod::DynamicContactSampling,
    );

    /*
     * Set up the testbed.
     */
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
    testbed.set_fluid_rendering_mode(FluidRenderingMode::VelocityColor { min: 0.0, max: 5.0 });
    testbed.mechanical_world_mut().set_timestep(1.0 / 200.0);
    testbed.look_at(Point2::new(0.0, -1.0), 300.0);
}

fn main() {
    let testbed = Testbed::from_builders(0, vec![("Boxes", init_world)]);
    testbed.run()
}
