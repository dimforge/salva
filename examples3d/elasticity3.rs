extern crate nalgebra as na;

use na::{Isometry3, Point3, Vector3};
use ncollide3d::shape::{Capsule, Cuboid, ShapeHandle};
use nphysics3d::force_generator::DefaultForceGeneratorSet;
use nphysics3d::joint::DefaultJointConstraintSet;
use nphysics3d::object::{
    BodyPartHandle, ColliderDesc, DefaultBodySet, DefaultColliderSet, Ground, RigidBodyDesc,
};
use nphysics3d::world::{DefaultGeometricalWorld, DefaultMechanicalWorld};
use nphysics_testbed3d::objects::FluidRenderingMode;
use nphysics_testbed3d::Testbed;
use salva3d::coupling::{ColliderCouplingSet, CouplingMethod};
use salva3d::object::{Boundary, Fluid};
use salva3d::solver::{
    Akinci2013SurfaceTension, Becker2009Elasticity, DFSPHSolver, DFSPHViscosity,
    He2014SurfaceTension, WCSPHSurfaceTension, XSPHViscosity,
};
use salva3d::LiquidWorld;
use std::f32;

#[path = "./helper.rs"]
mod helper;

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

    /*
     * Liquid world.
     */
    let particle_rad = 0.025;
    let solver: DFSPHSolver<f32> = DFSPHSolver::new();
    let mut liquid_world = LiquidWorld::new(solver, particle_rad, 2.0);
    let mut coupling_manager = ColliderCouplingSet::new();

    // Initialize the fluids and give them elasticity.
    let height = 0.4;
    let nparticles = 6;
    let nfluids = 2;
    for i in 0..nfluids {
        for j in 0..nfluids {
            for k in 0..nfluids {
                let mut shift = Vector3::new(i as f32, j as f32, k as f32);
                shift.apply(|e| e * particle_rad * nparticles as f32 * 2.0);
                let elasticity: Becker2009Elasticity<_> =
                    Becker2009Elasticity::new(100_000.0, 0.3, false);
                let viscosity = DFSPHViscosity::new(0.1);
                let mut fluid =
                    helper::cube_fluid(nparticles, nparticles, nparticles, particle_rad, 1000.0);
                fluid.transform_by(&Isometry3::translation(shift.x, shift.y + height, shift.z));
                fluid.nonpressure_forces.push(Box::new(elasticity));
                fluid.nonpressure_forces.push(Box::new(viscosity));
                let fluid_handle = liquid_world.add_fluid(fluid);
            }
        }
    }

    // Setup the ground.
    let ground_handle = bodies.insert(Ground::new());

    let ground_thickness = 0.2;
    let ground_half_width = 1.5;

    let ground_shape = ShapeHandle::new(Cuboid::new(Vector3::new(
        ground_half_width,
        ground_thickness,
        ground_half_width,
    )));

    let co = ColliderDesc::new(ground_shape).build(BodyPartHandle(ground_handle, 0));
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
    //    testbed.enable_boundary_particles_rendering(true);
    testbed.look_at(Point3::new(1.0, 1.0, 1.0), Point3::origin());
}

fn main() {
    let testbed = Testbed::from_builders(0, vec![("Boxes", init_world)]);
    testbed.run()
}
