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
use salva2d::solver::{Akinci2013SurfaceTension, ArtificialViscosity, IISPHSolver};
use salva2d::LiquidWorld;
use std::f32;

#[path = "./helper.rs"]
mod helper;

pub fn init_world(testbed: &mut Testbed) {
    /*
     * World
     */
    // We want to simulate a 1cm³ droplet. We use the spacial unit 1 = 1dm.
    // Therefore each particles must have a diameter of 0.005, and the gravity is -0.981 instead of -9.81.
    let mechanical_world = DefaultMechanicalWorld::new(Vector2::new(0.0, -0.981));
    let geometrical_world = DefaultGeometricalWorld::new();
    let mut bodies = DefaultBodySet::new();
    let mut colliders = DefaultColliderSet::new();
    let joint_constraints = DefaultJointConstraintSet::new();
    let force_generators = DefaultForceGeneratorSet::new();

    /*
     * Liquid world.
     */
    let particle_rad = 0.0025;
    let solver = IISPHSolver::<f32>::new();
    let mut liquid_world = LiquidWorld::new(solver, particle_rad, 2.0);
    let mut coupling_manager = ColliderCouplingSet::new();

    // Initialize the fluid and give it some surface tension. This will make the fluid take a spherical shape.
    let surface_tension = Akinci2013SurfaceTension::new(1.0, 0.0);
    let viscosity = ArtificialViscosity::new(0.01, 0.0);
    let mut fluid = helper::cube_fluid(20, 20, particle_rad, 1000.0);
    fluid.transform_by(&Isometry2::translation(0.0, 0.08));
    fluid.nonpressure_forces.push(Box::new(surface_tension));
    fluid.nonpressure_forces.push(Box::new(viscosity));
    let fluid_handle = liquid_world.add_fluid(fluid);
    testbed.set_fluid_color(fluid_handle, Point3::new(0.8, 0.7, 1.0));

    // Setup the ground.
    let ground_handle = bodies.insert(Ground::new());

    let ground_thickness = 0.02;
    let ground_half_width = 0.15;

    let ground_shape = ShapeHandle::new(Cuboid::new(Vector2::new(
        ground_half_width,
        ground_thickness,
    )));

    let samples =
        salva2d::sampling::shape_surface_ray_sample(&*ground_shape, particle_rad).unwrap();
    let co = ColliderDesc::new(ground_shape)
        .margin(0.0)
        .build(BodyPartHandle(ground_handle, 0));
    let co_handle = colliders.insert(co);
    let bo_handle = liquid_world.add_boundary(Boundary::new(Vec::new()));

    coupling_manager.register_coupling(
        bo_handle,
        co_handle,
        CouplingMethod::StaticSampling(samples),
    );

    /*
     * Set up the testbed.
     */
    testbed.set_world(
        mechanical_world,
        geometrical_world,
        bodies,
        colliders,
        joint_constraints,
        force_generators,
    );
    testbed.set_liquid_world(liquid_world, coupling_manager);
    testbed.mechanical_world_mut().set_timestep(1.0 / 200.0);
    testbed.set_fluid_rendering_mode(FluidRenderingMode::VelocityColor { min: 0.0, max: 5.0 });
    testbed.look_at(Point2::origin(), 2000.0);
    testbed.enable_boundary_particles_rendering(true);
}

fn main() {
    let testbed = Testbed::from_builders(0, vec![("Boxes", init_world)]);
    testbed.run()
}
