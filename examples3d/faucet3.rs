extern crate nalgebra as na;

use na::{Isometry3, Point3, Vector3};
use ncollide3d::shape::{Ball, Cuboid, ShapeHandle};
use nphysics3d::force_generator::DefaultForceGeneratorSet;
use nphysics3d::joint::DefaultJointConstraintSet;
use nphysics3d::object::{
    BodyPartHandle, ColliderDesc, DefaultBodySet, DefaultColliderSet, Ground,
};
use nphysics3d::world::{DefaultGeometricalWorld, DefaultMechanicalWorld};
use nphysics_testbed3d::objects::FluidRenderingMode;
use nphysics_testbed3d::Testbed;
use salva3d::coupling::{ColliderCouplingSet, CouplingMethod};
use salva3d::object::{Boundary, Fluid};
use salva3d::solver::{Akinci2013SurfaceTension, Becker2009Elasticity, DFSPHSolver, XSPHViscosity};
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

    let ground_rad = 0.15;

    /*
     * Liquid world.
     */
    let particle_rad = 0.025 / 2.0;
    let solver: DFSPHSolver<f32> = DFSPHSolver::new();
    let mut liquid_world = LiquidWorld::new(solver, particle_rad, 2.0);
    let mut coupling_manager = ColliderCouplingSet::new();

    // Initialize the fluid.
    let viscosity = XSPHViscosity::new(0.5, 0.0);
    let tension = Akinci2013SurfaceTension::new(1.0, 10.0);
    let mut fluid = Fluid::new(Vec::new(), particle_rad, 1000.0);
    fluid.nonpressure_forces.push(Box::new(viscosity));
    fluid.nonpressure_forces.push(Box::new(tension));
    let fluid_handle = liquid_world.add_fluid(fluid);
    testbed.set_fluid_color(fluid_handle, Point3::new(0.5, 1.0, 1.0));

    // Setup the ground.
    let ground_handle = bodies.insert(Ground::new());
    let ground_shape = ShapeHandle::new(Ball::new(ground_rad));
    let ball_samples =
        salva3d::sampling::shape_surface_ray_sample(&*ground_shape, particle_rad).unwrap();

    let co = ColliderDesc::new(ground_shape)
        .margin(0.0)
        .build(BodyPartHandle(ground_handle, 0));
    let co_handle = colliders.insert(co);
    let bo_handle = liquid_world.add_boundary(Boundary::new(Vec::new()));

    coupling_manager.register_coupling(
        bo_handle,
        co_handle,
        CouplingMethod::StaticSampling(ball_samples),
    );

    // Callback that will be executed on the main loop to generate new particles every second.
    let mut last_t = 0.0;

    testbed.add_callback_with_fluids(move |liquid_world, _, _, _, _, _, _, t| {
        let fluid = liquid_world.fluids_mut().get_mut(fluid_handle).unwrap();

        for i in 0..fluid.num_particles() {
            if fluid.positions[i].y < -2.0 {
                fluid.delete_particle_at_next_timestep(i);
            }
        }

        if t - last_t < 0.06 {
            return;
        }

        last_t = t;
        let height = 0.6;
        let diam = particle_rad * 2.0;
        let nparticles = 10;
        let mut particles = Vec::new();
        let mut velocities = Vec::new();
        let shift = -nparticles as f32 * particle_rad;
        let vel = 0.0;

        for i in 0..nparticles {
            for j in 0..nparticles {
                let pos = Point3::new(i as f32 * diam, height, j as f32 * diam);
                particles.push(pos + Vector3::new(shift, 0.0, shift));
                velocities.push(Vector3::y() * vel);
            }
        }

        fluid.add_particles(&particles, Some(&velocities));
    });

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
    testbed.set_fluid_rendering_mode(FluidRenderingMode::StaticColor);
    testbed.mechanical_world_mut().set_timestep(1.0 / 200.0);
    //    testbed.enable_boundary_particles_rendering(true);
    testbed.look_at(Point3::new(1.5, 0.0, 1.5), Point3::new(0.0, 0.0, 0.0));
}

fn main() {
    let testbed = Testbed::from_builders(0, vec![("Boxes", init_world)]);
    testbed.run()
}
