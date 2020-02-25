extern crate nalgebra as na;

use na::{Isometry2, Point2, Point3, Unit, Vector2};
use ncollide2d::shape::{Capsule, Cuboid, ShapeHandle};
use nphysics2d::force_generator::DefaultForceGeneratorSet;
use nphysics2d::joint::DefaultJointConstraintSet;
use nphysics2d::object::{
    BodyPartHandle, ColliderDesc, DefaultBodySet, DefaultColliderSet, Ground, RigidBodyDesc,
};
use nphysics2d::world::{DefaultGeometricalWorld, DefaultMechanicalWorld};
use nphysics_testbed2d::objects::FluidRenderingMode;
use nphysics_testbed2d::Testbed;
use salva2d::coupling::{ColliderCouplingSet, CouplingMethod};
use salva2d::object::{Boundary, Fluid};
use salva2d::solver::{
    Akinci2013SurfaceTension, ArtificialViscosity, Becker2009Elasticity, DFSPHSolver,
    DFSPHViscosity, He2014SurfaceTension, NonPressureForce, WCSPHSurfaceTension, XSPHViscosity,
};
use salva2d::LiquidWorld;
use std::f32;

#[path = "./helper.rs"]
mod helper;

pub fn init_world(testbed: &mut Testbed) {
    /*
     * World
     */
    let mechanical_world = DefaultMechanicalWorld::new(Vector2::zeros());
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
    let solver: DFSPHSolver<f32> = DFSPHSolver::new();
    let mut liquid_world = LiquidWorld::new(solver, particle_rad, 2.0);
    let mut coupling_manager = ColliderCouplingSet::new();

    // Liquid.
    let nparticles = 30;
    let custom_force1 = CustomForceField {
        origin: Point2::new(1.0, 0.0),
    };
    let custom_force2 = CustomForceField {
        origin: Point2::new(-1.0, 0.0),
    };
    let mut fluid = helper::cube_fluid(nparticles, nparticles, particle_rad, 1000.0);
    fluid.nonpressure_forces.push(Box::new(custom_force1));
    fluid.nonpressure_forces.push(Box::new(custom_force2));
    let fluid_handle = liquid_world.add_fluid(fluid);
    testbed.set_fluid_color(fluid_handle, Point3::new(0.8, 0.7, 1.0));

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
    testbed.set_fluid_rendering_mode(FluidRenderingMode::VelocityColor { min: 0.0, max: 5.0 });
    testbed.mechanical_world_mut().set_timestep(1.0 / 200.0);
    testbed.look_at(Point2::origin(), 300.0);
}

fn main() {
    let testbed = Testbed::from_builders(0, vec![("Boxes", init_world)]);
    testbed.run()
}

struct CustomForceField {
    origin: Point2<f32>,
}

impl NonPressureForce<f32> for CustomForceField {
    fn solve(
        &mut self,
        timestep: &salva2d::TimestepManager<f32>,
        kernel_radius: f32,
        fluid_fluid_contacts: &salva2d::geometry::ParticlesContacts<f32>,
        fluid_boundaries_contacts: &salva2d::geometry::ParticlesContacts<f32>,
        fluid: &mut Fluid<f32>,
        boundaries: &[Boundary<f32>],
        densities: &[f32],
    ) {
        for (pos, acc) in fluid.positions.iter().zip(fluid.accelerations.iter_mut()) {
            if let Some((dir, dist)) = Unit::try_new_and_get(self.origin - pos, 0.1) {
                *acc += *dir / dist;
            }
        }
    }

    fn apply_permutation(&mut self, permutation: &[usize]) {}
}
