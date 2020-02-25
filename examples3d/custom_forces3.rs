extern crate nalgebra as na;

use na::{Point3, Unit, Vector3};
use nphysics3d::force_generator::DefaultForceGeneratorSet;
use nphysics3d::joint::DefaultJointConstraintSet;
use nphysics3d::object::{DefaultBodySet, DefaultColliderSet};
use nphysics3d::world::{DefaultGeometricalWorld, DefaultMechanicalWorld};
use nphysics_testbed3d::objects::FluidRenderingMode;
use nphysics_testbed3d::Testbed;
use salva3d::coupling::ColliderCouplingSet;
use salva3d::object::{Boundary, Fluid};
use salva3d::solver::{DFSPHSolver, NonPressureForce};
use salva3d::LiquidWorld;
use std::f32;

#[path = "./helper.rs"]
mod helper;

pub fn init_world(testbed: &mut Testbed) {
    /*
     * World
     */
    let mechanical_world = DefaultMechanicalWorld::new(Vector3::zeros());
    let geometrical_world = DefaultGeometricalWorld::new();
    let bodies = DefaultBodySet::new();
    let colliders = DefaultColliderSet::new();
    let joint_constraints = DefaultJointConstraintSet::new();
    let force_generators = DefaultForceGeneratorSet::new();

    /*
     * Liquid world.
     */
    let particle_rad = 0.025;
    let solver: DFSPHSolver<f32> = DFSPHSolver::new();
    let mut liquid_world = LiquidWorld::new(solver, particle_rad, 2.0);
    let coupling_manager = ColliderCouplingSet::new();

    // Liquid.
    let nparticles = 10;
    let custom_force1 = CustomForceField {
        origin: Point3::new(1.0, 0.0, 0.0),
    };
    let custom_force2 = CustomForceField {
        origin: Point3::new(-1.0, 0.0, 0.0),
    };
    let mut fluid = helper::cube_fluid(nparticles, nparticles, nparticles, particle_rad, 1000.0);
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
    testbed.look_at(Point3::new(3.0, 3.0, 3.0), Point3::origin());
}

fn main() {
    let testbed = Testbed::from_builders(0, vec![("Boxes", init_world)]);
    testbed.run()
}

struct CustomForceField {
    origin: Point3<f32>,
}

impl NonPressureForce<f32> for CustomForceField {
    fn solve(
        &mut self,
        _timestep: &salva3d::TimestepManager<f32>,
        _kernel_radius: f32,
        _fluid_fluid_contacts: &salva3d::geometry::ParticlesContacts<f32>,
        _fluid_boundaries_contacts: &salva3d::geometry::ParticlesContacts<f32>,
        fluid: &mut Fluid<f32>,
        _boundaries: &[Boundary<f32>],
        _densities: &[f32],
    ) {
        for (pos, acc) in fluid.positions.iter().zip(fluid.accelerations.iter_mut()) {
            if let Some((dir, dist)) = Unit::try_new_and_get(self.origin - pos, 0.1) {
                *acc += *dir / dist;
            }
        }
    }

    fn apply_permutation(&mut self, _permutation: &[usize]) {}
}
