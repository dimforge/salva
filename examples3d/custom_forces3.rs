extern crate nalgebra as na;

use na::{Unit, Vector3};
use rapier3d::math::Vector;
use rapier3d::pipeline::PhysicsWorld;
use rapier_testbed3d::TestbedViewer;
use salva3d::integrations::rapier::{FluidsPipeline, FluidsRenderingMode, FluidsTestbedPlugin};
use salva3d::object::{Boundary, Fluid};
use salva3d::solver::NonPressureForce;
use std::f32;

#[path = "./helper.rs"]
mod helper;

const PARTICLE_RADIUS: f32 = 0.025;
const SMOOTHING_FACTOR: f32 = 2.0;

pub async fn run(viewer: &mut TestbedViewer) -> anyhow::Result<()> {
    /*
     * World
     */
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    world.integration_parameters.dt = 1.0 / 200.0;

    let mut plugin = FluidsTestbedPlugin::new();
    let mut fluids_pipeline = FluidsPipeline::new(PARTICLE_RADIUS, SMOOTHING_FACTOR);

    // fluids.
    let nparticles = 10;
    let custom_force1 = CustomForceField {
        origin: Vector3::new(1.0, 0.0, 0.0),
    };
    let custom_force2 = CustomForceField {
        origin: Vector3::new(-1.0, 0.0, 0.0),
    };
    let mut fluid = helper::cube_fluid(nparticles, nparticles, nparticles, PARTICLE_RADIUS, 1000.0);
    fluid.nonpressure_forces.push(Box::new(custom_force1));
    fluid.nonpressure_forces.push(Box::new(custom_force2));
    let fluid_handle = fluids_pipeline.liquid_world.add_fluid(fluid);
    plugin.set_fluid_color(fluid_handle, Vector3::new(0.8, 0.7, 1.0));

    /*
     * Set up the viewer and run the simulation.
     */
    plugin.set_pipeline(fluids_pipeline);
    plugin.set_fluid_rendering_mode(FluidsRenderingMode::VelocityColor { min: 0.0, max: 5.0 });
    viewer.set_world(&mut world);
    viewer.look_at(Vector::new(3.0, 3.0, 3.0), Vector::ZERO);

    while viewer.render_frame(&mut world).await {
        plugin.update_from_settings(viewer.example_settings_mut());
        plugin.draw(viewer);

        if viewer.simulating() {
            world.step();
            plugin.step(&mut world);
        }
    }

    Ok(())
}

struct CustomForceField {
    origin: Vector3<f32>,
}

impl NonPressureForce for CustomForceField {
    fn solve(
        &mut self,
        _timestep: &salva3d::TimestepManager,
        _kernel_radius: f32,
        _fluid_fluid_contacts: &salva3d::geometry::ParticlesContacts,
        _fluid_boundaries_contacts: &salva3d::geometry::ParticlesContacts,
        fluid: &mut Fluid,
        _boundaries: &[Boundary],
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
