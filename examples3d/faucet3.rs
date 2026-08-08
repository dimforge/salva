extern crate nalgebra as na;

use na::Vector3;
use rapier3d::dynamics::RigidBodyBuilder;
use rapier3d::geometry::ColliderBuilder;
use rapier3d::math::Vector;
use rapier3d::pipeline::PhysicsWorld;
use rapier_testbed3d::TestbedViewer;
use salva3d::integrations::rapier::{ColliderSampling, FluidsPipeline, FluidsTestbedPlugin};
use salva3d::object::interaction_groups::InteractionGroups;
use salva3d::object::{Boundary, Fluid};
use salva3d::solver::{Akinci2013SurfaceTension, XSPHViscosity};
use std::f32;

#[path = "./helper.rs"]
mod helper;

const PARTICLE_RADIUS: f32 = 0.025 / 2.0;
const SMOOTHING_FACTOR: f32 = 2.0;

pub async fn run(viewer: &mut TestbedViewer) -> anyhow::Result<()> {
    /*
     * World
     */
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::Y * -9.81;
    world.integration_parameters.dt = 1.0 / 200.0;

    let mut plugin = FluidsTestbedPlugin::new();
    let mut fluids_pipeline = FluidsPipeline::new(PARTICLE_RADIUS, SMOOTHING_FACTOR);

    let ground_rad = 0.15;

    // Initialize the fluid.
    let viscosity = XSPHViscosity::new(0.5, 0.0);
    let tension = Akinci2013SurfaceTension::new(1.0, 10.0);
    let mut fluid = Fluid::new(
        Vec::new(),
        PARTICLE_RADIUS,
        1000.0,
        InteractionGroups::default(),
    );
    fluid.nonpressure_forces.push(Box::new(viscosity));
    fluid.nonpressure_forces.push(Box::new(tension));
    let fluid_handle = fluids_pipeline.liquid_world.add_fluid(fluid);
    plugin.set_fluid_color(fluid_handle, Vector3::new(0.5, 1.0, 1.0));

    // Setup the ground.
    let ground_handle = world.bodies.insert(RigidBodyBuilder::fixed().build());
    let co = ColliderBuilder::ball(ground_rad).build();
    let ball_samples =
        salva3d::sampling::shape_surface_ray_sample(co.shape(), PARTICLE_RADIUS).unwrap();
    let co_handle = world
        .colliders
        .insert_with_parent(co, ground_handle, &mut world.bodies);
    let bo_handle = fluids_pipeline
        .liquid_world
        .add_boundary(Boundary::new(Vec::new(), InteractionGroups::default()));

    fluids_pipeline.coupling.register_coupling(
        bo_handle,
        co_handle,
        ColliderSampling::StaticSampling(ball_samples),
    );

    /*
     * Set up the viewer and run the simulation, generating new particles
     * every few timesteps (was a testbed callback).
     */
    plugin.set_pipeline(fluids_pipeline);
    viewer.set_world(&mut world);
    viewer.set_body_wireframe(ground_handle, true);
    viewer.look_at(Vector::new(1.5, 0.0, 1.5), Vector::ZERO);

    let mut time = 0.0f32;
    let mut last_t = 0.0f32;

    while viewer.render_frame(&mut world).await {
        plugin.update_from_settings(viewer.example_settings_mut());
        plugin.draw(viewer);

        if viewer.simulating() {
            world.step();
            plugin.step(&mut world);
            time += world.integration_parameters.dt;

            let fluid = plugin
                .pipeline_mut()
                .liquid_world
                .fluids_mut()
                .get_mut(fluid_handle)
                .unwrap();

            for i in 0..fluid.num_particles() {
                if fluid.positions[i].y < -2.0 {
                    fluid.delete_particle_at_next_timestep(i);
                }
            }

            if time - last_t >= 0.06 {
                last_t = time;
                let height = 0.6;
                let diam = PARTICLE_RADIUS * 2.0;
                let nparticles = 10;
                let mut particles = Vec::new();
                let mut velocities = Vec::new();
                let shift = -nparticles as f32 * PARTICLE_RADIUS;
                let vel = 0.0;

                for i in 0..nparticles {
                    for j in 0..nparticles {
                        let pos = Vector3::new(i as f32 * diam, height, j as f32 * diam);
                        particles.push(pos + Vector3::new(shift, 0.0, shift));
                        velocities.push(Vector3::y() * vel);
                    }
                }

                fluid.add_particles(&particles, Some(&velocities));
            }
        }
    }

    Ok(())
}
