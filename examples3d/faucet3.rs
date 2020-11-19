extern crate nalgebra as na;

use na::{Point3, Vector3};
use rapier3d::dynamics::{JointSet, RigidBodyBuilder, RigidBodySet};
use rapier3d::geometry::{ColliderBuilder, ColliderSet};
use rapier_testbed3d::Testbed;
use salva3d::integrations::rapier::{ColliderSampling, FluidsPipeline, FluidsTestbedPlugin};
use salva3d::object::{Boundary, Fluid};
use salva3d::solver::{Akinci2013SurfaceTension, XSPHViscosity};
use std::f32;

#[path = "./helper.rs"]
mod helper;

const PARTICLE_RADIUS: f32 = 0.025 / 2.0;
const SMOOTHING_FACTOR: f32 = 2.0;

pub fn init_world(testbed: &mut Testbed) {
    /*
     * World
     */
    let mut plugin = FluidsTestbedPlugin::new();
    let gravity = Vector3::y() * -9.81;
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let joints = JointSet::new();
    let mut fluids_pipeline = FluidsPipeline::new(PARTICLE_RADIUS, SMOOTHING_FACTOR);

    let ground_rad = 0.15;

    // Initialize the fluid.
    let viscosity = XSPHViscosity::new(0.5, 0.0);
    let tension = Akinci2013SurfaceTension::new(1.0, 10.0);
    let mut fluid = Fluid::new(Vec::new(), PARTICLE_RADIUS, 1000.0);
    fluid.nonpressure_forces.push(Box::new(viscosity));
    fluid.nonpressure_forces.push(Box::new(tension));
    let fluid_handle = fluids_pipeline.liquid_world.add_fluid(fluid);
    plugin.set_fluid_color(fluid_handle, Point3::new(0.5, 1.0, 1.0));

    // Setup the ground.
    let ground_handle = bodies.insert(RigidBodyBuilder::new_static().build());
    let co = ColliderBuilder::ball(ground_rad).build();
    let ball_samples =
        salva3d::sampling::shape_surface_ray_sample(co.shape(), PARTICLE_RADIUS).unwrap();
    let co_handle = colliders.insert(co, ground_handle, &mut bodies);
    let bo_handle = fluids_pipeline
        .liquid_world
        .add_boundary(Boundary::new(Vec::new()));

    fluids_pipeline.coupling.register_coupling(
        bo_handle,
        co_handle,
        ColliderSampling::StaticSampling(ball_samples),
    );

    // Callback that will be executed on the main loop to generate new particles every second.
    let mut last_t = 0.0;

    plugin.add_callback(move |_, _, fluids_pipeline, t| {
        let fluid = fluids_pipeline
            .liquid_world
            .fluids_mut()
            .get_mut(fluid_handle)
            .unwrap();

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
        let diam = PARTICLE_RADIUS * 2.0;
        let nparticles = 10;
        let mut particles = Vec::new();
        let mut velocities = Vec::new();
        let shift = -nparticles as f32 * PARTICLE_RADIUS;
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
    plugin.set_pipeline(fluids_pipeline);
    testbed.add_plugin(plugin);
    testbed.set_body_wireframe(ground_handle, true);
    testbed.set_world_with_gravity(bodies, colliders, joints, gravity);
    testbed.integration_parameters_mut().set_dt(1.0 / 200.0);
    //    testbed.enable_boundary_particles_rendering(true);
    testbed.look_at(Point3::new(1.5, 0.0, 1.5), Point3::new(0.0, 0.0, 0.0));
}

fn main() {
    let testbed = Testbed::from_builders(0, vec![("Boxes", init_world)]);
    testbed.run()
}
