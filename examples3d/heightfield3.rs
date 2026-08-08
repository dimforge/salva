extern crate nalgebra as na;

use na::{Isometry3, Vector3};
use rapier3d::geometry::Array2;
use rapier3d::na::ComplexField;
use rapier3d::prelude::*;
use rapier_testbed3d::TestbedViewer;
use salva3d::integrations::rapier::ColliderSampling;
use salva3d::integrations::rapier::FluidsPipeline;
use salva3d::integrations::rapier::FluidsTestbedPlugin;
use salva3d::object::{interaction_groups::InteractionGroups, Boundary};
use salva3d::solver::ArtificialViscosity;

#[path = "./helper.rs"]
mod helper;

const PARTICLE_RADIUS: f32 = 0.15;
const SMOOTHING_FACTOR: f32 = 2.0;

pub async fn run(viewer: &mut TestbedViewer) -> anyhow::Result<()> {
    /*
     * World
     */
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / 200.0;

    /* Fluid */
    let mut fluids_pipeline = FluidsPipeline::new(PARTICLE_RADIUS, SMOOTHING_FACTOR);

    let nparticles = 15;
    let mut fluid = helper::cube_fluid(nparticles, nparticles, nparticles, PARTICLE_RADIUS, 1000.0);
    fluid.transform_by(&Isometry3::translation(
        0.0,
        1.0 + nparticles as f32 * PARTICLE_RADIUS * 2.,
        0.0,
    ));
    let viscosity = ArtificialViscosity::new(1.0, 0.0);
    fluid.nonpressure_forces.push(Box::new(viscosity));
    fluid.velocities = vec![-Vector3::y() * 10.; fluid.velocities.len()];
    let fluid_handle = fluids_pipeline.liquid_world.add_fluid(fluid);

    /*
     * Ground
     */
    let ground_size = Vector::new(12.0, 1.0, 12.0);
    let nsubdivs = 40;

    let heights = Array2::new(
        nsubdivs + 1,
        nsubdivs + 1,
        (0..=nsubdivs)
            .flat_map(|j| {
                (0..=nsubdivs).map(move |i| {
                    if i == 0 || i == nsubdivs || j == 0 || j == nsubdivs {
                        3.0
                    } else {
                        let x = i as f32 * ground_size.x / (nsubdivs as f32);
                        let z = j as f32 * ground_size.z / (nsubdivs as f32);

                        // NOTE: make sure we use the sin/cos from simba to ensure
                        // cross-platform determinism of the example when the
                        // enhanced_determinism feature is enabled.
                        ComplexField::sin(x) + ComplexField::cos(z)
                    }
                })
            })
            .collect(),
    );

    let rigid_body = RigidBodyBuilder::fixed().build();
    let handle = world.bodies.insert(rigid_body);
    let ground_collider = ColliderBuilder::heightfield(heights.clone(), ground_size).build();
    let ground_handle =
        world
            .colliders
            .insert_with_parent(ground_collider.clone(), handle, &mut world.bodies);

    let samples =
        salva3d::sampling::shape_surface_ray_sample(ground_collider.shape(), PARTICLE_RADIUS / 1.5)
            .unwrap();

    let bo_handle = fluids_pipeline
        .liquid_world
        .add_boundary(Boundary::new(Vec::new(), InteractionGroups::default()));

    fluids_pipeline.coupling.register_coupling(
        bo_handle,
        ground_handle,
        ColliderSampling::StaticSampling(samples),
    );

    /*
     * Set up the viewer and run the simulation.
     */
    let mut plugin = FluidsTestbedPlugin::new();
    plugin.set_pipeline(fluids_pipeline);
    plugin.set_fluid_color(fluid_handle, Vector3::new(0.8, 0.7, 1.0));
    // plugin.render_boundary_particles = true;

    viewer.set_world(&mut world);
    viewer.set_body_wireframe(handle, true);
    viewer.look_at(Vector::new(100.0, 100.0, 100.0), Vector::ZERO);

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
