use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use na::{self, RealField, Unit};

use crate::geometry::{ContactManager, ParticlesContacts};
use crate::kernel::Kernel;
use crate::math::{Vector, DIM, SPATIAL_DIM};
use crate::object::{Boundary, Fluid};

// http://peridynamics.com/publications/2014-He-RSS.pdf
pub struct He2014SurfaceTension<N: RealField> {
    gradcs: Vec<Vec<N>>,
    colors: Vec<Vec<N>>,
}

impl<N: RealField> He2014SurfaceTension<N> {
    pub fn new() -> Self {
        Self {
            colors: Vec::new(),
            gradcs: Vec::new(),
        }
    }

    /// Initialize this solver with the given fluids.
    pub fn init_with_fluids(&mut self, fluids: &[Fluid<N>]) {
        self.gradcs.resize(fluids.len(), Vec::new());
        self.colors.resize(fluids.len(), Vec::new());

        for (fluid, gradcs, colors) in itertools::multizip((
            fluids.iter(),
            self.gradcs.iter_mut(),
            self.colors.iter_mut(),
        )) {
            gradcs.resize(fluid.num_particles(), N::zero());
            colors.resize(fluid.num_particles(), N::zero());
        }
    }

    fn compute_colors(
        &mut self,
        kernel_radius: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        densities: &[Vec<N>],
    ) {
        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let colors_i = &mut self.colors[fluid_id];
            let fluid_i = &fluids[fluid_id];
            let densities_i = &densities[fluid_id];

            par_iter_mut!(colors_i)
                .enumerate()
                .for_each(|(i, color_i)| {
                    let mut color = N::zero();

                    for c in fluid_fluid_contacts.particle_contacts(i) {
                        if c.j_model == fluid_id {
                            color += c.weight * fluid_i.particle_mass(c.j) / densities_i[c.j];
                        }
                    }

                    *color_i = color;
                })
        }
    }

    fn compute_gradc(
        &mut self,
        kernel_radius: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        densities: &[Vec<N>],
    ) {
        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let gradcs_i = &mut self.gradcs[fluid_id];
            let fluid_i = &fluids[fluid_id];
            let densities_i = &densities[fluid_id];
            let colors_i = &self.colors[fluid_id];

            par_iter_mut!(gradcs_i)
                .enumerate()
                .for_each(|(i, gradc_i)| {
                    let mut gradc = Vector::zeros();
                    let mut denom = N::zero();

                    for c in fluid_fluid_contacts.particle_contacts(i) {
                        if c.j_model == fluid_id {
                            gradc += c.gradient * colors_i[c.j] * fluid_i.particle_mass(c.j)
                                / densities_i[c.j];
                        }
                    }

                    *gradc_i = (gradc / colors_i[i]).norm_squared();
                })
        }
    }

    pub fn solve(
        &mut self,
        dt: N,
        inv_dt: N,
        kernel_radius: N,
        contact_manager: &mut ContactManager<N>,
        fluids: &mut [Fluid<N>],
        densities: &[Vec<N>],
        velocity_changes: &mut [Vec<Vector<N>>],
    ) {
        let _2: N = na::convert(2.0f64);

        self.compute_colors(
            kernel_radius,
            &contact_manager.fluid_fluid_contacts,
            fluids,
            densities,
        );

        self.compute_gradc(
            kernel_radius,
            &contact_manager.fluid_fluid_contacts,
            fluids,
            densities,
        );

        // Compute and apply forces.
        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &contact_manager.fluid_fluid_contacts[fluid_id];
            let gradcs_i = &self.gradcs[fluid_id];
            let fluid_i = &fluids[fluid_id];
            let densities_i = &densities[fluid_id];
            let velocity_changes_i = &mut velocity_changes[fluid_id];
            let gradc_i = &self.gradcs[fluid_id];

            par_iter_mut!(velocity_changes_i)
                .enumerate()
                .for_each(|(i, velocity_change_i)| {
                    let mi = fluid_i.particle_mass(i);

                    for c in fluid_fluid_contacts.particle_contacts(i) {
                        if c.j_model == fluid_id {
                            let mj = fluid_i.particle_mass(c.j);
                            let gradsum = gradcs_i[c.i] + gradcs_i[c.j];
                            let f = c.gradient
                                * (mi / densities_i[c.i] * mj / densities_i[c.j] * gradsum / _2);
                            *velocity_change_i += f * (fluid_i.surface_tension / _2 * dt / mi);
                        }
                    }
                })
        }
    }
}
