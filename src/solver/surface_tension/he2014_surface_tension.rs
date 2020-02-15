use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use na::{self, RealField, Unit};

use crate::geometry::{ContactManager, ParticlesContacts};
use crate::kernel::Kernel;
use crate::math::{Vector, DIM, SPATIAL_DIM};
use crate::object::{Boundary, Fluid};
use crate::solver::NonPressureForce;

// http://peridynamics.com/publications/2014-He-RSS.pdf
pub struct He2014SurfaceTension<N: RealField> {
    tension_coefficient: N,
    gradcs: Vec<N>,
    colors: Vec<N>,
}

impl<N: RealField> He2014SurfaceTension<N> {
    pub fn new(tension_coefficient: N) -> Self {
        Self {
            tension_coefficient,
            colors: Vec::new(),
            gradcs: Vec::new(),
        }
    }

    fn init(&mut self, fluid: &Fluid<N>) {
        if self.gradcs.len() != fluid.num_particles() {
            self.gradcs.resize(fluid.num_particles(), N::zero());
            self.colors.resize(fluid.num_particles(), N::zero());
        }
    }

    fn compute_colors(
        &mut self,
        kernel_radius: N,
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid: &Fluid<N>,
        densities: &[N],
    ) {
        par_iter_mut!(self.colors)
            .enumerate()
            .for_each(|(i, color_i)| {
                let mut color = N::zero();

                for c in fluid_fluid_contacts.particle_contacts(i) {
                    if c.i_model == c.j_model {
                        color += c.weight * fluid.particle_mass(c.j) / densities[c.j];
                    }
                }

                *color_i = color;
            })
    }

    fn compute_gradc(
        &mut self,
        kernel_radius: N,
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid: &Fluid<N>,
        densities: &[N],
    ) {
        let colors = &self.colors;

        par_iter_mut!(self.gradcs)
            .enumerate()
            .for_each(|(i, gradc_i)| {
                let mut gradc = Vector::zeros();
                let mut denom = N::zero();

                for c in fluid_fluid_contacts.particle_contacts(i) {
                    if c.i_model == c.j_model {
                        gradc +=
                            c.gradient * colors[c.j] * fluid.particle_mass(c.j) / densities[c.j];
                    }
                }

                *gradc_i = (gradc / colors[i]).norm_squared();
            })
    }
}

impl<N: RealField> NonPressureForce<N> for He2014SurfaceTension<N> {
    fn solve(
        &mut self,
        dt: N,
        kernel_radius: N,
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid: &Fluid<N>,
        densities: &[N],
        velocity_changes: &mut [Vector<N>],
    ) {
        self.init(fluid);
        let _2: N = na::convert(2.0f64);

        self.compute_colors(kernel_radius, fluid_fluid_contacts, fluid, densities);
        self.compute_gradc(kernel_radius, fluid_fluid_contacts, fluid, densities);

        // Compute and apply forces.
        let gradcs = &self.gradcs;
        let tension_coefficient = self.tension_coefficient;

        par_iter_mut!(velocity_changes)
            .enumerate()
            .for_each(|(i, velocity_change_i)| {
                let mi = fluid.particle_mass(i);

                for c in fluid_fluid_contacts.particle_contacts(i) {
                    if c.i_model == c.j_model {
                        let mj = fluid.particle_mass(c.j);
                        let gradsum = gradcs[c.i] + gradcs[c.j];
                        let f =
                            c.gradient * (mi / densities[c.i] * mj / densities[c.j] * gradsum / _2);
                        *velocity_change_i += f * (tension_coefficient / _2 * dt / mi);
                    }
                }
            })
    }
}
