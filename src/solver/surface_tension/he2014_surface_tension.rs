#[cfg(feature = "parallel")]
use rayon::prelude::*;

use na::{self, RealField};

use crate::geometry::ParticlesContacts;

use crate::math::Vector;
use crate::object::{Boundary, Fluid};
use crate::solver::NonPressureForce;
use crate::TimestepManager;

// http://peridynamics.com/publications/2014-He-RSS.pdf
/// Surface tension method introduced by He et al. 2014
pub struct He2014SurfaceTension<N: RealField> {
    fluid_tension_coefficient: N,
    boundary_tension_coefficient: N,
    gradcs: Vec<N>,
    colors: Vec<N>,
}

impl<N: RealField> He2014SurfaceTension<N> {
    /// Initializes a surface tension with the given surface tension coefficient and boundary adhesion coefficients.
    pub fn new(fluid_tension_coefficient: N, boundary_tension_coefficient: N) -> Self {
        Self {
            fluid_tension_coefficient,
            boundary_tension_coefficient,
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
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid_boundary_contacts: &ParticlesContacts<N>,
        fluid: &Fluid<N>,
        boundaries: &[Boundary<N>],
        densities: &[N],
    ) {
        par_iter_mut!(self.colors)
            .enumerate()
            .for_each(|(i, color_i)| {
                let mut color = N::zero();

                for c in fluid_fluid_contacts
                    .particle_contacts(i)
                    .read()
                    .unwrap()
                    .iter()
                {
                    if c.i_model == c.j_model {
                        color += c.weight * fluid.particle_mass(c.j) / densities[c.j];
                    }
                }

                for c in fluid_boundary_contacts
                    .particle_contacts(i)
                    .read()
                    .unwrap()
                    .iter()
                {
                    color += c.weight * boundaries[c.j_model].volumes[c.j];
                }

                *color_i = color;
            })
    }

    fn compute_gradc(
        &mut self,
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid: &Fluid<N>,
        densities: &[N],
    ) {
        let colors = &self.colors;

        par_iter_mut!(self.gradcs)
            .enumerate()
            .for_each(|(i, gradc_i)| {
                let mut gradc = Vector::zeros();
                let _denom = N::zero();

                for c in fluid_fluid_contacts
                    .particle_contacts(i)
                    .read()
                    .unwrap()
                    .iter()
                {
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
        _timestep: &TimestepManager<N>,
        _kernel_radius: N,
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid_boundary_contacts: &ParticlesContacts<N>,
        fluid: &mut Fluid<N>,
        boundaries: &[Boundary<N>],
        densities: &[N],
    ) {
        self.init(fluid);
        let _2: N = na::convert(2.0f64);

        self.compute_colors(
            fluid_fluid_contacts,
            fluid_boundary_contacts,
            fluid,
            boundaries,
            densities,
        );
        self.compute_gradc(fluid_fluid_contacts, fluid, densities);

        // Compute and apply forces.
        let gradcs = &self.gradcs;
        let fluid_tension_coefficient = self.fluid_tension_coefficient;
        let boundary_tension_coefficient = self.boundary_tension_coefficient;
        let density0 = fluid.density0;
        let volumes = &fluid.volumes;

        par_iter_mut!(fluid.accelerations)
            .enumerate()
            .for_each(|(i, acceleration_i)| {
                let mi = volumes[i] * density0;

                if fluid_tension_coefficient != N::zero() {
                    for c in fluid_fluid_contacts
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        if c.i_model == c.j_model {
                            let mj = volumes[c.j] * density0;
                            let gradsum = gradcs[c.i] + gradcs[c.j];
                            let f = c.gradient
                                * (mi / densities[c.i] * mj / densities[c.j] * gradsum / _2);
                            *acceleration_i += f * (fluid_tension_coefficient / (_2 * mi));
                        }
                    }
                }

                if boundary_tension_coefficient != N::zero() {
                    for c in fluid_boundary_contacts
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        let mj = boundaries[c.j_model].volumes[c.j] * density0;
                        let gradsum = gradcs[c.i];
                        let f = c.gradient
                            * (mi / densities[c.i] * mj / density0
                                * gradsum
                                * boundary_tension_coefficient
                                * na::convert(0.25));
                        *acceleration_i += f / mi;

                        boundaries[c.j_model].apply_force(c.j, -f);
                    }
                }
            })
    }

    fn apply_permutation(&mut self, _: &[usize]) {}
}
