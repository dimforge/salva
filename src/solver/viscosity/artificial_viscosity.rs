#[cfg(feature = "parallel")]
use rayon::prelude::*;

use na::{self, RealField};

use crate::geometry::ParticlesContacts;

use crate::math::{Real, Vector};
use crate::object::{Boundary, Fluid};
use crate::solver::NonPressureForce;
use crate::TimestepManager;

// See http://www.astro.lu.se/~david/teaching/SPH/notes/annurev.aa.30.090192.pdf
/// Implements artificial viscosity.
#[derive(Clone)]
pub struct ArtificialViscosity {
    /// The coefficient of the linear part of the viscosity.
    pub alpha: Real,
    /// The coefficient of the quadratic part of the viscosity.
    pub beta: Real,
    /// The speed of sound.
    pub speed_of_sound: Real,
    /// The fluid viscosity coefficient.
    pub fluid_viscosity_coefficient: Real,
    /// The viscosity coefficient when interacting with boundaries.
    pub boundary_viscosity_coefficient: Real,
}

impl ArtificialViscosity<Real> {
    /// Initializes the artificial viscosity with the given viscosity coefficients.
    pub fn new(fluid_viscosity_coefficient: Real, boundary_viscosity_coefficient: Real) -> Self {
        Self {
            alpha: Real::one(),
            beta: na::convert(0.0),
            speed_of_sound: na::convert(10.0),
            fluid_viscosity_coefficient,
            boundary_viscosity_coefficient,
        }
    }
}

impl NonPressureForce<Real> for ArtificialViscosity<Real> {
    fn solve(
        &mut self,
        _timestep: &TimestepManager,
        kernel_radius: Real,
        fluid_fluid_contacts: &ParticlesContacts,
        fluid_boundaries_contacts: &ParticlesContacts,
        fluid: &mut Fluid,
        boundaries: &[Boundary],
        densities: &[N],
    ) {
        let fluid_viscosity_coefficient = self.fluid_viscosity_coefficient;
        let boundary_viscosity_coefficient = self.boundary_viscosity_coefficient;
        let speed_of_sound = self.speed_of_sound;
        let alpha = self.alpha;
        let beta = self.beta;
        let density0 = fluid.density0;
        let volumes = &fluid.volumes;
        let positions = &fluid.positions;
        let velocities = &fluid.velocities;
        let _0_5: Real = na::convert(0.5);

        par_iter_mut!(fluid.accelerations)
            .enumerate()
            .for_each(|(i, acceleration)| {
                let mut fluid_acc = Vector::zeros();
                let mut boundary_acc = Vector::zeros();

                if self.fluid_viscosity_coefficient != N::zero() {
                    for c in fluid_fluid_contacts
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        if c.i_model == c.j_model {
                            let r_ij = positions[c.i] - positions[c.j];
                            let v_ij = velocities[c.i] - velocities[c.j];
                            let vr = r_ij.dot(&v_ij);

                            if vr < N::zero() {
                                let density_average = (densities[c.i] + densities[c.j]) * _0_5;
                                let eta2 = kernel_radius * kernel_radius * na::convert(0.01);
                                let mu_ij = kernel_radius * vr / (r_ij.norm_squared() + eta2);

                                fluid_acc += c.gradient
                                    * (fluid_viscosity_coefficient
                                        * (speed_of_sound * alpha * mu_ij - beta * mu_ij * mu_ij)
                                        * (volumes[c.j] * density0 / density_average));
                            }
                        }
                    }
                }

                if self.boundary_viscosity_coefficient != N::zero() {
                    for c in fluid_boundaries_contacts
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        let r_ij = positions[c.i] - boundaries[c.j_model].positions[c.j];
                        let v_ij = velocities[c.i] - boundaries[c.j_model].velocities[c.j];
                        let vr = r_ij.dot(&v_ij);

                        if vr < N::zero() {
                            let density_average = densities[c.i];
                            let eta2 = kernel_radius * kernel_radius * na::convert(0.01);
                            let mu_ij = kernel_radius * vr / (r_ij.norm_squared() + eta2);

                            boundary_acc += c.gradient
                                * (boundary_viscosity_coefficient
                                    * (speed_of_sound * alpha * mu_ij - beta * mu_ij * mu_ij)
                                    * (boundaries[c.j_model].volumes[c.j] * density0
                                        / density_average));
                            let mi = volumes[c.i] * density0;
                            boundaries[c.j_model].apply_force(c.j, boundary_acc * -mi);
                        }
                    }
                }

                *acceleration += fluid_acc + boundary_acc;
            })
    }

    fn apply_permutation(&mut self, _: &[usize]) {}
}
