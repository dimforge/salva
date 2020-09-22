#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::geometry::ParticlesContacts;

use crate::math::{Real, Vector};
use crate::object::{Boundary, Fluid};
use crate::solver::NonPressureForce;
use crate::TimestepManager;

#[derive(Clone)]
/// Implements the viscosity model introduced with the XSPH method.
pub struct XSPHViscosity {
    /// The viscosity coefficient when interacting with boundaries.
    pub boundary_viscosity_coefficient: Real,
    /// The fluid viscosity coefficient.
    pub fluid_viscosity_coefficient: Real,
}

impl XSPHViscosity {
    /// Initializes the XSPH viscosity with the given viscosity coefficients.
    pub fn new(fluid_viscosity_coefficient: Real, boundary_viscosity_coefficient: Real) -> Self {
        Self {
            boundary_viscosity_coefficient,
            fluid_viscosity_coefficient,
        }
    }
}

impl NonPressureForce for XSPHViscosity {
    fn solve(
        &mut self,
        timestep: &TimestepManager,
        _kernel_radius: Real,
        fluid_fluid_contacts: &ParticlesContacts,
        fluid_boundaries_contacts: &ParticlesContacts,
        fluid: &mut Fluid,
        boundaries: &[Boundary],
        densities: &[Real],
    ) {
        let boundary_viscosity_coefficient = self.boundary_viscosity_coefficient;
        let fluid_viscosity_coefficient = self.fluid_viscosity_coefficient;
        let velocities = &fluid.velocities;
        let volumes = &fluid.volumes;
        let density0 = fluid.density0;

        par_iter_mut!(fluid.accelerations)
            .enumerate()
            .for_each(|(i, acceleration)| {
                let mut added_fluid_vel = Vector::zeros();
                let mut added_boundary_vel = Vector::zeros();
                let vi = velocities[i];

                if self.fluid_viscosity_coefficient != na::zero::<Real>() {
                    for c in fluid_fluid_contacts
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        if c.i_model == c.j_model {
                            added_fluid_vel += (velocities[c.j] - vi)
                                * (fluid_viscosity_coefficient
                                    * c.weight
                                    * volumes[c.j]
                                    * density0
                                    / densities[c.j]);
                        }
                    }
                }

                if self.boundary_viscosity_coefficient != na::zero::<Real>() {
                    for c in fluid_boundaries_contacts
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        let delta = (boundaries[c.j_model].velocities[c.j] - vi)
                            * (boundary_viscosity_coefficient
                                * c.weight
                                * boundaries[c.j_model].volumes[c.j]
                                * density0
                                / densities[c.i]);
                        added_boundary_vel += delta;

                        let mi = volumes[c.i] * density0;
                        boundaries[c.j_model].apply_force(c.j, delta * (-mi * timestep.inv_dt()));
                    }
                }

                *acceleration +=
                    added_fluid_vel * timestep.inv_dt() + added_boundary_vel * timestep.inv_dt();
            })
    }

    fn apply_permutation(&mut self, _: &[usize]) {}
}
