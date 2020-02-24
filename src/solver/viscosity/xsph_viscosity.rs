#[cfg(feature = "parallel")]
use rayon::prelude::*;

use na::{self, RealField};

use crate::geometry::ParticlesContacts;

use crate::math::Vector;
use crate::object::Fluid;
use crate::solver::NonPressureForce;

#[derive(Clone)]
pub struct XSPHViscosity<N: RealField> {
    viscosity_coefficient: N,
}

impl<N: RealField> XSPHViscosity<N> {
    pub fn new(viscosity_coefficient: N) -> Self {
        Self {
            viscosity_coefficient,
        }
    }
}

impl<N: RealField> NonPressureForce<N> for XSPHViscosity<N> {
    fn solve(
        &mut self,
        _dt: N,
        _kernel_radius: N,
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid: &Fluid<N>,
        densities: &[N],
        velocity_changes: &mut [Vector<N>],
    ) {
        let viscosity_coefficient = self.viscosity_coefficient;

        par_iter_mut!(velocity_changes)
            .enumerate()
            .for_each(|(i, velocity_change)| {
                let mut added_vel = Vector::zeros();
                let vi = fluid.velocities[i];

                for c in fluid_fluid_contacts
                    .particle_contacts(i)
                    .read()
                    .unwrap()
                    .iter()
                {
                    if c.i_model == c.j_model {
                        added_vel += (fluid.velocities[c.j] - vi)
                            * (c.weight * fluid.particle_mass(c.j) / densities[c.j]);
                    }
                }

                *velocity_change += added_vel * viscosity_coefficient;
            })
    }

    fn apply_permutation(&mut self, _: &[usize]) {}
}
