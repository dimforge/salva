#[cfg(feature = "parallel")]
use rayon::prelude::*;

use na::{self, RealField};

use crate::geometry::ParticlesContacts;

use crate::math::Vector;
use crate::object::Fluid;
use crate::solver::NonPressureForce;
use crate::TimestepManager;

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
        timestep: &TimestepManager<N>,
        _kernel_radius: N,
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid: &mut Fluid<N>,
        densities: &[N],
    ) {
        let viscosity_coefficient = self.viscosity_coefficient;
        let velocities = &fluid.velocities;
        let volumes = &fluid.volumes;
        let density0 = fluid.density0;

        par_iter_mut!(fluid.accelerations)
            .enumerate()
            .for_each(|(i, acceleration)| {
                let mut added_vel = Vector::zeros();
                let vi = velocities[i];

                for c in fluid_fluid_contacts
                    .particle_contacts(i)
                    .read()
                    .unwrap()
                    .iter()
                {
                    if c.i_model == c.j_model {
                        added_vel += (velocities[c.j] - vi)
                            * (c.weight * volumes[c.j] * density0 / densities[c.j]);
                    }
                }

                *acceleration += added_vel * (viscosity_coefficient * timestep.inv_dt());
            })
    }

    fn apply_permutation(&mut self, _: &[usize]) {}
}
