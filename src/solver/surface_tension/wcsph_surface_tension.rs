#[cfg(feature = "parallel")]
use rayon::prelude::*;

use na::{self, RealField};

use crate::geometry::ParticlesContacts;

use crate::math::Vector;
use crate::object::Fluid;
use crate::solver::NonPressureForce;

// Surface tension of water: 0.01
// Stable values of surface tension: up to 3.4
// From https://cg.informatik.uni-freiburg.de/publications/2007_SCA_SPH.pdf
#[derive(Clone)]
pub struct WCSPHSurfaceTension<N: RealField> {
    tension_coefficient: N,
}

impl<N: RealField> WCSPHSurfaceTension<N> {
    pub fn new(tension_coefficient: N) -> Self {
        Self {
            tension_coefficient,
        }
    }
}

impl<N: RealField> NonPressureForce<N> for WCSPHSurfaceTension<N> {
    fn solve(
        &mut self,
        dt: N,
        _kernel_radius: N,
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid: &Fluid<N>,
        _densities: &[N],
        velocity_changes: &mut [Vector<N>],
    ) {
        let tension_coefficient = self.tension_coefficient;

        par_iter_mut!(velocity_changes)
            .enumerate()
            .for_each(|(i, velocity_change_i)| {
                for c in fluid_fluid_contacts.particle_contacts(i) {
                    if c.i_model == c.j_model {
                        let dpos = fluid.positions[c.i] - fluid.positions[c.j];
                        let cohesion_acc = dpos
                            * (-tension_coefficient * c.weight * fluid.particle_mass(c.j)
                                / fluid.particle_mass(c.i));
                        *velocity_change_i += cohesion_acc * dt;
                    }
                }
            })
    }
}
