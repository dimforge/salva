use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use na::{self, RealField, Unit};

use crate::geometry::{ContactManager, ParticlesContacts};
use crate::kernel::Kernel;
use crate::math::{Vector, DIM, SPATIAL_DIM};
use crate::object::{Boundary, Fluid};
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
        kernel_radius: N,
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid: &Fluid<N>,
        densities: &[N],
        velocity_changes: &mut [Vector<N>],
    ) {
        let tension_coefficient = self.tension_coefficient;

        par_iter_mut!(velocity_changes)
            .enumerate()
            .for_each(|(i, velocity_change_i)| {
                for c in fluid_fluid_contacts.particle_contacts(i) {
                    if c.i_model == c.j_model {
                        let dpos = fluid.positions[c.i] - fluid.positions[c.j];
                        let cohesion_acc =
                            dpos * (-tension_coefficient * fluid.particle_mass(c.j) * c.weight);
                        *velocity_change_i += cohesion_acc * dt;
                    }
                }
            })
    }
}
