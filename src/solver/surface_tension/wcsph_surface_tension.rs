#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::geometry::ParticlesContacts;

use crate::math::Real;
use crate::object::{Boundary, Fluid};
use crate::solver::NonPressureForce;
use crate::TimestepManager;

// Surface tension of water: 0.01
// Stable values of surface tension: up to 3.4
// From https://cg.informatik.uni-freiburg.de/publications/2007_SCA_SPH.pdf
/// Surface tension method introduced by the WCSPH method.
#[derive(Clone)]
pub struct WCSPHSurfaceTension {
    fluid_tension_coefficient: Real,
    boundary_tension_coefficient: Real,
}

impl WCSPHSurfaceTension {
    /// Initializes a surface tension with the given surface tension coefficient and boundary adhesion coefficients.
    pub fn new(fluid_tension_coefficient: Real, boundary_tension_coefficient: Real) -> Self {
        Self {
            fluid_tension_coefficient,
            boundary_tension_coefficient,
        }
    }
}

impl NonPressureForce for WCSPHSurfaceTension {
    fn solve(
        &mut self,
        _timestep: &TimestepManager,
        _kernel_radius: Real,
        fluid_fluid_contacts: &ParticlesContacts,
        _fluid_boundaries_contacts: &ParticlesContacts,
        fluid: &mut Fluid,
        boundaries: &[Boundary],
        _densities: &[Real],
    ) {
        let fluid_tension_coefficient = self.fluid_tension_coefficient;
        let boundary_tension_coefficient = self.boundary_tension_coefficient;
        let positions = &fluid.positions;
        let volumes = &fluid.volumes;
        let density0 = fluid.density0;

        par_iter_mut!(fluid.accelerations)
            .enumerate()
            .for_each(|(i, acceleration_i)| {
                if fluid_tension_coefficient != na::zero::<Real>() {
                    for c in fluid_fluid_contacts
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        if c.i_model == c.j_model {
                            let dpos = positions[c.i] - positions[c.j];
                            let cohesion_acc = dpos
                                * (-fluid_tension_coefficient * c.weight * volumes[c.j] * density0
                                    / (volumes[c.i] * density0));
                            *acceleration_i += cohesion_acc;
                        }
                    }
                }

                if boundary_tension_coefficient != na::zero::<Real>() {
                    for c in fluid_fluid_contacts
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        let dpos = positions[c.i] - boundaries[c.j_model].positions[c.j];
                        let mi = volumes[c.i] * density0;
                        let cohesion_force = dpos
                            * (boundary_tension_coefficient
                                * c.weight
                                * boundaries[c.j_model].volumes[c.j]
                                * density0);
                        *acceleration_i -= cohesion_force / mi;
                        boundaries[c.j_model].apply_force(c.j, cohesion_force);
                    }
                }
            })
    }

    fn apply_permutation(&mut self, _: &[usize]) {}
}
