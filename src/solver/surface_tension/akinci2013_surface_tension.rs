use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use na::{self, RealField, Unit};

use crate::geometry::{ContactManager, ParticlesContacts};
use crate::kernel::Kernel;
use crate::math::{Vector, DIM, SPATIAL_DIM};
use crate::object::{Boundary, Fluid};

pub struct Akinci2013SurfaceTension<N: RealField> {
    normals: Vec<Vec<Vector<N>>>,
}

impl<N: RealField> Akinci2013SurfaceTension<N> {
    pub fn new() -> Self {
        Self {
            normals: Vec::new(),
        }
    }

    /// Initialize this solver with the given fluids.
    pub fn init_with_fluids(&mut self, fluids: &[Fluid<N>]) {
        self.normals.resize(fluids.len(), Vec::new());

        for (fluid, normals) in itertools::multizip((fluids.iter(), self.normals.iter_mut())) {
            normals.resize(fluid.num_particles(), Vector::zeros());
        }
    }

    fn compute_normals(
        &mut self,
        kernel_radius: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        densities: &[Vec<N>],
    ) {
        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let normals_i = &mut self.normals[fluid_id];
            let fluid_i = &fluids[fluid_id];
            let densities_i = &densities[fluid_id];

            par_iter_mut!(normals_i)
                .enumerate()
                .for_each(|(i, normal_i)| {
                    let mut normal = Vector::zeros();

                    for c in fluid_fluid_contacts.particle_contacts(i) {
                        if c.j_model == fluid_id {
                            normal += c
                                .gradient
//                                .try_normalize(na::convert(0.01))
//                                .unwrap_or(c.gradient)
                                * (fluid_i.particle_mass(c.j) / densities_i[c.j]);
                        }
                    }

                    *normal_i = normal * kernel_radius;
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

        self.compute_normals(
            kernel_radius,
            &contact_manager.fluid_fluid_contacts,
            fluids,
            densities,
        );

        // Compute and apply forces.
        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &contact_manager.fluid_fluid_contacts[fluid_id];
            let normals_i = &self.normals[fluid_id];
            let fluid_i = &fluids[fluid_id];
            let densities_i = &densities[fluid_id];
            let velocity_changes_i = &mut velocity_changes[fluid_id];

            par_iter_mut!(velocity_changes_i)
                .enumerate()
                .for_each(|(i, velocity_change_i)| {
                    let mi = fluid_i.particle_mass(i);

                    for c in fluid_fluid_contacts.particle_contacts(i) {
                        if c.j_model == fluid_id {
                            let dpos = fluid_i.positions[c.i] - fluid_i.positions[c.j];
                            let cohesion_vec = if let Some((dir, dist)) =
                                Unit::try_new_and_get(dpos, N::default_epsilon())
                            {
                                *dir * cohesion_kernel(dist, kernel_radius)
                            } else {
                                Vector::zeros()
                            };

                            let cohesion_acc = cohesion_vec
                                * (-fluid_i.surface_tension * fluid_i.particle_mass(c.j));
                            let curvature_acc =
                                (normals_i[c.i] - normals_i[c.j]) * -fluid_i.surface_tension;
                            let kij = _2 * fluid_i.density0 / (densities_i[c.i] + densities_i[c.j]);

                            //                            println!(
                            //                                "Surface tension velocity change. {}",
                            //                                (cohesion_acc + curvature_acc) * (kij * dt / mi)
                            //                            );

                            *velocity_change_i += (curvature_acc + cohesion_acc) * (kij * dt);
                        }
                    }
                })
        }
    }
}

fn cohesion_kernel<N: RealField>(r: N, h: N) -> N {
    //    #[cfg(feature = "dim3")]
    let normalizer = na::convert::<_, N>(32.0f64) / (N::pi() * h.powi(9));
    let _2: N = na::convert(2.0f64);

    let coeff = if r <= h / _2 {
        _2 * (h - r).powi(3) * r.powi(3) - h.powi(6) / na::convert(64.0f64)
    } else if r <= h {
        (h - r).powi(3) * r.powi(3)
    } else {
        N::zero()
    };

    normalizer * coeff * h
}
