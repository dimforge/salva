use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use na::{self, RealField};

use crate::geometry::{ContactManager, ParticlesContacts};
use crate::kernel::{CubicSplineKernel, Kernel, Poly6Kernel, SpikyKernel};
use crate::math::{Vector, DIM, SPATIAL_DIM};
use crate::object::{Boundary, Fluid};

#[cfg(feature = "dim2")]
type BetaMatrix<N> = na::Matrix3<N>;
#[cfg(feature = "dim3")]
type BetaMatrix<N> = na::Matrix6<N>;
#[cfg(feature = "dim2")]
type BetaGradientMatrix<N> = na::Matrix3x2<N>;
#[cfg(feature = "dim3")]
type BetaGradientMatrix<N> = na::Matrix6x3<N>;
#[cfg(feature = "dim2")]
type StrainRate<N> = na::Vector3<N>;
#[cfg(feature = "dim3")]
type StrainRate<N> = na::Vector6<N>;

#[derive(Copy, Clone, Debug)]
struct StrainRates<N: RealField> {
    target: StrainRate<N>,
    error: StrainRate<N>,
}

impl<N: RealField> StrainRates<N> {
    pub fn new() -> Self {
        Self {
            target: StrainRate::zeros(),
            error: StrainRate::zeros(),
        }
    }
}

fn compute_strain_rate<N: RealField>(gradient: &Vector<N>, v_ji: &Vector<N>) -> StrainRate<N> {
    let _2: N = na::convert(2.0f64);

    #[cfg(feature = "dim3")]
    return StrainRate::new(
        _2 * v_ji.x * gradient.x,
        _2 * v_ji.y * gradient.y,
        _2 * v_ji.z * gradient.z,
        v_ji.x * gradient.y + v_ji.y * gradient.x,
        v_ji.x * gradient.z + v_ji.z * gradient.x,
        v_ji.y * gradient.z + v_ji.z * gradient.y,
    );

    #[cfg(feature = "dim2")]
    return StrainRate::new(
        _2 * v_ji.x * gradient.x,
        _2 * v_ji.y * gradient.y,
        v_ji.x * gradient.y + v_ji.y * gradient.x,
    );
}

fn compute_gradient_matrix<N: RealField>(gradient: &Vector<N>) -> BetaGradientMatrix<N> {
    let _2: N = na::convert(2.0f64);

    #[cfg(feature = "dim2")]
        #[rustfmt::skip]
        return BetaGradientMatrix::new(
        gradient.x * _2, N::zero(),
        N::zero(), gradient.y * _2,
        gradient.y, gradient.x,
    );

    #[cfg(feature = "dim3")]
        #[rustfmt::skip]
        return BetaGradientMatrix::new(
        gradient.x * _2, N::zero(), N::zero(),
        N::zero(), gradient.y * _2, N::zero(),
        N::zero(), N::zero(), gradient.z * _2,
        gradient.y, gradient.x, N::zero(),
        gradient.z, N::zero(), gradient.x,
        N::zero(), gradient.z, gradient.y,
    );
}

pub struct DFSPHViscosity<N: RealField> {
    min_viscosity_iter: usize,
    max_viscosity_iter: usize,
    max_viscosity_error: N,
    betas: Vec<Vec<BetaMatrix<N>>>,
    strain_rates: Vec<Vec<StrainRates<N>>>, // Contains (target rate, error rate)
}

impl<N: RealField> DFSPHViscosity<N> {
    pub fn new() -> Self {
        Self {
            min_viscosity_iter: 1,
            max_viscosity_iter: 1000,
            max_viscosity_error: na::convert(0.01),
            betas: Vec::new(),
            strain_rates: Vec::new(),
        }
    }

    /// Initialize this solver with the given fluids.
    pub fn init_with_fluids(&mut self, fluids: &[Fluid<N>]) {
        // Resize every buffer.
        self.betas.resize(fluids.len(), Vec::new());
        self.strain_rates.resize(fluids.len(), Vec::new());

        for (fluid, betas, strain_rates) in itertools::multizip((
            fluids.iter(),
            self.betas.iter_mut(),
            self.strain_rates.iter_mut(),
        )) {
            betas.resize(fluid.num_particles(), BetaMatrix::zeros());
            strain_rates.resize(fluid.num_particles(), StrainRates::new());
        }
    }

    // NOTE: this actually computes beta / density_i^3
    fn compute_betas(
        &mut self,
        inv_dt: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
    ) {
        let _2: N = na::convert(2.0f64);

        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let fluid_boundary_contacts = &fluid_boundary_contacts[fluid_id];
            let betas_i = &mut self.betas[fluid_id];
            let fluid_i = &fluids[fluid_id];

            par_iter_mut!(betas_i).enumerate().for_each(|(i, beta_i)| {
                let mut grad_sum = BetaGradientMatrix::zeros();
                let mut squared_grad_sum = BetaMatrix::zeros();

                for c in fluid_fluid_contacts.particle_contacts(i) {
                    if c.j_model == fluid_id {
                        let mat = compute_gradient_matrix(&c.gradient);
                        let grad_i = mat * (fluid_i.particle_mass(c.j) / _2);
                        squared_grad_sum += grad_i * grad_i.transpose();
                        grad_sum += grad_i;
                    }
                }

                let mut denominator = squared_grad_sum + grad_sum * grad_sum.transpose();

                // Preconditionner.
                let mut inv_diag = denominator.diagonal();
                inv_diag.apply(|n| {
                    if n.abs() < na::convert(1.0e-6) {
                        N::one()
                    } else {
                        N::one() / n
                    }
                });

                for i in 0..SPATIAL_DIM {
                    denominator.column_mut(i).component_mul_mut(&inv_diag);
                }

                *beta_i = denominator
                    .try_inverse()
                    .unwrap_or_else(|| BetaMatrix::zeros());

                for i in 0..SPATIAL_DIM {
                    let mut col = beta_i.column_mut(i);
                    col *= inv_diag[i];
                }
            })
        }
    }

    // NOTE: this actually computes the strain rates * density_i
    fn compute_strain_rates(
        &mut self,
        dt: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        velocity_changes: &mut [Vec<Vector<N>>],
        compute_error: bool,
    ) -> N {
        let mut max_error = N::zero();
        let _2: N = na::convert(2.0f64);

        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let fluid_boundary_contacts = &fluid_boundary_contacts[fluid_id];
            let strain_rates_i = &mut self.strain_rates[fluid_id];
            let fluid_i = &fluids[fluid_id];

            let err = par_iter_mut!(strain_rates_i).enumerate().fold(
                N::zero(),
                |curr_err, (i, strain_rates_i)| {
                    let mut fluid_rate = StrainRate::zeros();

                    for c in fluid_fluid_contacts.particle_contacts(i) {
                        if c.j_model == fluid_id {
                            let v_i = fluid_i.velocities[c.i] + velocity_changes[fluid_id][c.i];
                            let v_j = fluid_i.velocities[c.j] + velocity_changes[fluid_id][c.j];
                            let v_ji = v_j - v_i;
                            let rate = compute_strain_rate(&c.gradient, &v_ji);

                            fluid_rate += rate * (fluid_i.particle_mass(c.j) / _2);
                        }
                    }

                    if compute_error {
                        strain_rates_i.error = fluid_rate - strain_rates_i.target;
                        curr_err + strain_rates_i.error.lp_norm(1) / na::convert(6.0f64)
                    } else {
                        strain_rates_i.target = fluid_rate * fluid_i.viscosity;
                        N::zero()
                    }
                },
            );

            let nparts = fluids[fluid_id].num_particles();
            if nparts != 0 {
                max_error = max_error.max(err / na::convert(nparts as f64));
            }
        }

        max_error
    }

    fn compute_velocity_changes_for_viscosity(
        &self,
        dt: N,
        inv_dt: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        velocity_changes: &mut [Vec<Vector<N>>],
    ) {
        let strain_rates = &self.strain_rates;
        let betas = &self.betas;
        let _2: N = na::convert(2.0);

        for (fluid_id, fluid1) in fluids.iter().enumerate() {
            par_iter_mut!(velocity_changes[fluid_id])
                .enumerate()
                .for_each(|(i, velocity_change)| {
                    let ui = betas[fluid_id][i] * strain_rates[fluid_id][i].error;

                    for c in fluid_fluid_contacts[fluid_id].particle_contacts(i) {
                        if c.j_model == fluid_id {
                            let uj = betas[fluid_id][c.j] * strain_rates[fluid_id][c.j].error;
                            let gradient = compute_gradient_matrix(&c.gradient);

                            // Compute velocity change.
                            let coeff = (ui + uj) * (fluid1.particle_mass(c.i) / _2);
                            *velocity_change += gradient.tr_mul(&coeff) * fluid1.particle_mass(c.i);
                        }
                    }
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
        velocity_changes: &mut [Vec<Vector<N>>],
    ) {
        let _ = self.compute_betas(
            inv_dt,
            &contact_manager.fluid_fluid_contacts,
            &contact_manager.fluid_boundary_contacts,
            fluids,
        );

        let _ = self.compute_strain_rates(
            dt,
            &contact_manager.fluid_fluid_contacts,
            &contact_manager.fluid_boundary_contacts,
            fluids,
            velocity_changes,
            false,
        );

        let mut last_err = N::max_value();

        for i in 0..self.max_viscosity_iter {
            let avg_err = self.compute_strain_rates(
                dt,
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
                velocity_changes,
                true,
            );

            if avg_err > last_err
                || (avg_err <= self.max_viscosity_error && i >= self.min_viscosity_iter)
            {
                println!(
                    "Average viscosity error: {}, break after niters: {}, unstable: {}",
                    avg_err,
                    i,
                    avg_err > last_err
                );
                break;
            }

            last_err = avg_err;

            self.compute_velocity_changes_for_viscosity(
                dt,
                inv_dt,
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
                velocity_changes,
            );
        }
    }
}
