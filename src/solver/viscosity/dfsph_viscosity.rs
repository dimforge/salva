#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::geometry::ParticlesContacts;
use crate::math::{Real, Vector, SPATIAL_DIM};
use crate::object::{Boundary, Fluid};
use crate::solver::NonPressureForce;
use crate::TimestepManager;

#[cfg(feature = "dim2")]
type BetaMatrix<Real> = na::Matrix3<Real>;
#[cfg(feature = "dim3")]
type BetaMatrix<Real> = na::Matrix6<Real>;
#[cfg(feature = "dim2")]
type BetaGradientMatrix<Real> = na::Matrix3x2<Real>;
#[cfg(feature = "dim3")]
type BetaGradientMatrix<Real> = na::Matrix6x3<Real>;
#[cfg(feature = "dim2")]
type StrainRate<Real> = na::Vector3<Real>;
#[cfg(feature = "dim3")]
type StrainRate<Real> = na::Vector6<Real>;

#[derive(Copy, Clone, Debug)]
struct StrainRates {
    target: StrainRate<Real>,
    error: StrainRate<Real>,
}

impl StrainRates {
    pub fn new() -> Self {
        Self {
            target: StrainRate::zeros(),
            error: StrainRate::zeros(),
        }
    }
}

fn compute_strain_rate(gradient: &Vector<Real>, v_ji: &Vector<Real>) -> StrainRate<Real> {
    let _2: Real = na::convert::<_, Real>(2.0f64);

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

fn compute_gradient_matrix(gradient: &Vector<Real>) -> BetaGradientMatrix<Real> {
    let _2: Real = na::convert::<_, Real>(2.0f64);

    #[cfg(feature = "dim2")]
        #[rustfmt::skip]
        return BetaGradientMatrix::new(
        gradient.x * _2, na::zero::<Real>(),
        na::zero::<Real>(), gradient.y * _2,
        gradient.y, gradient.x,
    );

    #[cfg(feature = "dim3")]
        #[rustfmt::skip]
        return BetaGradientMatrix::new(
        gradient.x * _2, na::zero::<Real>(), na::zero::<Real>(),
        na::zero::<Real>(), gradient.y * _2, na::zero::<Real>(),
        na::zero::<Real>(), na::zero::<Real>(), gradient.z * _2,
        gradient.y, gradient.x, na::zero::<Real>(),
        gradient.z, na::zero::<Real>(), gradient.x,
        na::zero::<Real>(), gradient.z, gradient.y,
    );
}

/// Viscosity introduced with the Viscous DFSPH method.
///
/// This does not include any viscosity with boundaries so it can be useful to
/// combine this with another viscosity model and include only its boundary part.
pub struct DFSPHViscosity {
    /// Minimum number of iterations that must be executed for viscosity resolution.
    pub min_viscosity_iter: usize,
    /// Maximum number of iterations that must be executed for viscosity resolution.
    pub max_viscosity_iter: usize,
    /// Maximum acceptable strain error (in percents).
    ///
    /// The viscosity solver will continue iterating until the strain error drops bellow this
    /// threshold, or until the maximum number of iterations is reached.
    pub max_viscosity_error: Real,
    /// The viscosity coefficient.
    pub viscosity_coefficient: Real,
    betas: Vec<BetaMatrix<Real>>,
    strain_rates: Vec<StrainRates>,
}

impl DFSPHViscosity {
    /// Initialize a new DFSPH visocisity solver.
    pub fn new(viscosity_coefficient: Real) -> Self {
        assert!(
            viscosity_coefficient >= na::zero::<Real>()
                && viscosity_coefficient <= na::one::<Real>(),
            "The viscosity coefficient must be between 0.0 and 1.0."
        );

        Self {
            min_viscosity_iter: 1,
            max_viscosity_iter: 50,
            max_viscosity_error: na::convert::<_, Real>(0.01),
            viscosity_coefficient,
            betas: Vec::new(),
            strain_rates: Vec::new(),
        }
    }

    fn init(&mut self, fluid: &Fluid) {
        if self.betas.len() != fluid.num_particles() {
            self.betas
                .resize(fluid.num_particles(), BetaMatrix::zeros());
            self.strain_rates
                .resize(fluid.num_particles(), StrainRates::new());
        }
    }

    fn compute_betas(
        &mut self,
        fluid_fluid_contacts: &ParticlesContacts,
        fluid: &Fluid,
        densities: &[Real],
    ) {
        let _2: Real = na::convert::<_, Real>(2.0f64);

        par_iter_mut!(self.betas)
            .enumerate()
            .for_each(|(i, beta_i)| {
                let mut grad_sum = BetaGradientMatrix::zeros();
                let mut squared_grad_sum = BetaMatrix::zeros();

                for c in fluid_fluid_contacts
                    .particle_contacts(i)
                    .read()
                    .unwrap()
                    .iter()
                {
                    if c.i_model == c.j_model {
                        let mat = compute_gradient_matrix(&c.gradient);
                        let grad_i = mat * (fluid.particle_mass(c.j) / (_2 * densities[c.i]));
                        squared_grad_sum += grad_i * grad_i.transpose() / densities[c.i];
                        grad_sum += grad_i;
                    }
                }

                let mut denominator =
                    squared_grad_sum + grad_sum * grad_sum.transpose() / densities[i];

                // Preconditionner.
                let mut inv_diag = denominator.diagonal();
                inv_diag.apply(|n| {
                    if n.abs() < na::convert::<_, Real>(1.0e-6) {
                        *n = na::one::<Real>();
                    } else {
                        *n = na::one::<Real>() / *n
                    }
                });

                for i in 0..SPATIAL_DIM {
                    denominator.column_mut(i).component_mul_assign(&inv_diag);
                }

                if SPATIAL_DIM == 3 {
                    if denominator.determinant().abs() < na::convert::<_, Real>(1.0e-6) {
                        *beta_i = BetaMatrix::zeros()
                    } else {
                        *beta_i = denominator
                            .try_inverse()
                            .unwrap_or_else(|| BetaMatrix::zeros())
                    }
                }
                let lu = denominator.lu();

                if lu.determinant().abs() < na::convert::<_, Real>(1.0e-6) {
                    *beta_i = BetaMatrix::zeros();
                } else {
                    *beta_i = lu.try_inverse().unwrap_or_else(|| BetaMatrix::zeros());
                }

                for i in 0..SPATIAL_DIM {
                    let mut col = beta_i.column_mut(i);
                    col *= inv_diag[i];
                }
            })
    }

    fn compute_strain_rates(
        &mut self,
        timestep: &TimestepManager,
        fluid_fluid_contacts: &ParticlesContacts,
        fluid: &Fluid,
        densities: &[Real],
        compute_error: bool,
    ) -> Real {
        let mut max_error = na::zero::<Real>();
        let viscosity_coefficient = self.viscosity_coefficient;
        let _2: Real = na::convert::<_, Real>(2.0f64);

        let it = par_iter_mut!(self.strain_rates)
            .enumerate()
            .map(|(i, strain_rates_i)| {
                let mut fluid_rate = StrainRate::zeros();

                for c in fluid_fluid_contacts
                    .particle_contacts(i)
                    .read()
                    .unwrap()
                    .iter()
                {
                    if c.i_model == c.j_model {
                        let v_i = fluid.velocities[c.i] + fluid.accelerations[c.i] * timestep.dt();
                        let v_j = fluid.velocities[c.j] + fluid.accelerations[c.j] * timestep.dt();
                        let v_ji = v_j - v_i;
                        let rate = compute_strain_rate(&c.gradient, &v_ji);

                        fluid_rate += rate * (fluid.particle_mass(c.j) / (_2 * densities[c.i]));
                    }
                }

                if compute_error {
                    strain_rates_i.error = fluid_rate - strain_rates_i.target;
                    strain_rates_i.error.lp_norm(1) / na::convert::<_, Real>(6.0f64)
                } else {
                    strain_rates_i.target =
                        fluid_rate * (na::one::<Real>() - viscosity_coefficient);
                    na::zero::<Real>()
                }
            });

        let err = par_reduce_sum!(na::zero::<Real>(), it);

        let nparts = fluid.num_particles();
        if nparts != 0 {
            max_error = max_error.max(err / na::convert::<_, Real>(nparts as f64));
        }

        max_error
    }

    fn compute_accelerations(
        &self,
        timestep: &TimestepManager,
        fluid_fluid_contacts: &ParticlesContacts,
        fluid: &mut Fluid,
        densities: &[Real],
    ) {
        let strain_rates = &self.strain_rates;
        let betas = &self.betas;
        let volumes = &fluid.volumes;
        let density0 = fluid.density0;
        let _2: Real = na::convert::<_, Real>(2.0);

        par_iter_mut!(fluid.accelerations)
            .enumerate()
            .for_each(|(i, acceleration)| {
                let ui = betas[i] * strain_rates[i].error / (densities[i] * densities[i]);

                for c in fluid_fluid_contacts
                    .particle_contacts(i)
                    .read()
                    .unwrap()
                    .iter()
                {
                    if c.i_model == c.j_model {
                        let uj = betas[c.j] * strain_rates[c.j].error
                            / (densities[c.j] * densities[c.j]);
                        let gradient = compute_gradient_matrix(&c.gradient);

                        // Compute velocity change.
                        let coeff = (ui + uj) * (volumes[c.j] * density0 / _2);
                        *acceleration +=
                            gradient.tr_mul(&coeff) * (volumes[c.i] * density0 * timestep.inv_dt());
                    }
                }
            })
    }
}

impl NonPressureForce for DFSPHViscosity {
    fn solve(
        &mut self,
        timestep: &TimestepManager,
        _kernel_radius: Real,
        fluid_fluid_contacts: &ParticlesContacts,
        _fluid_boundaries_contacts: &ParticlesContacts,
        fluid: &mut Fluid,
        _boundaries: &[Boundary],
        densities: &[Real],
    ) {
        self.init(fluid);

        let _ = self.compute_betas(fluid_fluid_contacts, fluid, densities);

        let _ = self.compute_strain_rates(timestep, fluid_fluid_contacts, fluid, densities, false);

        for i in 0..self.max_viscosity_iter {
            let avg_err =
                self.compute_strain_rates(timestep, fluid_fluid_contacts, fluid, densities, true);

            if avg_err <= self.max_viscosity_error && i >= self.min_viscosity_iter {
                //                println!(
                //                    "Average viscosity error: {}, break after niters: {}, unstable: {}",
                //                    avg_err,
                //                    i,
                //                    avg_err > last_err
                //                );
                break;
            }

            self.compute_accelerations(timestep, fluid_fluid_contacts, fluid, densities);
        }
    }

    fn apply_permutation(&mut self, _: &[usize]) {}
}
