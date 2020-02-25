#[cfg(feature = "parallel")]
use rayon::prelude::*;

use na::{self, RealField};

use crate::geometry::ParticlesContacts;
use crate::math::{Vector, SPATIAL_DIM};
use crate::object::{Boundary, Fluid};
use crate::solver::NonPressureForce;
use crate::TimestepManager;

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
    pub min_viscosity_iter: usize,
    pub max_viscosity_iter: usize,
    pub max_viscosity_error: N,
    viscosity_coefficient: N,
    betas: Vec<BetaMatrix<N>>,
    strain_rates: Vec<StrainRates<N>>,
}

impl<N: RealField> DFSPHViscosity<N> {
    pub fn new(viscosity_coefficient: N) -> Self {
        assert!(
            viscosity_coefficient >= N::zero() && viscosity_coefficient <= N::one(),
            "The viscosity coefficient must be between 0.0 and 1.0."
        );

        Self {
            min_viscosity_iter: 1,
            max_viscosity_iter: 50,
            max_viscosity_error: na::convert(0.01),
            viscosity_coefficient,
            betas: Vec::new(),
            strain_rates: Vec::new(),
        }
    }

    fn init(&mut self, fluid: &Fluid<N>) {
        if self.betas.len() != fluid.num_particles() {
            self.betas
                .resize(fluid.num_particles(), BetaMatrix::zeros());
            self.strain_rates
                .resize(fluid.num_particles(), StrainRates::new());
        }
    }

    fn compute_betas(
        &mut self,
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid: &Fluid<N>,
        densities: &[N],
    ) {
        let _2: N = na::convert(2.0f64);

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
                    if n.abs() < na::convert(1.0e-6) {
                        N::one()
                    } else {
                        N::one() / n
                    }
                });

                for i in 0..SPATIAL_DIM {
                    denominator.column_mut(i).component_mul_assign(&inv_diag);
                }

                if SPATIAL_DIM == 3 {
                    if denominator.determinant().abs() < na::convert(1.0e-6) {
                        *beta_i = BetaMatrix::zeros()
                    } else {
                        *beta_i = denominator
                            .try_inverse()
                            .unwrap_or_else(|| BetaMatrix::zeros())
                    }
                }
                let lu = denominator.lu();

                if lu.determinant().abs() < na::convert(1.0e-6) {
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
        timestep: &TimestepManager<N>,
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid: &Fluid<N>,
        densities: &[N],
        compute_error: bool,
    ) -> N {
        let mut max_error = N::zero();
        let viscosity_coefficient = self.viscosity_coefficient;
        let _2: N = na::convert(2.0f64);

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
                    strain_rates_i.error.lp_norm(1) / na::convert(6.0f64)
                } else {
                    strain_rates_i.target = fluid_rate * (N::one() - viscosity_coefficient);
                    N::zero()
                }
            });

        let err = par_reduce_sum!(N::zero(), it);

        let nparts = fluid.num_particles();
        if nparts != 0 {
            max_error = max_error.max(err / na::convert(nparts as f64));
        }

        max_error
    }

    fn compute_accelerations(
        &self,
        timestep: &TimestepManager<N>,
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid: &mut Fluid<N>,
        densities: &[N],
    ) {
        let strain_rates = &self.strain_rates;
        let betas = &self.betas;
        let volumes = &fluid.volumes;
        let density0 = fluid.density0;
        let _2: N = na::convert(2.0);

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

impl<N: RealField> NonPressureForce<N> for DFSPHViscosity<N> {
    fn solve(
        &mut self,
        timestep: &TimestepManager<N>,
        _kernel_radius: N,
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid_boundaries_contacts: &ParticlesContacts<N>,
        fluid: &mut Fluid<N>,
        boundaries: &[Boundary<N>],
        densities: &[N],
    ) {
        self.init(fluid);

        let _ = self.compute_betas(fluid_fluid_contacts, fluid, densities);

        let _ = self.compute_strain_rates(timestep, fluid_fluid_contacts, fluid, densities, false);

        let mut last_err = N::max_value();

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

            last_err = avg_err;

            self.compute_accelerations(timestep, fluid_fluid_contacts, fluid, densities);
        }
    }

    fn apply_permutation(&mut self, _: &[usize]) {}
}
