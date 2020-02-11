use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use na::{self, RealField};

use crate::geometry::{ContactManager, ParticlesContacts};
use crate::kernel::{CubicSplineKernel, Kernel, Poly6Kernel, SpikyKernel};
use crate::math::{Matrix, Point, RotationMatrix, SpatialVector, Vector, DIM, SPATIAL_DIM};
use crate::object::{Boundary, Fluid};
use crate::solver::NonPressureForce;
use itertools::Itertools;

fn elasticity_coefficients<N: RealField>(young_modulus: N, poisson_ratio: N) -> (N, N, N) {
    let _1 = N::one();
    let _2: N = na::convert(2.0);

    let d0 =
        (young_modulus * (_1 - poisson_ratio)) / ((_1 + poisson_ratio) * (_1 - _2 * poisson_ratio));
    let d1 = (young_modulus * poisson_ratio) / ((_1 + poisson_ratio) * (_1 - _2 * poisson_ratio));
    let d2 = (young_modulus * (_1 - _2 * poisson_ratio))
        / (_2 * (_1 + poisson_ratio) * (_1 - _2 * poisson_ratio));
    (d0, d1, d2)
}

fn sym_mat_mul_vec<N: RealField>(mat: &SpatialVector<N>, v: &Vector<N>) -> Vector<N> {
    #[cfg(feature = "dim2")]
    return Vector::new(mat.x * v.x + mat.z * v.y, mat.z * v.x + mat.y * v.y);

    #[cfg(feature = "dim3")]
    return Vector::new(
        mat.x * v.x + mat.w * v.y + mat.a * v.z,
        mat.w * v.x + mat.y * v.y + mat.b * v.z,
        mat.a * v.x + mat.b * v.y + mat.z * v.z,
    );
}

// https://cg.informatik.uni-freiburg.de/publications/2009_NP_corotatedSPH.pdf
pub struct Becker2009Elasticity<
    N: RealField,
    KernelDensity: Kernel = CubicSplineKernel,
    KernelGradient: Kernel = CubicSplineKernel,
> {
    d0: N,
    d1: N,
    d2: N,
    nonlinear_strain: bool,
    positions0: Vec<Point<N>>,
    contacts0: ParticlesContacts<N>,
    rotations: Vec<RotationMatrix<N>>,
    deformation_gradient_tr: Vec<Matrix<N>>,
    stress: Vec<SpatialVector<N>>,
    phantom: PhantomData<(KernelDensity, KernelGradient)>,
}

impl<N: RealField, KernelDensity: Kernel, KernelGradient: Kernel>
    Becker2009Elasticity<N, KernelDensity, KernelGradient>
{
    fn compute_rotations(&mut self, kernel_radius: N, fluid: &Fluid<N>) {
        let _2: N = na::convert(2.0f64);

        let contacts0 = &self.contacts0;
        let positions0 = &self.positions0;

        par_iter_mut!(&mut self.rotations)
            .enumerate()
            .for_each(|(i, rotation)| {
                let mut a_pq = Matrix::zeros();

                for c in contacts0.particle_contacts(i) {
                    let p_ji = fluid.positions[c.j] - fluid.positions[c.i];
                    let p0_ji = positions0[c.j] - positions0[c.i];
                    let coeff = c.weight * fluid.particle_mass(c.j);
                    a_pq += p_ji * (p0_ji * coeff).transpose();
                }

                // Extract the rotation matrix.
                *rotation =
                    RotationMatrix::from_matrix_eps(&a_pq, N::default_epsilon(), 20, *rotation);
            })
    }

    fn compute_stresses(&mut self, kernel_radius: N, fluid: &Fluid<N>) {
        let _2: N = na::convert(2.0f64);
        let _0_5: N = na::convert(0.564);

        let contacts0 = &self.contacts0;
        let rotations = &self.rotations;
        let positions0 = &self.positions0;

        // let _0 = N::zero();
        // let c = Matrix::new(
        //     d0, d1, d1, _0, _0, _0,
        //     d1, d0, d1, _0, _0, _0,
        //     d1, d1, d0, _0, _0, _0,
        //     _0, _0, _0, d2, _0, _0,
        //     _0, _0, _0, _0, d2, _0,
        //     _0, _0, _0, _0, _0, d2,
        // );
        #[rustfmt::skip]
            #[cfg(feature = "dim3")]
            let c_top_left = Matrix::new(
            self.d0, self.d1, self.d1,
            self.d1, self.d0, self.d1,
            self.d1, self.d1, self.d0,
        );
        #[rustfmt::skip]
            #[cfg(feature = "dim2")]
            let c_top_left = Matrix::new(
            self.d0, self.d1,
            self.d1, self.d0,
        );
        let d2 = self.d2;

        let nonlinear_strain = self.nonlinear_strain;

        par_iter_mut!(&mut self.deformation_gradient_tr)
            .zip(&mut self.stress)
            .enumerate()
            .for_each(|(i, (deformation_grad_tr, stress))| {
                let mut grad_tr = Matrix::zeros();

                for c in contacts0.particle_contacts(i) {
                    let p_ji = fluid.positions[c.j] - fluid.positions[c.i];
                    let p0_ji = positions0[c.j] - positions0[c.i];
                    let u_ji = rotations[c.i].inverse_transform_vector(&(p_ji)) - p0_ji;
                    grad_tr += (c.gradient * fluid.volumes[c.j]) * u_ji.transpose();
                }

                *deformation_grad_tr = grad_tr;

                #[cfg(feature = "dim3")]
                {
                    if nonlinear_strain {
                        let j = grad_tr + Matrix::identity();
                        let jjt = j * j.transpose();

                        let stress012 = c_top_left
                            * Vector::new(
                                jjt.m11 - N::one(),
                                jjt.m22 - N::one(),
                                jjt.m33 - N::one(),
                            )
                            * _0_5;
                        *stress = SpatialVector::new(
                            stress012.x,
                            stress012.y,
                            stress012.z,
                            jjt.m21 * _0_5 * d2,
                            jjt.m31 * _0_5 * d2,
                            jjt.m32 * _0_5 * d2,
                        );
                    } else {
                        // let strain = Vector::new(
                        //     grad_tr.m11,
                        //     grad_tr.m22,
                        //     grad_tr.m33,
                        //     (grad_tr.m21 + grad_tr.m12) * _0_5,
                        //     (grad_tr.m31 + grad_tr.m13) * _0_5,
                        //     (grad_tr.m23 + grad_tr.m32) * _0_5,
                        // );

                        let stress012 =
                            c_top_left * Vector::new(grad_tr.m11, grad_tr.m22, grad_tr.m33);
                        *stress = SpatialVector::new(
                            stress012.x,
                            stress012.y,
                            stress012.z,
                            (grad_tr.m21 + grad_tr.m12) * _0_5 * d2,
                            (grad_tr.m31 + grad_tr.m13) * _0_5 * d2,
                            (grad_tr.m23 + grad_tr.m32) * _0_5 * d2,
                        );
                    }
                }
            })
    }
}

impl<N: RealField, KernelDensity: Kernel, KernelGradient: Kernel> NonPressureForce<N>
    for Becker2009Elasticity<N, KernelDensity, KernelGradient>
{
    fn solve(
        &mut self,
        dt: N,
        kernel_radius: N,
        fluid: &Fluid<N>,
        velocity_changes: &mut [Vector<N>],
    ) {
        let _0_5: N = na::convert(0.5f64);
        self.compute_rotations(kernel_radius, fluid);
        self.compute_stresses(kernel_radius, fluid);

        // Compute and apply forces.
        let contacts0 = &self.contacts0;
        let deformation_gradient_tr = &self.deformation_gradient_tr;
        let rotations = &self.rotations;
        let stress = &self.stress;

        if self.nonlinear_strain {
            par_iter_mut!(velocity_changes)
                .enumerate()
                .for_each(|(i, velocity_change)| {
                    for c in contacts0.particle_contacts(i) {
                        let mut force = Vector::zeros();

                        let grad_tr_i = &deformation_gradient_tr[c.i];
                        let d_ij = c.gradient * fluid.volumes[c.j];
                        let sigma_d_ij = sym_mat_mul_vec(&stress[c.i], &d_ij);
                        let f_ji = (sigma_d_ij + grad_tr_i * sigma_d_ij) * -fluid.volumes[c.i];

                        let grad_tr_j = &deformation_gradient_tr[c.j];
                        let d_ji = c.gradient * (-fluid.volumes[c.i]);
                        let sigma_d_ji = sym_mat_mul_vec(&stress[c.j], &d_ji);
                        let f_ij = (sigma_d_ji + grad_tr_j * sigma_d_ji) * -fluid.volumes[c.j];

                        force += (rotations[c.j] * f_ij - (rotations[c.i] * f_ij)) / _0_5;

                        *velocity_change += force * (dt / fluid.particle_mass(i));
                    }
                })
        } else {
            par_iter_mut!(velocity_changes)
                .enumerate()
                .for_each(|(i, velocity_change)| {
                    for c in contacts0.particle_contacts(i) {
                        let mut force = Vector::zeros();

                        let d_ij = c.gradient * fluid.volumes[c.j];
                        let f_ji = sym_mat_mul_vec(&stress[c.i], &d_ij) * -fluid.volumes[c.i];

                        let d_ji = c.gradient * (-fluid.volumes[c.i]);
                        let f_ij = sym_mat_mul_vec(&stress[c.j], &d_ji) * -fluid.volumes[c.j];

                        force += (rotations[c.j] * f_ij - (rotations[c.i] * f_ij)) / _0_5;

                        *velocity_change += force * (dt / fluid.particle_mass(i));
                    }
                })
        }
    }
}
