use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::iter;
use std::marker::PhantomData;
use std::ops::{AddAssign, SubAssign};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "dim3")]
use na::Vector2;
use na::{self, DVector, DVectorSlice, DVectorSliceMut, RealField, Unit, VectorSliceMutN};

use crate::boundary::Boundary;
use crate::fluid::Fluid;
use crate::geometry::ParticlesContacts;
use crate::kernel::{Kernel, Poly6Kernel, SpikyKernel};
use crate::math::{Dim, Point, Vector, DIM};

macro_rules! par_iter {
    ($t: expr) => {{
        #[cfg(not(feature = "parallel"))]
        let it = $t.iter();

        #[cfg(feature = "parallel")]
        let it = $t.par_iter();
        it
    }};
}

macro_rules! par_iter_mut {
    ($t: expr) => {{
        #[cfg(not(feature = "parallel"))]
        let it = $t.iter_mut();

        #[cfg(feature = "parallel")]
        let it = $t.par_iter_mut();
        it
    }};
}

pub struct PBFSolver<
    N: RealField,
    KernelDensity: Kernel = Poly6Kernel,
    KernelGradient: Kernel = SpikyKernel,
> {
    h: N,
    lambdas: Vec<Vec<N>>,
    densities: Vec<Vec<N>>,
    position_changes: Vec<Vec<Vector<N>>>,
    phantoms: PhantomData<(KernelDensity, KernelGradient)>,
}

impl<N, KernelDensity, KernelGradient> PBFSolver<N, KernelDensity, KernelGradient>
where
    N: RealField,
    KernelDensity: Kernel,
    KernelGradient: Kernel,
{
    pub fn new(h: N) -> Self {
        Self {
            h,
            lambdas: Vec::new(),
            densities: Vec::new(),
            position_changes: Vec::new(),
            phantoms: PhantomData,
        }
    }

    fn update_contacts(
        &mut self,
        fluid_contacts: &mut [ParticlesContacts<N>],
        boundary_contacts: &mut [ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        let h = self.h;

        for contacts in fluid_contacts.iter_mut() {
            par_iter_mut!(contacts.contacts_mut()).for_each(|c| {
                let fluid1 = &fluids[c.i_model];
                let fluid2 = &fluids[c.j_model];

                let pi = fluid1.positions[c.i] + self.position_changes[c.i_model][c.i];
                let pj = fluid2.positions[c.j] + self.position_changes[c.j_model][c.j];

                c.weight = KernelDensity::points_apply(&pi, &pj, h);
                c.gradient = KernelGradient::points_apply_diff1(&pi, &pj, h);
            })
        }

        for contacts in boundary_contacts.iter_mut() {
            par_iter_mut!(contacts.contacts_mut()).for_each(|c| {
                let fluid1 = &fluids[c.i_model];
                let bound2 = &boundaries[c.j_model];

                let pi = fluid1.positions[c.i] + self.position_changes[c.i_model][c.i];
                let pj = bound2.positions[c.j];

                c.weight = KernelDensity::points_apply(&pi, &pj, h);
                c.gradient = KernelGradient::points_apply_diff1(&pi, &pj, h);
            })
        }
    }

    fn resize_buffers(&mut self, dt: N, fluids: &[Fluid<N>]) {
        // Resize every buffer.
        self.lambdas.resize(fluids.len(), Vec::new());
        self.densities.resize(fluids.len(), Vec::new());
        self.position_changes.resize(fluids.len(), Vec::new());

        for (fluid, lambdas, densities, position_changes) in itertools::multizip((
            fluids.iter(),
            self.lambdas.iter_mut(),
            self.densities.iter_mut(),
            self.position_changes.iter_mut(),
        )) {
            lambdas.resize(fluid.num_particles(), N::zero());
            densities.resize(fluid.num_particles(), N::zero());
            position_changes.resize(fluid.num_particles(), Vector::zeros());
        }
    }

    fn predict_advection(&mut self, dt: N, gravity: &Vector<N>, fluids: &[Fluid<N>]) {
        for (fluid, position_changes) in fluids.iter().zip(self.position_changes.iter_mut()) {
            par_iter_mut!(position_changes)
                .zip(par_iter!(fluid.velocities))
                .for_each(|(position_change, vel)| *position_change = (vel + gravity * dt) * dt)
        }
    }

    /*
    fn compute_boundary_volumes(
        boundary: &HGrid<N, usize>,
        boundary_positions: &DVector<N>,
        boundary_volumes: &mut DVector<N>
    ) {
        let num_boundary_particles = boundary_volumes.len();
        boundary_volumes.fill(N::zero());

        for i in 0..num_boundary_particles {
            let pi = Point::from(boundary_positions.fixed_rows::<Dim>(i * DIM).into_owned());
            let mut denominator = N::zero();

            for j in boundary.elements_close_to_point(&pi, h).cloned() {
                let pj = Point::from(boundary_positions.fixed_rows::<Dim>(j * DIM).into_owned());
                let weight = KernelDensity::points_apply(&pi, &pj, h);
                denominator += weight;
            }

            boundary_volumes[i] = N::one() / denominator;
        }
    }*/

    fn compute_densities(
        &mut self,
        fluid_contacts: &[ParticlesContacts<N>],
        boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        for fluid_id in 0..fluids.len() {
            par_iter_mut!(self.densities[fluid_id])
                .enumerate()
                .for_each(|(i, density)| {
                    *density = N::zero();

                    for c in fluid_contacts[fluid_id].particle_contacts(i) {
                        *density += fluids[c.j_model].volumes[c.j] * c.weight;
                    }

                    for c in boundary_contacts[fluid_id].particle_contacts(i) {
                        *density += boundaries[c.j_model].volumes[c.j] * c.weight;
                    }
                })
        }
    }

    /*
    fn compute_average_density_error(num_particles: usize, densities: &DVector<N>, densities0: &DVector<N>) -> N {
        let mut avg_density_err = N::zero();

        for i in 0..num_particles {
            avg_density_err += densities0[i] * (densities[i] - N::one()).max(N::zero());
        }

        avg_density_err / na::convert(num_particles as f64)
    }*/

    fn compute_lambdas(
        &mut self,
        fluid_contacts: &[ParticlesContacts<N>],
        boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        for fluid_id in 0..fluids.len() {
            let fluid1 = &fluids[fluid_id];
            let fluid_contacts = &fluid_contacts[fluid_id];
            let boundary_contacts = &boundary_contacts[fluid_id];
            let densities1 = self.densities[fluid_id].as_slice();
            let lambdas1 = &mut self.lambdas[fluid_id];

            par_iter_mut!(lambdas1)
                .enumerate()
                .for_each(|(i, lambda1)| {
                    let density_err = densities1[i] - N::one();

                    if density_err > N::zero() {
                        let mut total_gradient = Vector::zeros();
                        let mut denominator = N::zero();

                        for c in fluid_contacts.particle_contacts(i) {
                            let grad_i = c.gradient * fluids[c.j_model].volumes[c.j];
                            denominator += grad_i.norm_squared();
                            total_gradient += grad_i;
                        }

                        for c in boundary_contacts.particle_contacts(i) {
                            let grad_i = c.gradient * boundaries[c.j_model].volumes[c.j];
                            denominator += grad_i.norm_squared();
                            total_gradient += grad_i;
                        }

                        let denominator = denominator + total_gradient.norm_squared();
                        *lambda1 = -density_err / (denominator + na::convert(1.0e-6));
                    }
                })
        }
    }

    fn compute_position_changes(
        &mut self,
        fluid_contacts: &[ParticlesContacts<N>],
        boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        let h = self.h;
        let densities = &self.densities;
        let lambdas = &self.lambdas;

        for (fluid_id, fluid1) in fluids.iter().enumerate() {
            par_iter_mut!(self.position_changes[fluid_id])
                .enumerate()
                .for_each(|(i, position_change)| {

                 position_change.fill(N::zero());

                for c in fluid_contacts[fluid_id].particle_contacts(i) {
                    let fluid2 = &fluids[c.j_model];

                    // Compute virtual pressure.
                    let k: N = na::convert(0.001);
                    let n = 4;
                    let dq = N::zero();
                    let scorr = -k * (c.weight / KernelDensity::scalar_apply(dq, h)).powi(n);

                    // Compute velocity change.
                    let coeff = fluid2.volumes[c.j] * (lambdas[c.i_model][c.i] + densities[c.j_model][c.j] / densities[c.i_model][c.i] * lambdas[c.j_model][c.j])/* + scorr*/;
                    position_change.axpy(coeff, &c.gradient, N::one());
                }


                for c in boundary_contacts[fluid_id].particle_contacts(i) {
                    let boundary2 = &boundaries[c.j_model];

                    let lambda = lambdas[c.i_model][c.i];
                    let coeff = boundary2.volumes[c.j] * (lambda + lambda)/* + scorr*/;
                    position_change.axpy(coeff, &c.gradient, N::one());
                    // XXX: apply the force to the boundary too.
                }
            })
        }
    }

    /*
        fn apply_viscosity(&mut self) {
            let viscosity: N = na::convert(0.01); // XXX

            // Add XSPH viscosity
            let mut viscosity_velocities = DVector::zeros(self.velocities.len());

            for c in &self.fluid_contacts {
                if c.i != c.j {
                    let vi = self.velocities.fixed_rows::<Dim>(c.i * DIM);
                    let vj = self.velocities.fixed_rows::<Dim>(c.j * DIM);
                    let extra_vel = (vj - vi) * c.weight;

                    viscosity_velocities.fixed_rows_mut::<Dim>(c.i * DIM).add_assign(&extra_vel);
                    viscosity_velocities.fixed_rows_mut::<Dim>(c.j * DIM).sub_assign(&extra_vel);
                }
            }

            self.velocities.axpy(viscosity, &viscosity_velocities, N::one());
        }
    }
        */

    pub fn update_velocities_and_positions(&mut self, inv_dt: N, fluids: &mut [Fluid<N>]) {
        for (fluid, delta) in fluids.iter_mut().zip(self.position_changes.iter()) {
            par_iter_mut!(fluid.positions)
                .zip(par_iter_mut!(fluid.velocities))
                .zip(par_iter!(delta))
                .for_each(|((pos, vel), delta)| {
                    *vel = delta * inv_dt;
                    *pos += delta;
                })
        }
    }

    pub fn step(
        &mut self,
        dt: N,
        gravity: &Vector<N>,
        fluid_contacts: &mut [ParticlesContacts<N>],
        boundary_contacts: &mut [ParticlesContacts<N>],
        fluids: &mut [Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        if dt.is_zero() {
            return;
        }

        let inv_dt = N::one() / dt;

        let niters = 10;
        self.resize_buffers(dt, fluids);
        self.predict_advection(dt, gravity, fluids);

        for loop_i in 0..niters {
            self.update_contacts(fluid_contacts, boundary_contacts, fluids, boundaries);
            self.compute_densities(fluid_contacts, boundary_contacts, fluids, boundaries);
            //                let err = Self::compute_average_density_error(self.num_particles, &densities, &self.densities0);
            self.compute_lambdas(fluid_contacts, boundary_contacts, fluids, boundaries);
            self.compute_position_changes(fluid_contacts, boundary_contacts, fluids, boundaries);
        }

        // Compute actual velocities.
        self.update_velocities_and_positions(inv_dt, fluids);
        // self.apply_viscosity();
    }
}
