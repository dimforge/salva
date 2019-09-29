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
use crate::geometry::{ContactManager, ParticlesContacts};
use crate::kernel::{Kernel, Poly6Kernel, SpikyKernel};
use crate::math::{Dim, Point, Vector, DIM};
use crate::TimestepManager;

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
    lambdas: Vec<Vec<N>>,
    densities: Vec<Vec<N>>,
    boundaries_volumes: Vec<Vec<N>>,
    position_changes: Vec<Vec<Vector<N>>>,
    nonpressure_forces: Vec<Vec<Vector<N>>>,
    phantoms: PhantomData<(KernelDensity, KernelGradient)>,
}

impl<N, KernelDensity, KernelGradient> PBFSolver<N, KernelDensity, KernelGradient>
where
    N: RealField,
    KernelDensity: Kernel,
    KernelGradient: Kernel,
{
    pub fn new() -> Self {
        Self {
            lambdas: Vec::new(),
            densities: Vec::new(),
            boundaries_volumes: Vec::new(),
            position_changes: Vec::new(),
            nonpressure_forces: Vec::new(),
            phantoms: PhantomData,
        }
    }

    fn update_fluid_contacts(
        &mut self,
        kernel_radius: N,
        fluid_fluid_contacts: &mut [ParticlesContacts<N>],
        fluid_boundary_contacts: &mut [ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    )
    {
        for contacts in fluid_fluid_contacts.iter_mut() {
            par_iter_mut!(contacts.contacts_mut()).for_each(|c| {
                let fluid1 = &fluids[c.i_model];
                let fluid2 = &fluids[c.j_model];

                let pi = fluid1.positions[c.i] + self.position_changes[c.i_model][c.i];
                let pj = fluid2.positions[c.j] + self.position_changes[c.j_model][c.j];

                c.weight = KernelDensity::points_apply(&pi, &pj, kernel_radius);
                c.gradient = KernelGradient::points_apply_diff1(&pi, &pj, kernel_radius);
            })
        }

        for contacts in fluid_boundary_contacts.iter_mut() {
            par_iter_mut!(contacts.contacts_mut()).for_each(|c| {
                let fluid1 = &fluids[c.i_model];
                let bound2 = &boundaries[c.j_model];

                let pi = fluid1.positions[c.i] + self.position_changes[c.i_model][c.i];
                let pj = bound2.positions[c.j];

                c.weight = KernelDensity::points_apply(&pi, &pj, kernel_radius);
                c.gradient = KernelGradient::points_apply_diff1(&pi, &pj, kernel_radius);
            })
        }
    }

    fn update_boundary_contacts(
        &mut self,
        kernel_radius: N,
        boundary_boundary_contacts: &mut [ParticlesContacts<N>],
        boundaries: &[Boundary<N>],
    )
    {
        for contacts in boundary_boundary_contacts.iter_mut() {
            par_iter_mut!(contacts.contacts_mut()).for_each(|c| {
                let fluid1 = &boundaries[c.i_model];
                let bound2 = &boundaries[c.j_model];

                let pi = fluid1.positions[c.i];
                let pj = bound2.positions[c.j];

                c.weight = KernelDensity::points_apply(&pi, &pj, kernel_radius);
                c.gradient = KernelGradient::points_apply_diff1(&pi, &pj, kernel_radius);
            })
        }
    }

    fn resize_buffers(&mut self, dt: N, fluids: &[Fluid<N>], boundaries: &[Boundary<N>]) {
        // Resize every buffer.
        self.lambdas.resize(fluids.len(), Vec::new());
        self.densities.resize(fluids.len(), Vec::new());
        self.boundaries_volumes.resize(boundaries.len(), Vec::new());
        self.position_changes.resize(fluids.len(), Vec::new());
        self.nonpressure_forces.resize(fluids.len(), Vec::new());

        for (fluid, lambdas, densities, position_changes, nonpressure_forces) in
            itertools::multizip((
                fluids.iter(),
                self.lambdas.iter_mut(),
                self.densities.iter_mut(),
                self.position_changes.iter_mut(),
                self.nonpressure_forces.iter_mut(),
            ))
        {
            lambdas.resize(fluid.num_particles(), N::zero());
            densities.resize(fluid.num_particles(), N::zero());
            position_changes.resize(fluid.num_particles(), Vector::zeros());
            nonpressure_forces.resize(fluid.num_particles(), Vector::zeros());
        }

        for (boundary, volumes) in boundaries.iter().zip(self.boundaries_volumes.iter_mut()) {
            volumes.resize(boundary.num_particles(), N::zero())
        }
    }

    fn predict_advection(&mut self, dt: N, gravity: &Vector<N>, fluids: &[Fluid<N>]) {
        for (fluid, position_changes) in fluids.iter().zip(self.position_changes.iter_mut()) {
            par_iter_mut!(position_changes)
                .zip(par_iter!(fluid.velocities))
                .for_each(|(position_change, vel)| {
                    *position_change = (vel + gravity * dt) * dt;
                })
        }
    }

    fn compute_boundary_volumes(
        &mut self,
        kernel_radius: N,
        boundary_boundary_contacts: &[ParticlesContacts<N>],
        boundaries: &[Boundary<N>],
    )
    {
        for boundary_id in 0..boundaries.len() {
            par_iter_mut!(self.boundaries_volumes[boundary_id])
                .enumerate()
                .for_each(|(i, volume)| {
                    let mut denominator = N::zero();
                    let pi = boundaries[boundary_id].positions[i];

                    for c in boundary_boundary_contacts[boundary_id].particle_contacts(i) {
                        let pj = boundaries[c.j_model].positions[c.j];
                        denominator += KernelDensity::points_apply(&pi, &pj, kernel_radius);
                    }

                    assert!(!denominator.is_zero());
                    *volume = N::one() / denominator;
                })
        }
    }

    fn compute_densities(
        &mut self,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    )
    {
        let boundaries_volumes = &self.boundaries_volumes;

        for fluid_id in 0..fluids.len() {
            par_iter_mut!(self.densities[fluid_id])
                .enumerate()
                .for_each(|(i, density)| {
                    *density = N::zero();
                    let mut nnz_weights = 0;

                    for c in fluid_fluid_contacts[fluid_id].particle_contacts(i) {
                        *density += fluids[c.j_model].volumes[c.j] * c.weight;
                    }

                    for c in fluid_boundary_contacts[fluid_id].particle_contacts(i) {
                        *density += boundaries_volumes[c.j_model][c.j] * c.weight;
                    }

                    assert!(!density.is_zero());
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
    }
    */

    fn compute_lambdas(
        &mut self,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    )
    {
        let boundaries_volumes = &self.boundaries_volumes;

        for fluid_id in 0..fluids.len() {
            let fluid1 = &fluids[fluid_id];
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let fluid_boundary_contacts = &fluid_boundary_contacts[fluid_id];
            let densities1 = self.densities[fluid_id].as_slice();
            let lambdas1 = &mut self.lambdas[fluid_id];

            par_iter_mut!(lambdas1)
                .enumerate()
                .for_each(|(i, lambda1)| {
                    let density_err = densities1[i] - N::one();

                    if density_err > N::zero() {
                        let mut total_gradient = Vector::zeros();
                        let mut denominator = N::zero();

                        for c in fluid_fluid_contacts.particle_contacts(i) {
                            let grad_i = c.gradient * fluids[c.j_model].volumes[c.j];
                            denominator += grad_i.norm_squared();
                            total_gradient += grad_i;
                        }

                        for c in fluid_boundary_contacts.particle_contacts(i) {
                            let grad_i = c.gradient * boundaries_volumes[c.j_model][c.j];
                            denominator += grad_i.norm_squared();
                            total_gradient += grad_i;
                        }

                        let denominator = denominator + total_gradient.norm_squared();
                        *lambda1 = -density_err / (denominator + na::convert(1.0e-6));
                    } else {
                        *lambda1 = N::zero();
                    }
                })
        }
    }

    fn compute_position_changes(
        &mut self,
        inv_dt: N,
        kernel_radius: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    )
    {
        let densities = &self.densities;
        let lambdas = &self.lambdas;
        let boundaries_volumes = &self.boundaries_volumes;

        for (fluid_id, fluid1) in fluids.iter().enumerate() {
            par_iter_mut!(self.position_changes[fluid_id])
                .enumerate()
                .for_each(|(i, position_change)| {

                for c in fluid_fluid_contacts[fluid_id].particle_contacts(i) {
                    let fluid2 = &fluids[c.j_model];

                    // Compute virtual pressure.
                    let k: N = na::convert(0.001);
                    let n = 4;
                    let dq = N::zero();
                    let scorr = -k * (c.weight / KernelDensity::scalar_apply(dq, kernel_radius)).powi(n);

                    // Compute position change.
                    let coeff = fluid2.volumes[c.j] * (lambdas[c.i_model][c.i] + fluid2.density0 / fluid1.density0 * lambdas[c.j_model][c.j])/* + scorr*/;
                    position_change.axpy(coeff, &c.gradient, N::one());
                }


                for c in fluid_boundary_contacts[fluid_id].particle_contacts(i) {
                    let boundary2 = &boundaries[c.j_model];

                    let lambda = lambdas[c.i_model][c.i];
                    let coeff = boundaries_volumes[c.j_model][c.j] * (lambda + lambda)/* + scorr*/;
                    let delta = c.gradient * coeff;
                    *position_change += delta;

                    // Apply the force to the boundary too.
                    let particle_mass = fluid1.volumes[c.i] * fluid1.density0;
                    boundary2.apply_force(c.j, delta * (-inv_dt * inv_dt * particle_mass));
                }
            })
        }
    }

    fn update_velocities_and_positions(&mut self, inv_dt: N, fluids: &mut [Fluid<N>]) {
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

    fn clear_nonpressure_forces(&mut self, fluids: &mut [Fluid<N>]) {
        for forces in &mut self.nonpressure_forces {
            par_iter_mut!(forces).for_each(|f| f.fill(N::zero()))
        }
    }

    fn integrate_nonpressure_forces(&mut self, dt: N, fluids: &mut [Fluid<N>]) {
        for (fluid, forces) in fluids.iter_mut().zip(self.nonpressure_forces.iter()) {
            let velocities = &mut fluid.velocities;

            par_iter_mut!(velocities)
                .zip(par_iter!(forces))
                .for_each(|(v, f)| *v += f * dt)
        }
    }

    fn apply_viscosity(
        &mut self,
        inv_dt: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluids: &mut [Fluid<N>],
    )
    {
        // Add XSPH viscosity
        for (fluid_id, fluid_i) in fluids.iter().enumerate() {
            let contacts = &fluid_fluid_contacts[fluid_id];
            let velocities = &fluid_i.velocities;
            let forces = &mut self.nonpressure_forces[fluid_id];

            par_iter_mut!(forces).enumerate().for_each(|(i, f)| {
                for c in contacts.particle_contacts(i) {
                    let fluid_j = &fluids[c.j_model];
                    let dvel = fluid_j.velocities[c.j] - fluid_i.velocities[c.i];
                    let extra_vel = dvel * (c.weight * fluid_i.viscosity);

                    *f += extra_vel * inv_dt;
                }
            })
        }
    }

    fn pressure_solve(
        &mut self,
        dt: N,
        inv_dt: N,
        gravity: &Vector<N>,
        kernel_radius: N,
        contact_manager: &mut ContactManager<N>,
        fluids: &mut [Fluid<N>],
        boundaries: &[Boundary<N>],
    )
    {
        let niters = 10;

        for loop_i in 0..niters {
            self.update_fluid_contacts(
                kernel_radius,
                &mut contact_manager.fluid_fluid_contacts,
                &mut contact_manager.fluid_boundary_contacts,
                fluids,
                boundaries,
            );

            self.compute_densities(
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
                boundaries,
            );

            //            println!("Densities: {:?}", self.densities);
            //                let err = Self::compute_average_density_error(self.num_particles, &densities, &self.densities0);

            self.compute_lambdas(
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
                boundaries,
            );

            self.compute_position_changes(
                inv_dt,
                kernel_radius,
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
                boundaries,
            );
        }

        // Compute actual velocities.
        self.update_velocities_and_positions(N::one() / dt, fluids);
    }

    fn nonpressure_solve(
        &mut self,
        dt: N,
        inv_dt: N,
        contact_manager: &mut ContactManager<N>,
        fluids: &mut [Fluid<N>],
    )
    {
        // Nonpressure forces.
        self.clear_nonpressure_forces(fluids);
        self.apply_viscosity(inv_dt, &contact_manager.fluid_fluid_contacts, fluids);
        self.integrate_nonpressure_forces(dt, fluids);
    }

    pub fn step(
        &mut self,
        dt: N,
        timestep_manager: &TimestepManager<N>,
        contact_manager: &mut ContactManager<N>,
        gravity: &Vector<N>,
        kernel_radius: N,
        particle_radius: N,
        fluids: &mut [Fluid<N>],
        boundaries: &[Boundary<N>],
    )
    {
        let mut remaining_time = dt;

        // Init buffers.
        self.resize_buffers(dt, fluids, boundaries);

        // Perform substeps.
        while remaining_time > N::zero() {
            // Substep length.
            let substep_dt = timestep_manager.compute_substep(
                dt,
                remaining_time,
                particle_radius,
                fluids,
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
            );

            let substep_inv_dt = N::one() / substep_dt;

            contact_manager.update_contacts(
                kernel_radius,
                fluids,
                boundaries,
                Some(&self.position_changes),
            );

            self.predict_advection(substep_dt, gravity, fluids);

            // Init boundary-related data.
            self.update_boundary_contacts(
                kernel_radius,
                &mut contact_manager.boundary_boundary_contacts,
                boundaries,
            );
            self.compute_boundary_volumes(
                kernel_radius,
                &contact_manager.boundary_boundary_contacts,
                boundaries,
            );

            let solver_start_time = instant::now();
            self.pressure_solve(
                substep_dt,
                substep_inv_dt,
                gravity,
                kernel_radius,
                contact_manager,
                fluids,
                boundaries,
            );

            self.nonpressure_solve(dt, substep_inv_dt, contact_manager, fluids);

            remaining_time -= substep_dt;
            println!("Performed substep: {}", substep_dt);
        }
    }
}
