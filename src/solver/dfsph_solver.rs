use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use na::{self, RealField};

use crate::geometry::{ContactManager, ParticlesContacts};
use crate::kernel::{CubicSplineKernel, Kernel, Poly6Kernel, SpikyKernel};
use crate::math::Vector;
use crate::object::{Boundary, Fluid};

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

/// A Position Based Fluid solver.
pub struct DFSPHSolver<
    N: RealField,
    KernelDensity: Kernel = Poly6Kernel,
    KernelGradient: Kernel = SpikyKernel,
> {
    alphas: Vec<Vec<N>>,
    densities: Vec<Vec<N>>,
    boundaries_volumes: Vec<Vec<N>>,
    velocity_changes: Vec<Vec<Vector<N>>>,
    nonpressure_forces: Vec<Vec<Vector<N>>>,
    phantoms: PhantomData<(KernelDensity, KernelGradient)>,
}

impl<N, KernelDensity, KernelGradient> DFSPHSolver<N, KernelDensity, KernelGradient>
where
    N: RealField,
    KernelDensity: Kernel,
    KernelGradient: Kernel,
{
    /// Initialize a new Position Based Fluid solver.
    pub fn new() -> Self {
        Self {
            alphas: Vec::new(),
            densities: Vec::new(),
            boundaries_volumes: Vec::new(),
            velocity_changes: Vec::new(),
            nonpressure_forces: Vec::new(),
            phantoms: PhantomData,
        }
    }

    fn update_fluid_contacts(
        &mut self,
        dt: N,
        kernel_radius: N,
        fluid_fluid_contacts: &mut [ParticlesContacts<N>],
        fluid_boundary_contacts: &mut [ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        for contacts in fluid_fluid_contacts.iter_mut() {
            par_iter_mut!(contacts.contacts_mut()).for_each(|c| {
                let fluid1 = &fluids[c.i_model];
                let fluid2 = &fluids[c.j_model];

                let pi = fluid1.positions[c.i]
                    + (fluid1.velocities[c.i] + self.velocity_changes[c.i_model][c.i]) * dt;
                let pj = fluid2.positions[c.j]
                    + (fluid2.velocities[c.j] + self.velocity_changes[c.j_model][c.j]) * dt;

                c.weight = KernelDensity::points_apply(&pi, &pj, kernel_radius);
                c.gradient = KernelGradient::points_apply_diff1(&pi, &pj, kernel_radius);
            })
        }

        for contacts in fluid_boundary_contacts.iter_mut() {
            par_iter_mut!(contacts.contacts_mut()).for_each(|c| {
                let fluid1 = &fluids[c.i_model];
                let bound2 = &boundaries[c.j_model];

                let pi = fluid1.positions[c.i]
                    + (fluid1.velocities[c.i] + self.velocity_changes[c.i_model][c.i]) * dt;
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
    ) {
        for contacts in boundary_boundary_contacts.iter_mut() {
            par_iter_mut!(contacts.contacts_mut()).for_each(|c| {
                let bound1 = &boundaries[c.i_model];
                let bound2 = &boundaries[c.j_model];

                let pi = bound1.positions[c.i];
                let pj = bound2.positions[c.j];

                c.weight = KernelDensity::points_apply(&pi, &pj, kernel_radius);
                c.gradient = KernelGradient::points_apply_diff1(&pi, &pj, kernel_radius);
            })
        }
    }

    /// Gets the set of fluid particle velocity changes resulting from pressure resolution.
    pub fn velocity_changes(&self) -> &[Vec<Vector<N>>] {
        &self.velocity_changes
    }

    /// Gets a mutable reference to the set of fluid particle velocity changes resulting from
    /// pressure resolution.
    pub fn velocity_changes_mut(&mut self) -> &mut [Vec<Vector<N>>] {
        &mut self.velocity_changes
    }

    /// Initialize this solver with the given fluids.
    pub fn init_with_fluids(&mut self, fluids: &[Fluid<N>]) {
        // Resize every buffer.
        self.alphas.resize(fluids.len(), Vec::new());
        self.densities.resize(fluids.len(), Vec::new());
        self.velocity_changes.resize(fluids.len(), Vec::new());
        self.nonpressure_forces.resize(fluids.len(), Vec::new());

        for (fluid, alphas, densities, velocity_changes, nonpressure_forces) in
            itertools::multizip((
                fluids.iter(),
                self.alphas.iter_mut(),
                self.densities.iter_mut(),
                self.velocity_changes.iter_mut(),
                self.nonpressure_forces.iter_mut(),
            ))
        {
            alphas.resize(fluid.num_particles(), N::zero());
            densities.resize(fluid.num_particles(), N::zero());
            velocity_changes.resize(fluid.num_particles(), Vector::zeros());
            nonpressure_forces.resize(fluid.num_particles(), Vector::zeros());
        }
    }

    /// Initialize this solver with the given boundaries.
    pub fn init_with_boundaries(&mut self, boundaries: &[Boundary<N>]) {
        self.boundaries_volumes.resize(boundaries.len(), Vec::new());

        for (boundary, volumes) in boundaries.iter().zip(self.boundaries_volumes.iter_mut()) {
            volumes.resize(boundary.num_particles(), N::zero())
        }
    }

    fn compute_boundary_volumes(
        &mut self,
        boundary_boundary_contacts: &[ParticlesContacts<N>],
        boundaries: &[Boundary<N>],
    ) {
        for boundary_id in 0..boundaries.len() {
            par_iter_mut!(self.boundaries_volumes[boundary_id])
                .enumerate()
                .for_each(|(i, volume)| {
                    let mut denominator = N::zero();

                    for c in boundary_boundary_contacts[boundary_id].particle_contacts(i) {
                        denominator += c.weight;
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
    ) {
        let boundaries_volumes = &self.boundaries_volumes;

        for fluid_id in 0..fluids.len() {
            par_iter_mut!(self.densities[fluid_id])
                .enumerate()
                .for_each(|(i, density)| {
                    *density = N::zero();

                    for c in fluid_fluid_contacts[fluid_id].particle_contacts(i) {
                        *density += fluids[c.j_model].particle_mass(c.j) * c.weight;
                    }

                    for c in fluid_boundary_contacts[fluid_id].particle_contacts(i) {
                        *density += boundaries_volumes[c.j_model][c.j]
                            * fluids[c.i_model].density0
                            * c.weight;
                    }

                    assert!(!density.is_zero());
                })
        }
    }

    /// Predicts advection with the given gravity.
    pub fn predict_advection(&mut self, dt: N, gravity: &Vector<N>, fluids: &[Fluid<N>]) {
        for (fluid, velocity_changes) in fluids.iter().zip(self.velocity_changes.iter_mut()) {
            par_iter_mut!(velocity_changes).for_each(|velocity_change| {
                *velocity_change = gravity * dt;
            })
        }
    }

    fn compute_alphas(
        &mut self,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
    ) {
        let boundaries_volumes = &self.boundaries_volumes;

        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let fluid_boundary_contacts = &fluid_boundary_contacts[fluid_id];
            let densities1 = self.densities[fluid_id].as_slice();
            let alphas1 = &mut self.alphas[fluid_id];
            let fluid_i = &fluids[fluid_id];

            par_iter_mut!(alphas1).enumerate().for_each(|(i, alpha1)| {
                if densities1[i] > fluid_i.density0 {
                    let mut total_gradient = Vector::zeros();
                    let mut denominator = N::zero();
                    let mut divergence = N::zero();

                    for c in fluid_fluid_contacts.particle_contacts(i) {
                        let grad_i = c.gradient * fluids[c.j_model].particle_mass(c.j);
                        denominator += grad_i.norm_squared();
                        total_gradient += grad_i;
                    }

                    for c in fluid_boundary_contacts.particle_contacts(i) {
                        let grad_i =
                            c.gradient * boundaries_volumes[c.j_model][c.j] * fluid_i.density0;
                        denominator += grad_i.norm_squared();
                        total_gradient += grad_i;
                    }

                    let denominator = denominator + total_gradient.norm_squared();
                    *alpha1 = N::one() / denominator.max(na::convert(1.0e-6));
                } else {
                    *alpha1 = N::zero();
                }
            })
        }
    }

    /*
    fn compute_divergence(
        &mut self,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
    ) {
        let boundaries_volumes = &self.boundaries_volumes;

        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let fluid_boundary_contacts = &fluid_boundary_contacts[fluid_id];
            let densities1 = self.densities[fluid_id].as_slice();
            let divergences1 = &mut self.divergences[fluid_id];
            let fluid_i = &fluids[fluid_id];

            par_iter_mut!(divergences1)
                .enumerate()
                .for_each(|(i, divergence1)| {
                    if densities1[i] > fluid_i.density0 {
                        let mut total_gradient = Vector::zeros();
                        let mut denominator = N::zero();
                        let mut divergence = N::zero();

                        for c in fluid_fluid_contacts.particle_contacts(i) {
                            let grad_i = c.gradient * fluids[c.j_model].particle_mass(c.j);
                            let dvel = fluid_i.velocities[c.i] - fluids[c.j_model].velocities[c.j];
                            divergence += dvel.dot(&grad_i);
                        }

                        //                    for c in fluid_boundary_contacts.particle_contacts(i) {
                        //                        let grad_i =
                        //                            c.gradient * boundaries_volumes[c.j_model][c.j] * fluid_i.density0;
                        //                        denominator += grad_i.norm_squared();
                        //                        total_gradient += grad_i;
                        //                    }

                        *divergence1 *= divergence
                    } else {
                        *divergence1 = N::zero();
                    }
                })
        }
    }
    */

    fn compute_velocity_changes(
        &mut self,
        inv_dt: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        let alphas = &self.alphas;
        let boundaries_volumes = &self.boundaries_volumes;
        let densities = &self.densities;

        for (fluid_id, fluid1) in fluids.iter().enumerate() {
            par_iter_mut!(self.velocity_changes[fluid_id])
                .enumerate()
                .for_each(|(i, velocity_change)| {
                    for c in fluid_fluid_contacts[fluid_id].particle_contacts(i) {
                        let fluid1 = &fluids[c.i_model];
                        let fluid2 = &fluids[c.j_model];

                        let ki =
                            (densities[c.i_model][c.i] - fluid1.density0) * alphas[c.i_model][c.i];

                        let kj =
                            (densities[c.j_model][c.j] - fluid2.density0) * alphas[c.j_model][c.j];

                        // Compute velocity change.
                        let coeff = -(ki + kj) * fluid2.particle_mass(c.j);
                        *velocity_change += c.gradient * (coeff * inv_dt);
                    }

                    for c in fluid_boundary_contacts[fluid_id].particle_contacts(i) {
                        let boundary2 = &boundaries[c.j_model];

                        let ki =
                            (densities[c.i_model][c.i] - fluid1.density0) * alphas[c.i_model][c.i];
                        let coeff =
                            -(ki + ki) * boundaries_volumes[c.j_model][c.j] * fluid1.density0;
                        let delta = c.gradient * (coeff * inv_dt);
                        *velocity_change += delta;

                        // Apply the force to the boundary too.
                        let particle_mass = fluid1.volumes[c.i] * fluid1.density0;
                        boundary2.apply_force(c.j, delta * (-inv_dt * particle_mass));
                    }
                })
        }
    }

    fn update_velocities_and_positions(&mut self, dt: N, fluids: &mut [Fluid<N>]) {
        for (fluid, delta) in fluids.iter_mut().zip(self.velocity_changes.iter()) {
            par_iter_mut!(fluid.positions)
                .zip(par_iter_mut!(fluid.velocities))
                .zip(par_iter!(delta))
                .for_each(|((pos, vel), delta)| {
                    *vel += delta;
                    *pos += *vel * dt;
                })
        }
    }

    fn clear_nonpressure_forces(&mut self) {
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
    ) {
        // Add XSPH viscosity
        for (fluid_id, fluid_i) in fluids.iter().enumerate() {
            let contacts = &fluid_fluid_contacts[fluid_id];
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
        kernel_radius: N,
        contact_manager: &mut ContactManager<N>,
        fluids: &mut [Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        let niters = 10;

        for _ in 0..niters {
            self.update_fluid_contacts(
                dt,
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
            );

            //            println!("Densities: {:?}", self.densities);
            //                let err = Self::compute_average_density_error(self.num_particles, &densities, &self.densities0);

            self.compute_alphas(
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
            );

            self.compute_velocity_changes(
                inv_dt,
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
                boundaries,
            );
        }

        // Compute actual velocities.
        self.update_velocities_and_positions(dt, fluids);
    }

    /*
    fn divergence_solve(
        &mut self,
        dt: N,
        inv_dt: N,
        kernel_radius: N,
        contact_manager: &mut ContactManager<N>,
        fluids: &mut [Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        let niters = 10;

        for _ in 0..niters {
            self.update_fluid_contacts(
                dt,
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
            );

            //            println!("Densities: {:?}", self.densities);
            //                let err = Self::compute_average_density_error(self.num_particles, &densities, &self.densities0);

            self.compute_alphas(
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
            );

            self.compute_velocity_changes(
                inv_dt,
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
                boundaries,
            );
        }

        // Compute actual velocities.
        self.update_velocities_and_positions(N::one() / dt, fluids);
    }
    */

    fn nonpressure_solve(
        &mut self,
        dt: N,
        inv_dt: N,
        contact_manager: &mut ContactManager<N>,
        fluids: &mut [Fluid<N>],
    ) {
        // Nonpressure forces.
        self.clear_nonpressure_forces();
        self.apply_viscosity(inv_dt, &contact_manager.fluid_fluid_contacts, fluids);
        self.integrate_nonpressure_forces(dt, fluids);
    }

    /// Solves pressure and non-pressure force for the given fluids and boundaries.
    ///
    /// Both `self.init_with_fluids` and `self.init_with_boundaries` must be called before this
    /// method.
    pub fn step(
        &mut self,
        dt: N,
        contact_manager: &mut ContactManager<N>,
        kernel_radius: N,
        fluids: &mut [Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        let inv_dt = N::one() / dt;

        // Init boundary-related data.
        self.update_boundary_contacts(
            kernel_radius,
            &mut contact_manager.boundary_boundary_contacts,
            boundaries,
        );
        self.compute_boundary_volumes(&contact_manager.boundary_boundary_contacts, boundaries);

        self.pressure_solve(
            dt,
            inv_dt,
            kernel_radius,
            contact_manager,
            fluids,
            boundaries,
        );

        self.nonpressure_solve(dt, inv_dt, contact_manager, fluids);
    }
}
