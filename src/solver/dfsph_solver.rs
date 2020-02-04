use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use na::{self, RealField};

use crate::geometry::{ContactManager, ParticlesContacts};
use crate::kernel::{CubicSplineKernel, Kernel, Poly6Kernel, SpikyKernel};
use crate::math::{Vector, DIM};
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
    fn new() -> Self {
        Self {
            target: StrainRate::zeros(),
            error: StrainRate::zeros(),
        }
    }
}

/// A Position Based Fluid solver.
pub struct DFSPHSolver<
    N: RealField,
    KernelDensity: Kernel = CubicSplineKernel, // Poly6Kernel,  // CubicSplineKernel
    KernelGradient: Kernel = CubicSplineKernel, // SpikyKernel, // CubicSplineKernel
> {
    min_pressure_iter: usize,
    max_pressure_iter: usize,
    max_density_error: N,
    min_divergence_iter: usize,
    max_divergence_iter: usize,
    max_divergence_error: N,
    min_neighbors_for_divergence_solve: usize,
    min_viscosity_iter: usize,
    max_viscosity_iter: usize,
    max_viscosity_error: N,
    alphas: Vec<Vec<N>>,
    betas: Vec<Vec<BetaMatrix<N>>>,
    strain_rates: Vec<Vec<StrainRates<N>>>, // Contains (target rate, error rate)
    densities: Vec<Vec<N>>,
    predicted_densities: Vec<Vec<N>>,
    divergences: Vec<Vec<N>>,
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
            min_pressure_iter: 1,
            max_pressure_iter: 50,
            max_density_error: na::convert(0.05),
            min_divergence_iter: 1,
            max_divergence_iter: 50,
            max_divergence_error: na::convert(0.1),
            min_viscosity_iter: 1,
            max_viscosity_iter: 50,
            max_viscosity_error: na::convert(0.01),
            min_neighbors_for_divergence_solve: if DIM == 2 { 6 } else { 20 },
            alphas: Vec::new(),
            betas: Vec::new(),
            strain_rates: Vec::new(),
            densities: Vec::new(),
            predicted_densities: Vec::new(),
            divergences: Vec::new(),
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
        let velocity_changes = &self.velocity_changes;
        for contacts in fluid_fluid_contacts.iter_mut() {
            par_iter_mut!(contacts.contacts_mut()).for_each(|c| {
                let fluid1 = &fluids[c.i_model];
                let fluid2 = &fluids[c.j_model];
                let pi = fluid1.positions[c.i];
                let pj = fluid2.positions[c.j];

                c.weight = KernelDensity::points_apply(&pi, &pj, kernel_radius);
                c.gradient = KernelGradient::points_apply_diff1(&pi, &pj, kernel_radius);
            })
        }

        for contacts in fluid_boundary_contacts.iter_mut() {
            par_iter_mut!(contacts.contacts_mut()).for_each(|c| {
                let fluid1 = &fluids[c.i_model];
                let bound2 = &boundaries[c.j_model];

                let pi = fluid1.positions[c.i];
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
        self.betas.resize(fluids.len(), Vec::new());
        self.strain_rates.resize(fluids.len(), Vec::new());
        self.densities.resize(fluids.len(), Vec::new());
        self.predicted_densities.resize(fluids.len(), Vec::new());
        self.divergences.resize(fluids.len(), Vec::new());
        self.velocity_changes.resize(fluids.len(), Vec::new());
        self.nonpressure_forces.resize(fluids.len(), Vec::new());

        for (
            fluid,
            (
                alphas,
                betas,
                strain_rates,
                densities,
                predicted_densities,
                divergences,
                velocity_changes,
                nonpressure_forces,
            ),
        ) in fluids.iter().zip(itertools::multizip((
            self.alphas.iter_mut(),
            self.betas.iter_mut(),
            self.strain_rates.iter_mut(),
            self.densities.iter_mut(),
            self.predicted_densities.iter_mut(),
            self.divergences.iter_mut(),
            self.velocity_changes.iter_mut(),
            self.nonpressure_forces.iter_mut(),
        ))) {
            alphas.resize(fluid.num_particles(), N::zero());
            betas.resize(fluid.num_particles(), BetaMatrix::zeros());
            strain_rates.resize(fluid.num_particles(), StrainRates::new());
            densities.resize(fluid.num_particles(), N::zero());
            predicted_densities.resize(fluid.num_particles(), N::zero());
            divergences.resize(fluid.num_particles(), N::zero());
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
                        *density += fluids[c.j_model].volumes[c.j] * c.weight;
                    }

                    for c in fluid_boundary_contacts[fluid_id].particle_contacts(i) {
                        *density += boundaries_volumes[c.j_model][c.j] * c.weight;
                    }

                    assert!(!density.is_zero());
                })
        }
    }

    fn compute_predicted_densities(
        &mut self,
        dt: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
    ) -> N {
        let boundaries_volumes = &self.boundaries_volumes;
        let velocity_changes = &self.velocity_changes;
        let densities = &self.densities;
        let mut max_error = N::zero();

        for fluid_id in 0..fluids.len() {
            let err = par_iter_mut!(self.predicted_densities[fluid_id])
                .enumerate()
                .fold(N::zero(), |curr_err, (i, density)| {
                    let fluid_i = &fluids[fluid_id];
                    let mut delta = N::zero();

                    for c in fluid_fluid_contacts[fluid_id].particle_contacts(i) {
                        let fluid_j = &fluids[c.j_model];
                        let vi = fluid_i.velocities[c.i] + velocity_changes[c.i_model][c.i];
                        let vj = fluid_j.velocities[c.j] + velocity_changes[c.j_model][c.j];

                        delta += fluids[c.j_model].volumes[c.j] * (vi - vj).dot(&c.gradient);
                    }

                    for c in fluid_boundary_contacts[fluid_id].particle_contacts(i) {
                        let vi = fluid_i.velocities[c.i] + velocity_changes[c.i_model][c.i];
                        // FIXME: take the velocity of j too?

                        delta += boundaries_volumes[c.j_model][c.j] * vi.dot(&c.gradient);
                    }

                    *density = densities[fluid_id][i] + delta * dt;
                    *density = density.max(N::one());
                    assert!(!density.is_zero());
                    curr_err + *density - N::one()
                });

            let nparts = fluids[fluid_id].num_particles();
            if nparts != 0 {
                max_error = max_error.max(err / na::convert(nparts as f64));
            }
        }

        max_error
    }

    /// Predicts advection with the given gravity.
    pub fn predict_advection(&mut self, dt: N, gravity: &Vector<N>, fluids: &[Fluid<N>]) {
        for (fluid, velocity_changes) in fluids.iter().zip(self.velocity_changes.iter_mut()) {
            par_iter_mut!(velocity_changes).for_each(|velocity_change| {
                *velocity_change += gravity * dt;
            })
        }
    }

    // NOTE: this actually computes alpha_i / density_i
    fn compute_alphas(
        &mut self,
        inv_dt: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
    ) {
        let boundaries_volumes = &self.boundaries_volumes;

        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let fluid_boundary_contacts = &fluid_boundary_contacts[fluid_id];
            let alphas_i = &mut self.alphas[fluid_id];
            let fluid_i = &fluids[fluid_id];

            par_iter_mut!(alphas_i)
                .enumerate()
                .for_each(|(i, alpha_i)| {
                    let mut grad_sum = Vector::zeros();
                    let mut squared_grad_sum = N::zero();

                    for c in fluid_fluid_contacts.particle_contacts(i) {
                        let grad_i = c.gradient * fluids[c.j_model].volumes[c.j];
                        squared_grad_sum += grad_i.norm_squared();
                        grad_sum += grad_i;
                    }

                    for c in fluid_boundary_contacts.particle_contacts(i) {
                        let grad_i = c.gradient * boundaries_volumes[c.j_model][c.j];
                        squared_grad_sum += grad_i.norm_squared();
                        grad_sum += grad_i;
                    }

                    let denominator = squared_grad_sum + grad_sum.norm_squared();
                    *alpha_i = N::one() / denominator.max(na::convert(1.0e-6));
                })
        }
    }

    fn compute_divergences(
        &mut self,
        dt: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
    ) -> N {
        let boundaries_volumes = &self.boundaries_volumes;
        let velocity_changes = &self.velocity_changes;
        let min_neighbors_for_divergence_solve = self.min_neighbors_for_divergence_solve;
        let mut max_error = N::zero();

        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let fluid_boundary_contacts = &fluid_boundary_contacts[fluid_id];
            let divergences_i = &mut self.divergences[fluid_id];
            let fluid_i = &fluids[fluid_id];

            let err = par_iter_mut!(divergences_i).enumerate().fold(
                N::zero(),
                |curr_err, (i, divergence_i)| {
                    *divergence_i = N::zero();

                    if fluid_fluid_contacts.particle_contacts(i).len()
                        + fluid_boundary_contacts.particle_contacts(i).len()
                        < min_neighbors_for_divergence_solve
                    {
                        return curr_err;
                    }

                    for c in fluid_fluid_contacts.particle_contacts(i) {
                        let fluid_j = &fluids[c.j_model];
                        let v_i = fluid_i.velocities[c.i] + velocity_changes[c.i_model][c.i];
                        let v_j = fluid_j.velocities[c.j] + velocity_changes[c.j_model][c.j];
                        let dvel = v_i - v_j;
                        *divergence_i += dvel.dot(&c.gradient) * fluid_j.volumes[c.j];
                    }

                    for c in fluid_boundary_contacts.particle_contacts(i) {
                        let v_i = fluid_i.velocities[c.i] + velocity_changes[c.i_model][c.i];
                        // FIXME: take the velocity of j too?

                        let dvel = v_i;
                        *divergence_i += dvel.dot(&c.gradient) * boundaries_volumes[c.j_model][c.j];
                    }

                    *divergence_i = divergence_i.max(N::zero());
                    curr_err + *divergence_i
                },
            );

            let nparts = fluids[fluid_id].num_particles();
            if nparts != 0 {
                max_error = max_error.max(err / na::convert(nparts as f64));
            }
        }

        max_error
    }

    fn compute_velocity_changes(
        &mut self,
        dt: N,
        inv_dt: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        let alphas = &self.alphas;
        let boundaries_volumes = &self.boundaries_volumes;
        let predicted_densities = &self.predicted_densities;

        for (fluid_id, fluid1) in fluids.iter().enumerate() {
            par_iter_mut!(self.velocity_changes[fluid_id])
                .enumerate()
                .for_each(|(i, velocity_change)| {
                    let ki = (predicted_densities[fluid_id][i] - N::one()) * alphas[fluid_id][i];

                    for c in fluid_fluid_contacts[fluid_id].particle_contacts(i) {
                        let fluid1 = &fluids[c.i_model];
                        let fluid2 = &fluids[c.j_model];

                        let kj = (predicted_densities[c.j_model][c.j] - N::one())
                            * alphas[c.j_model][c.j];

                        let kij = ki + kj * fluid2.density0 / fluid1.density0;

                        // Compute velocity change.
                        if kij > N::default_epsilon() {
                            let coeff = kij * fluid2.volumes[c.j];
                            *velocity_change -= c.gradient * (coeff * inv_dt);
                        }
                    }

                    if ki > N::default_epsilon() {
                        for c in fluid_boundary_contacts[fluid_id].particle_contacts(i) {
                            let coeff = ki * boundaries_volumes[c.j_model][c.j];
                            let delta = c.gradient * (coeff * inv_dt);

                            *velocity_change -= delta;

                            // Apply the force to the boundary too.
                            let particle_mass = fluid1.volumes[c.i] * fluid1.density0;
                            boundaries[c.j_model]
                                .apply_force(c.j, delta * (inv_dt * particle_mass));
                        }
                    }
                })
        }
    }

    fn compute_velocity_changes_for_divergence(
        &mut self,
        dt: N,
        inv_dt: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        let alphas = &self.alphas;
        let boundaries_volumes = &self.boundaries_volumes;
        let divergences = &self.divergences;

        for (fluid_id, fluid1) in fluids.iter().enumerate() {
            par_iter_mut!(self.velocity_changes[fluid_id])
                .enumerate()
                .for_each(|(i, velocity_change)| {
                    let ki = divergences[fluid_id][i] * alphas[fluid_id][i];

                    for c in fluid_fluid_contacts[fluid_id].particle_contacts(i) {
                        let fluid2 = &fluids[c.j_model];
                        let kj =
                            divergences[c.j_model][c.j] * alphas[c.j_model][c.j] * fluid2.density0
                                / fluid1.density0;

                        // Compute velocity change.
                        let coeff = -(ki + kj) * fluid2.volumes[c.j];
                        *velocity_change += c.gradient * coeff;
                    }

                    for c in fluid_boundary_contacts[fluid_id].particle_contacts(i) {
                        let boundary2 = &boundaries[c.j_model];

                        // Compute velocity change.
                        let coeff = -ki * boundaries_volumes[c.j_model][c.j];
                        let delta = c.gradient * coeff;
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

    fn update_positions(&mut self, dt: N, fluids: &mut [Fluid<N>]) {
        for (fluid, delta) in fluids.iter_mut().zip(self.velocity_changes.iter()) {
            par_iter_mut!(fluid.positions)
                .zip(par_iter!(fluid.velocities))
                .zip(par_iter!(delta))
                .for_each(|((pos, vel), delta)| {
                    *pos += (*vel + delta) * dt;
                })
        }
    }

    fn update_velocities(&mut self, dt: N, fluids: &mut [Fluid<N>]) {
        for (fluid, delta) in fluids.iter_mut().zip(self.velocity_changes.iter()) {
            par_iter_mut!(fluid.velocities)
                .zip(par_iter!(delta))
                .for_each(|(vel, delta)| {
                    *vel += delta;
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
        for i in 0..self.max_pressure_iter {
            let avg_err = self.compute_predicted_densities(
                dt,
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
            );

            if avg_err <= self.max_density_error && i >= self.min_pressure_iter {
                println!(
                    "Average density error: {}, break after niters: {}",
                    avg_err, i
                );
                break;
            }

            self.compute_velocity_changes(
                dt,
                inv_dt,
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
                boundaries,
            );
        }
    }

    fn divergence_solve(
        &mut self,
        dt: N,
        inv_dt: N,
        kernel_radius: N,
        contact_manager: &mut ContactManager<N>,
        fluids: &mut [Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        for i in 0..self.max_divergence_iter {
            let avg_err = self.compute_divergences(
                dt,
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
            );

            if avg_err <= self.max_divergence_error && i >= self.min_divergence_iter {
                println!(
                    "Average divergence error: {}, break after niters: {}",
                    avg_err, i
                );
                break;
            }

            self.compute_velocity_changes_for_divergence(
                dt,
                inv_dt,
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
                boundaries,
            );
        }
    }

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

    // NOTE: this actually computes beta / density_i^3
    fn compute_betas(
        &mut self,
        inv_dt: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
    ) {
        let boundaries_volumes = &self.boundaries_volumes;
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
                    #[cfg(feature = "dim2")]
                    #[rustfmt::skip]
                    let mat = BetaGradientMatrix::new(
                        c.gradient.x * _2, N::zero(),
                        N::zero(), c.gradient.y * _2,
                        c.gradient.y, c.gradient.x,
                    );

                    #[cfg(feature = "dim3")]
                    #[rustfmt::skip]
                    let mat = BetaGradientMatrix::new(
                        c.gradient.x * _2, N::zero(), N::zero(),
                        N::zero(), c.gradient.y * _2, N::zero(),
                        N::zero(), N::zero(), c.gradient.z,
                        c.gradient.y, c.gradient.x, N::zero(),
                        c.gradient.z, N::zero(), c.gradient.x,
                        N::zero(), c.gradient.z, c.gradient.y,
                    );

                    let grad_i = mat * (fluids[c.j_model].volumes[c.j] / _2);
                    squared_grad_sum += grad_i * grad_i.transpose();
                    grad_sum += grad_i;
                }

                //                for c in fluid_boundary_contacts.particle_contacts(i) {
                //                    let grad_i = c.gradient * boundaries_volumes[c.j_model][c.j];
                //                    squared_grad_sum += grad_i.norm_squared();
                //                    grad_sum += grad_i;
                //                }

                let denominator = squared_grad_sum + grad_sum * grad_sum.transpose();
                *beta_i = denominator
                    .try_inverse()
                    .unwrap_or_else(|| BetaMatrix::zeros());
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
        compute_error: bool,
    ) -> N {
        let boundaries_volumes = &self.boundaries_volumes;
        let velocity_changes = &self.velocity_changes;
        let min_neighbors_for_divergence_solve = self.min_neighbors_for_divergence_solve;
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
                    let out_rate = if compute_error {
                        &mut strain_rates_i.error
                    } else {
                        &mut strain_rates_i.target
                    };

                    *out_rate = StrainRate::zeros();

                    for c in fluid_fluid_contacts.particle_contacts(i) {
                        let fluid_j = &fluids[c.j_model];
                        let v_i = fluid_i.velocities[c.i] + velocity_changes[c.i_model][c.i];
                        let v_j = fluid_j.velocities[c.j] + velocity_changes[c.j_model][c.j];
                        let v_ji = v_j - v_i;
                        #[cfg(feature = "dim3")]
                        let rate = StrainRate::new(
                            _2 * v_ji.x * c.gradient.x,
                            _2 * v_ji.y * c.gradient.y,
                            _2 * v_ji.z * c.gradient.z,
                            v_ji.x * c.gradient.y + v_ji.y * c.gradient.x,
                            v_ji.x * c.gradient.z + v_ji.z * c.gradient.x,
                            v_ji.y * c.gradient.z + v_ji.z * c.gradient.y,
                        );
                        #[cfg(feature = "dim2")]
                        let rate = StrainRate::new(
                            _2 * v_ji.x * c.gradient.x,
                            _2 * v_ji.y * c.gradient.y,
                            v_ji.x * c.gradient.y + v_ji.y * c.gradient.x,
                        );

                        *out_rate += rate * (fluid_j.volumes[c.j] / _2);
                    }

                    //                    for c in fluid_boundary_contacts.particle_contacts(i) {
                    //                        let v_i = fluid_i.velocities[c.i] + velocity_changes[c.i_model][c.i];
                    //                        // FIXME: take the velocity of j too?
                    //
                    //                        let dvel = v_i;
                    //                        *strain_rates_i +=
                    //                            dvel.dot(&c.gradient) * boundaries_volumes[c.j_model][c.j];
                    //                    }

                    if compute_error {
                        strain_rates_i.error -= strain_rates_i.target;
                        curr_err + strain_rates_i.error.norm_squared()
                    } else {
                        strain_rates_i.target *= fluid_i.viscosity;
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
        &mut self,
        dt: N,
        inv_dt: N,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        let alphas = &self.alphas;
        let boundaries_volumes = &self.boundaries_volumes;
        let strain_rates = &self.strain_rates;
        let betas = &self.betas;
        let _2: N = na::convert(2.0);

        for (fluid_id, fluid1) in fluids.iter().enumerate() {
            par_iter_mut!(self.velocity_changes[fluid_id])
                .enumerate()
                .for_each(|(i, velocity_change)| {
                    let ui = betas[fluid_id][i] * strain_rates[fluid_id][i].error;

                    for c in fluid_fluid_contacts[fluid_id].particle_contacts(i) {
                        let fluid2 = &fluids[c.j_model];
                        let uj = betas[c.j_model][c.j]
                            * strain_rates[c.j_model][c.j].error
                            * (fluid2.density0 / fluid1.density0);

                        #[cfg(feature = "dim2")]
                        #[rustfmt::skip]
                        let gradient = BetaGradientMatrix::new(
                            c.gradient.x * _2, N::zero(),
                            N::zero(), c.gradient.y * _2,
                            c.gradient.y, c.gradient.x,
                        );

                        #[cfg(feature = "dim3")]
                        #[rustfmt::skip]
                        let gradient = BetaGradientMatrix::new(
                            c.gradient.x * _2, N::zero(), N::zero(),
                            N::zero(), c.gradient.y * _2, N::zero(),
                            N::zero(), N::zero(), c.gradient.z,
                            c.gradient.y, c.gradient.x, N::zero(),
                            c.gradient.z, N::zero(), c.gradient.x,
                            N::zero(), c.gradient.z, c.gradient.y,
                        );

                        // Compute velocity change.
                        let coeff = -(ui + uj) * (fluid2.volumes[c.j] / _2);
                        *velocity_change += gradient.tr_mul(&coeff);
                    }

                    //                    for c in fluid_boundary_contacts[fluid_id].particle_contacts(i) {
                    //                        let boundary2 = &boundaries[c.j_model];
                    //
                    //                        // Compute velocity change.
                    //                        let coeff = -ui * boundaries_volumes[c.j_model][c.j];
                    //                        let delta = c.gradient * coeff;
                    //                        *velocity_change += delta;
                    //
                    //                        // Apply the force to the boundary too.
                    //                        let particle_mass = fluid1.volumes[c.i] * fluid1.density0;
                    //                        boundary2.apply_force(c.j, delta * (-inv_dt * particle_mass));
                    //                    }
                })
        }
    }

    fn viscosity_solve(
        &mut self,
        dt: N,
        inv_dt: N,
        kernel_radius: N,
        contact_manager: &mut ContactManager<N>,
        fluids: &mut [Fluid<N>],
        boundaries: &[Boundary<N>],
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
            false,
        );

        for i in 0..self.max_viscosity_iter {
            let avg_err = self.compute_strain_rates(
                dt,
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
                true,
            );
            println!(
                "Average viscosity error: {}, break after niters: {}",
                avg_err, i
            );
            if avg_err <= self.max_viscosity_error && i >= self.min_viscosity_iter {
                break;
            }

            self.compute_velocity_changes_for_viscosity(
                dt,
                inv_dt,
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
                boundaries,
            );
        }
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

        self.compute_alphas(
            inv_dt,
            &contact_manager.fluid_fluid_contacts,
            &contact_manager.fluid_boundary_contacts,
            fluids,
        );

        self.divergence_solve(
            dt,
            inv_dt,
            kernel_radius,
            contact_manager,
            fluids,
            boundaries,
        );

        self.update_velocities(dt, fluids);
        self.velocity_changes
            .iter_mut()
            .for_each(|vs| vs.iter_mut().for_each(|v| v.fill(N::zero())));

        //        self.nonpressure_solve(dt, inv_dt, contact_manager, fluids);

        self.pressure_solve(
            dt,
            inv_dt,
            kernel_radius,
            contact_manager,
            fluids,
            boundaries,
        );

        self.viscosity_solve(
            dt,
            inv_dt,
            kernel_radius,
            contact_manager,
            fluids,
            boundaries,
        );

        self.update_positions(dt, fluids);
    }
}
