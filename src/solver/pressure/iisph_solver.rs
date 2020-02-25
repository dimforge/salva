use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use na::{self, RealField};

use crate::counters::Counters;
use crate::geometry::{ContactManager, ParticlesContacts};
use crate::kernel::{CubicSplineKernel, Kernel};
use crate::math::Vector;
use crate::object::{Boundary, Fluid};
use crate::solver::{helper, PressureSolver};
use crate::TimestepManager;

/// A Position Based Fluid solver.
pub struct IISPHSolver<
    N: RealField,
    KernelDensity: Kernel = CubicSplineKernel,
    KernelGradient: Kernel = CubicSplineKernel,
> {
    min_pressure_iter: usize,
    max_pressure_iter: usize,
    max_density_error: N,
    omega: N,
    densities: Vec<Vec<N>>,
    aii: Vec<Vec<N>>,
    dii: Vec<Vec<Vector<N>>>,
    dij_pjl: Vec<Vec<Vector<N>>>,
    pressures: Vec<Vec<N>>,
    next_pressures: Vec<Vec<N>>,
    predicted_densities: Vec<Vec<N>>,
    velocity_changes: Vec<Vec<Vector<N>>>,
    phantoms: PhantomData<(KernelDensity, KernelGradient)>,
}

impl<N, KernelDensity, KernelGradient> IISPHSolver<N, KernelDensity, KernelGradient>
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
            omega: na::convert(0.5),
            densities: Vec::new(),
            dii: Vec::new(),
            aii: Vec::new(),
            dij_pjl: Vec::new(),
            pressures: Vec::new(),
            next_pressures: Vec::new(),
            predicted_densities: Vec::new(),
            velocity_changes: Vec::new(),
            phantoms: PhantomData,
        }
    }

    fn compute_boundary_volumes(
        &mut self,
        boundary_boundary_contacts: &[ParticlesContacts<N>],
        boundaries: &mut [Boundary<N>],
    ) {
        for boundary_id in 0..boundaries.len() {
            par_iter_mut!(boundaries[boundary_id].volumes)
                .enumerate()
                .for_each(|(i, volume)| {
                    let mut denominator = N::zero();

                    for c in boundary_boundary_contacts[boundary_id]
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        denominator += c.weight;
                    }

                    assert!(!denominator.is_zero());
                    *volume = N::one() / denominator;
                })
        }
    }

    fn compute_predicted_densities(
        &mut self,
        timestep: &TimestepManager<N>,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        let velocity_changes = &self.velocity_changes;
        let densities = &self.densities;
        let _max_error = N::zero();

        for fluid_id in 0..fluids.len() {
            let _it = par_iter_mut!(self.predicted_densities[fluid_id])
                .enumerate()
                .for_each(|(i, predicted_density)| {
                    let fluid_i = &fluids[fluid_id];
                    let mut delta = N::zero();

                    for c in fluid_fluid_contacts[fluid_id]
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        let fluid_j = &fluids[c.j_model];
                        let vi = fluid_i.velocities[c.i] + velocity_changes[c.i_model][c.i];
                        let vj = fluid_j.velocities[c.j] + velocity_changes[c.j_model][c.j];

                        delta += fluids[c.j_model].particle_mass(c.j) * (vi - vj).dot(&c.gradient);
                    }

                    for c in fluid_boundary_contacts[fluid_id]
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        let vi = fluid_i.velocities[c.i] + velocity_changes[c.i_model][c.i];
                        let vj = boundaries[c.j_model].velocities[c.j];

                        delta += boundaries[c.j_model].volumes[c.j]
                            * fluid_i.density0
                            * (vi - vj).dot(&c.gradient);
                    }

                    *predicted_density = densities[fluid_id][i] + delta * timestep.dt();
                    assert!(!predicted_density.is_zero());
                });
        }
    }

    fn compute_dii(
        &mut self,
        timestep: &TimestepManager<N>,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let fluid_boundary_contacts = &fluid_boundary_contacts[fluid_id];
            let dii = &mut self.dii[fluid_id];
            let fluid_i = &fluids[fluid_id];
            let densities = &self.densities;

            par_iter_mut!(dii).enumerate().for_each(|(i, dii)| {
                dii.fill(N::zero());

                let rhoi = densities[fluid_id][i];
                let factor = -timestep.dt() * timestep.dt() / (rhoi * rhoi);

                for c in fluid_fluid_contacts
                    .particle_contacts(i)
                    .read()
                    .unwrap()
                    .iter()
                {
                    let mj = fluids[c.j_model].particle_mass(c.j);
                    *dii += c.gradient * (mj * factor);
                }

                for c in fluid_boundary_contacts
                    .particle_contacts(i)
                    .read()
                    .unwrap()
                    .iter()
                {
                    let mj = boundaries[c.j_model].volumes[c.j] * fluid_i.density0;
                    *dii += c.gradient * (mj * factor);
                }
            })
        }
    }

    fn compute_aii(
        &mut self,
        timestep: &TimestepManager<N>,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let fluid_boundary_contacts = &fluid_boundary_contacts[fluid_id];
            let aii = &mut self.aii[fluid_id];
            let dii = &self.dii[fluid_id];
            let fluid_i = &fluids[fluid_id];
            let densities = &self.densities;

            par_iter_mut!(aii).enumerate().for_each(|(i, aii)| {
                *aii = N::zero();
                let rhoi = densities[fluid_id][i];
                let mi = fluids[fluid_id].particle_mass(i);
                let factor = timestep.dt() * timestep.dt() * mi / (rhoi * rhoi);

                for c in fluid_fluid_contacts
                    .particle_contacts(i)
                    .read()
                    .unwrap()
                    .iter()
                {
                    let mj = fluids[c.j_model].particle_mass(c.j);
                    let dji = c.gradient * factor;
                    *aii += mj * (dii[c.i] - dji).dot(&c.gradient);
                }

                for c in fluid_boundary_contacts
                    .particle_contacts(i)
                    .read()
                    .unwrap()
                    .iter()
                {
                    let mj = boundaries[c.j_model].volumes[c.j] * fluid_i.density0;
                    let dji = c.gradient * factor;
                    *aii += mj * (dii[c.i] - dji).dot(&c.gradient);
                }
            })
        }
    }

    fn compute_dij_pjl(
        &mut self,
        timestep: &TimestepManager<N>,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
    ) {
        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let _fluid_boundary_contacts = &fluid_boundary_contacts[fluid_id];
            let dij_pjl = &mut self.dij_pjl[fluid_id];
            let _fluid_i = &fluids[fluid_id];
            let densities = &self.densities;
            let pressures = &self.pressures;

            par_iter_mut!(dij_pjl).enumerate().for_each(|(i, dij_pjl)| {
                dij_pjl.fill(N::zero());

                for c in fluid_fluid_contacts
                    .particle_contacts(i)
                    .read()
                    .unwrap()
                    .iter()
                {
                    let rhoj = densities[c.j_model][c.j];
                    let mj = fluids[c.j_model].particle_mass(c.j);
                    let p_jl = pressures[c.j_model][c.j];
                    *dij_pjl += c.gradient * (-mj * p_jl / (rhoj * rhoj));
                }

                *dij_pjl *= timestep.dt() * timestep.dt();
            })
        }
    }

    fn compute_next_pressures(
        &mut self,
        timestep: &TimestepManager<N>,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    ) -> N {
        let mut max_error = N::zero();

        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let fluid_boundary_contacts = &fluid_boundary_contacts[fluid_id];
            let next_pressures = &mut self.next_pressures[fluid_id];
            let pressures = &self.pressures;
            let fluid_i = &fluids[fluid_id];
            let densities = &self.densities;
            let predicted_densities = &self.predicted_densities;
            let omega = self.omega;
            let aii = &self.aii[fluid_id];
            let dij_pjl = &self.dij_pjl;
            let dii = &self.dii;

            let it = par_iter_mut!(next_pressures)
                .enumerate()
                .map(|(i, next_pressure)| {
                    if aii[i].abs() > na::convert(1.0e-9) {
                        let mut sum = N::zero();
                        let pi = pressures[fluid_id][i];
                        let mi = fluid_i.particle_mass(i);
                        let rhoi = densities[fluid_id][i];
                        let derr = fluid_i.density0 - predicted_densities[fluid_id][i];

                        for c in fluid_fluid_contacts
                            .particle_contacts(i)
                            .read()
                            .unwrap()
                            .iter()
                        {
                            let mj = fluids[c.j_model].particle_mass(c.j);
                            let dji =
                                c.gradient * (timestep.dt() * timestep.dt() * mi / (rhoi * rhoi));
                            let factor = dij_pjl[c.i_model][c.i]
                                - dii[c.j_model][c.j] * pressures[c.j_model][c.j]
                                - (dij_pjl[c.j_model][c.j] - dji * pi);
                            sum += mj * factor.dot(&c.gradient);
                        }

                        for c in fluid_boundary_contacts
                            .particle_contacts(i)
                            .read()
                            .unwrap()
                            .iter()
                        {
                            let mj = boundaries[c.j_model].volumes[c.j] * fluid_i.density0;
                            sum += mj * dij_pjl[c.i_model][c.i].dot(&c.gradient);
                        }

                        *next_pressure = (N::one() - omega) * pi + omega * (derr - sum) / aii[i];

                        if *next_pressure > N::zero() {
                            *next_pressure = next_pressure.max(N::zero());
                            (-sum - aii[i] * *next_pressure) / fluid_i.density0
                        } else {
                            // Clamp negative pressures.
                            *next_pressure = N::zero();
                            N::zero()
                        }
                    } else {
                        *next_pressure = N::zero();
                        N::zero()
                    }
                });
            let err = par_reduce_sum!(N::zero(), it);

            let nparts = fluids[fluid_id].num_particles();
            if nparts != 0 {
                max_error = max_error.max(err / na::convert(nparts as f64));
            }
        }

        max_error
    }

    fn compute_velocity_changes(
        &mut self,
        timestep: &TimestepManager<N>,
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        let densities = &self.densities;
        let pressures = &self.pressures;

        for (fluid_id, _fluid1) in fluids.iter().enumerate() {
            par_iter_mut!(self.velocity_changes[fluid_id])
                .enumerate()
                .for_each(|(i, velocity_change)| {
                    let fluid_i = &fluids[fluid_id];
                    let pi = pressures[fluid_id][i];
                    let rhoi = densities[fluid_id][i];

                    for c in fluid_fluid_contacts[fluid_id]
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        let mj = fluids[c.j_model].particle_mass(c.j);
                        let pj = pressures[c.j_model][c.j];
                        let rhoj = densities[c.j_model][c.j];

                        *velocity_change -= c.gradient
                            * (timestep.dt() * mj * (pi / (rhoi * rhoi) + pj / (rhoj * rhoj)));
                    }

                    for c in fluid_boundary_contacts[fluid_id]
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        let mj = boundaries[c.j_model].volumes[c.j] * fluid_i.density0;
                        *velocity_change -= c.gradient * (timestep.dt() * mj * pi / (rhoi * rhoi));
                    }
                })
        }
    }

    fn update_velocities_and_positions(
        &mut self,
        timestep: &TimestepManager<N>,
        fluids: &mut [Fluid<N>],
    ) {
        for (fluid, delta) in fluids.iter_mut().zip(self.velocity_changes.iter()) {
            par_iter_mut!(fluid.positions)
                .zip(par_iter_mut!(fluid.velocities))
                .zip(par_iter!(delta))
                .for_each(|((pos, vel), delta)| {
                    *vel += delta;
                    *pos += *vel * timestep.dt();
                })
        }
    }

    fn pressure_solve(
        &mut self,
        timestep: &TimestepManager<N>,
        _kernel_radius: N,
        contact_manager: &mut ContactManager<N>,
        fluids: &mut [Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        for i in 0..self.max_pressure_iter {
            self.compute_dij_pjl(
                timestep,
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
            );

            let avg_err = self.compute_next_pressures(
                timestep,
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                fluids,
                boundaries,
            );

            std::mem::swap(&mut self.pressures, &mut self.next_pressures);

            if avg_err <= self.max_density_error && i >= self.min_pressure_iter {
                //                println!(
                //                    "Average density error: {}, break after niters: {}",
                //                    avg_err, i
                //                );
                break;
            }
        }
    }

    fn integrate_and_clear_accelerations(
        &mut self,
        timestep: &TimestepManager<N>,
        fluids: &mut [Fluid<N>],
    ) {
        for (velocity_changes, fluid) in self.velocity_changes.iter_mut().zip(fluids.iter_mut()) {
            par_iter_mut!(velocity_changes)
                .zip(par_iter_mut!(fluid.accelerations))
                .for_each(|(velocity_change, acceleration)| {
                    *velocity_change += *acceleration * timestep.dt();
                    acceleration.fill(N::zero());
                })
        }
    }
}

impl<N, KernelDensity, KernelGradient> PressureSolver<N>
    for IISPHSolver<N, KernelDensity, KernelGradient>
where
    N: RealField,
    KernelDensity: Kernel,
    KernelGradient: Kernel,
{
    fn init_with_fluids(&mut self, fluids: &[Fluid<N>]) {
        // Resize every buffer.
        self.densities.resize(fluids.len(), Vec::new());
        self.predicted_densities.resize(fluids.len(), Vec::new());
        self.velocity_changes.resize(fluids.len(), Vec::new());
        self.aii.resize(fluids.len(), Vec::new());
        self.dii.resize(fluids.len(), Vec::new());
        self.dij_pjl.resize(fluids.len(), Vec::new());
        self.pressures.resize(fluids.len(), Vec::new());
        self.next_pressures.resize(fluids.len(), Vec::new());

        for i in 0..fluids.len() {
            let nparticles = fluids[i].num_particles();

            self.densities[i].resize(nparticles, N::zero());
            self.predicted_densities[i].resize(nparticles, N::zero());
            self.velocity_changes[i].resize(nparticles, Vector::zeros());
            self.aii[i].resize(nparticles, N::zero());
            self.dii[i].resize(nparticles, Vector::zeros());
            self.dij_pjl[i].resize(nparticles, Vector::zeros());
            self.pressures[i].resize(nparticles, N::zero());
            self.next_pressures[i].resize(nparticles, N::zero());
        }
    }

    fn init_with_boundaries(&mut self, _boundaries: &[Boundary<N>]) {}

    fn predict_advection(
        &mut self,
        timestep: &TimestepManager<N>,
        kernel_radius: N,
        contact_manager: &ContactManager<N>,
        gravity: &Vector<N>,
        fluids: &mut [Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        for fluid in fluids.iter_mut() {
            par_iter_mut!(fluid.accelerations).for_each(|acceleration| {
                *acceleration += gravity;
            })
        }

        for (fluid, fluid_fluid_contacts, fluid_boundary_contacts, densities) in
            itertools::multizip((
                &mut *fluids,
                &contact_manager.fluid_fluid_contacts,
                &contact_manager.fluid_boundary_contacts,
                &self.densities,
            ))
        {
            let mut forces = std::mem::replace(&mut fluid.nonpressure_forces, Vec::new());

            for np_force in &mut forces {
                np_force.solve(
                    timestep,
                    kernel_radius,
                    fluid_fluid_contacts,
                    fluid_boundary_contacts,
                    fluid,
                    boundaries,
                    densities,
                );
            }

            fluid.nonpressure_forces = forces;
        }
    }

    fn evaluate_kernels(
        &mut self,
        kernel_radius: N,
        contact_manager: &mut ContactManager<N>,
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        helper::update_fluid_contacts::<_, KernelDensity, KernelGradient>(
            kernel_radius,
            &mut contact_manager.fluid_fluid_contacts,
            &mut contact_manager.fluid_boundary_contacts,
            fluids,
            boundaries,
        );

        helper::update_boundary_contacts::<_, KernelDensity, KernelGradient>(
            kernel_radius,
            &mut contact_manager.boundary_boundary_contacts,
            boundaries,
        );
    }

    fn compute_densities(
        &mut self,
        contact_manager: &ContactManager<N>,
        fluids: &[Fluid<N>],
        boundaries: &mut [Boundary<N>],
    ) {
        self.compute_boundary_volumes(&contact_manager.boundary_boundary_contacts, boundaries);

        for fluid_id in 0..fluids.len() {
            par_iter_mut!(self.densities[fluid_id])
                .enumerate()
                .for_each(|(i, density)| {
                    *density = N::zero();

                    for c in contact_manager.fluid_fluid_contacts[fluid_id]
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        *density += fluids[c.j_model].particle_mass(c.j) * c.weight;
                    }

                    for c in contact_manager.fluid_boundary_contacts[fluid_id]
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        *density += boundaries[c.j_model].volumes[c.j]
                            * fluids[c.i_model].density0
                            * c.weight;
                    }

                    assert!(!density.is_zero());
                })
        }
    }

    fn step(
        &mut self,
        counters: &mut Counters,
        timestep: &mut TimestepManager<N>,
        gravity: &Vector<N>,
        contact_manager: &mut ContactManager<N>,
        kernel_radius: N,
        fluids: &mut [Fluid<N>],
        boundaries: &[Boundary<N>],
    ) {
        self.predict_advection(
            timestep,
            kernel_radius,
            contact_manager,
            gravity,
            fluids,
            boundaries,
        );
        timestep.advance(fluids);
        self.integrate_and_clear_accelerations(timestep, fluids);

        counters.solver.pressure_resolution_time.resume();
        self.compute_dii(
            timestep,
            &contact_manager.fluid_fluid_contacts,
            &contact_manager.fluid_boundary_contacts,
            fluids,
            boundaries,
        );

        let _0_5: N = na::convert(0.5);
        self.pressures
            .iter_mut()
            .flat_map(|v| v.iter_mut())
            .for_each(|p| *p *= _0_5);

        let _ = self.compute_predicted_densities(
            timestep,
            &contact_manager.fluid_fluid_contacts,
            &contact_manager.fluid_boundary_contacts,
            fluids,
            boundaries,
        );

        self.compute_aii(
            timestep,
            &contact_manager.fluid_fluid_contacts,
            &contact_manager.fluid_boundary_contacts,
            fluids,
            boundaries,
        );

        self.pressure_solve(timestep, kernel_radius, contact_manager, fluids, boundaries);

        self.compute_velocity_changes(
            timestep,
            &contact_manager.fluid_fluid_contacts,
            &contact_manager.fluid_boundary_contacts,
            fluids,
            boundaries,
        );

        self.update_velocities_and_positions(timestep, fluids);

        self.velocity_changes
            .iter_mut()
            .for_each(|vs| vs.iter_mut().for_each(|v| v.fill(N::zero())));
        counters.solver.pressure_resolution_time.pause();
    }
}
