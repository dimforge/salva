use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use num::Zero;

use crate::counters::Counters;
use crate::geometry::{ContactManager, ParticlesContacts};
use crate::kernel::{CubicSplineKernel, Kernel};
use crate::math::{Real, Vector};
use crate::object::{Boundary, Fluid};
use crate::solver::{helper, PressureSolver};
use crate::TimestepManager;

/// A IISPH (Implicit Incompressible Smoothed Particle Hydrodynamics) pressure solver.
pub struct IISPHSolver<
    KernelDensity: Kernel = CubicSplineKernel,
    KernelGradient: Kernel = CubicSplineKernel,
> {
    /// Minimum number of iterations that must be executed for pressure resolution.
    pub min_pressure_iter: usize,
    /// Maximum number of iterations that must be executed for pressure resolution.
    pub max_pressure_iter: usize,
    /// Maximum acceptable density error (in percents).
    ///
    /// The pressure solver will continue iterating until the density error drops bellow this
    /// threshold, or until the maximum number of pressure iterations is reached.
    pub max_density_error: Real,
    omega: Real,
    densities: Vec<Vec<Real>>,
    aii: Vec<Vec<Real>>,
    dii: Vec<Vec<Vector<Real>>>,
    dij_pjl: Vec<Vec<Vector<Real>>>,
    pressures: Vec<Vec<Real>>,
    next_pressures: Vec<Vec<Real>>,
    predicted_densities: Vec<Vec<Real>>,
    velocity_changes: Vec<Vec<Vector<Real>>>,
    phantoms: PhantomData<(KernelDensity, KernelGradient)>,
}

impl<KernelDensity, KernelGradient> IISPHSolver<KernelDensity, KernelGradient>
where
    KernelDensity: Kernel,
    KernelGradient: Kernel,
{
    /// Initialize a new IISPH pressure solver.
    pub fn new() -> Self {
        Self {
            min_pressure_iter: 1,
            max_pressure_iter: 50,
            max_density_error: na::convert::<_, Real>(0.05),
            omega: na::convert::<_, Real>(0.5),
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
        boundary_boundary_contacts: &[ParticlesContacts],
        boundaries: &mut [Boundary],
    ) {
        for boundary_id in 0..boundaries.len() {
            par_iter_mut!(boundaries[boundary_id].volumes)
                .enumerate()
                .for_each(|(i, volume)| {
                    let mut denominator = na::zero::<Real>();

                    for c in boundary_boundary_contacts[boundary_id]
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        denominator += c.weight;
                    }

                    assert!(!denominator.is_zero());
                    *volume = na::one::<Real>() / denominator;
                })
        }
    }

    fn compute_predicted_densities(
        &mut self,
        timestep: &TimestepManager,
        fluid_fluid_contacts: &[ParticlesContacts],
        fluid_boundary_contacts: &[ParticlesContacts],
        fluids: &[Fluid],
        boundaries: &[Boundary],
    ) {
        let velocity_changes = &self.velocity_changes;
        let densities = &self.densities;
        let _max_error = na::zero::<Real>();

        for fluid_id in 0..fluids.len() {
            let _it = par_iter_mut!(self.predicted_densities[fluid_id])
                .enumerate()
                .for_each(|(i, predicted_density)| {
                    let fluid_i = &fluids[fluid_id];
                    let mut delta = na::zero::<Real>();

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
        timestep: &TimestepManager,
        fluid_fluid_contacts: &[ParticlesContacts],
        fluid_boundary_contacts: &[ParticlesContacts],
        fluids: &[Fluid],
        boundaries: &[Boundary],
    ) {
        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let fluid_boundary_contacts = &fluid_boundary_contacts[fluid_id];
            let dii = &mut self.dii[fluid_id];
            let fluid_i = &fluids[fluid_id];
            let densities = &self.densities;

            par_iter_mut!(dii).enumerate().for_each(|(i, dii)| {
                dii.fill(na::zero::<Real>());

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
        timestep: &TimestepManager,
        fluid_fluid_contacts: &[ParticlesContacts],
        fluid_boundary_contacts: &[ParticlesContacts],
        fluids: &[Fluid],
        boundaries: &[Boundary],
    ) {
        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let fluid_boundary_contacts = &fluid_boundary_contacts[fluid_id];
            let aii = &mut self.aii[fluid_id];
            let dii = &self.dii[fluid_id];
            let fluid_i = &fluids[fluid_id];
            let densities = &self.densities;

            par_iter_mut!(aii).enumerate().for_each(|(i, aii)| {
                *aii = na::zero::<Real>();
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
        timestep: &TimestepManager,
        fluid_fluid_contacts: &[ParticlesContacts],
        fluid_boundary_contacts: &[ParticlesContacts],
        fluids: &[Fluid],
    ) {
        for fluid_id in 0..fluids.len() {
            let fluid_fluid_contacts = &fluid_fluid_contacts[fluid_id];
            let _fluid_boundary_contacts = &fluid_boundary_contacts[fluid_id];
            let dij_pjl = &mut self.dij_pjl[fluid_id];
            let _fluid_i = &fluids[fluid_id];
            let densities = &self.densities;
            let pressures = &self.pressures;

            par_iter_mut!(dij_pjl).enumerate().for_each(|(i, dij_pjl)| {
                dij_pjl.fill(na::zero::<Real>());

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
        timestep: &TimestepManager,
        fluid_fluid_contacts: &[ParticlesContacts],
        fluid_boundary_contacts: &[ParticlesContacts],
        fluids: &[Fluid],
        boundaries: &[Boundary],
    ) -> Real {
        let mut max_error = na::zero::<Real>();

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
                    if aii[i].abs() > na::convert::<_, Real>(1.0e-9) {
                        let mut sum = na::zero::<Real>();
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

                        *next_pressure =
                            (na::one::<Real>() - omega) * pi + omega * (derr - sum) / aii[i];

                        if *next_pressure > na::zero::<Real>() {
                            *next_pressure = next_pressure.max(na::zero::<Real>());
                            (-sum - aii[i] * *next_pressure) / fluid_i.density0
                        } else {
                            // Clamp negative pressures.
                            *next_pressure = na::zero::<Real>();
                            na::zero::<Real>()
                        }
                    } else {
                        *next_pressure = na::zero::<Real>();
                        na::zero::<Real>()
                    }
                });
            let err = par_reduce_sum!(na::zero::<Real>(), it);

            let nparts = fluids[fluid_id].num_particles();
            if nparts != 0 {
                max_error = max_error.max(err / na::convert::<_, Real>(nparts as f64));
            }
        }

        max_error
    }

    fn compute_velocity_changes(
        &mut self,
        timestep: &TimestepManager,
        fluid_fluid_contacts: &[ParticlesContacts],
        fluid_boundary_contacts: &[ParticlesContacts],
        fluids: &[Fluid],
        boundaries: &[Boundary],
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
                        let acc = c.gradient * (mj * pi / (rhoi * rhoi));
                        *velocity_change -= acc * timestep.dt();

                        // Apply the force to the boundary too.
                        let mi = fluid_i.particle_mass(c.i);
                        boundaries[c.j_model].apply_force(c.j, acc * mi);
                    }
                })
        }
    }

    fn update_velocities_and_positions(
        &mut self,
        timestep: &TimestepManager,
        fluids: &mut [Fluid],
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
        timestep: &TimestepManager,
        _kernel_radius: Real,
        contact_manager: &mut ContactManager,
        fluids: &mut [Fluid],
        boundaries: &[Boundary],
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
        timestep: &TimestepManager,
        fluids: &mut [Fluid],
    ) {
        for (velocity_changes, fluid) in self.velocity_changes.iter_mut().zip(fluids.iter_mut()) {
            par_iter_mut!(velocity_changes)
                .zip(par_iter_mut!(fluid.accelerations))
                .for_each(|(velocity_change, acceleration)| {
                    *velocity_change += *acceleration * timestep.dt();
                    acceleration.fill(na::zero::<Real>());
                })
        }
    }
}

impl<KernelDensity, KernelGradient> PressureSolver for IISPHSolver<KernelDensity, KernelGradient>
where
    KernelDensity: Kernel,
    KernelGradient: Kernel,
{
    fn init_with_fluids(&mut self, fluids: &[Fluid]) {
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

            self.densities[i].resize(nparticles, na::zero::<Real>());
            self.predicted_densities[i].resize(nparticles, na::zero::<Real>());
            self.velocity_changes[i].resize(nparticles, Vector::zeros());
            self.aii[i].resize(nparticles, na::zero::<Real>());
            self.dii[i].resize(nparticles, Vector::zeros());
            self.dij_pjl[i].resize(nparticles, Vector::zeros());
            self.pressures[i].resize(nparticles, na::zero::<Real>());
            self.next_pressures[i].resize(nparticles, na::zero::<Real>());

            if fluids[i].num_deleted_particles() != 0 {
                crate::helper::filter_from_mask(
                    fluids[i].deleted_particles_mask(),
                    &mut self.densities[i],
                );
                crate::helper::filter_from_mask(
                    fluids[i].deleted_particles_mask(),
                    &mut self.predicted_densities[i],
                );
                crate::helper::filter_from_mask(
                    fluids[i].deleted_particles_mask(),
                    &mut self.velocity_changes[i],
                );
                crate::helper::filter_from_mask(
                    fluids[i].deleted_particles_mask(),
                    &mut self.aii[i],
                );
                crate::helper::filter_from_mask(
                    fluids[i].deleted_particles_mask(),
                    &mut self.dii[i],
                );
                crate::helper::filter_from_mask(
                    fluids[i].deleted_particles_mask(),
                    &mut self.dij_pjl[i],
                );
                crate::helper::filter_from_mask(
                    fluids[i].deleted_particles_mask(),
                    &mut self.pressures[i],
                );
                crate::helper::filter_from_mask(
                    fluids[i].deleted_particles_mask(),
                    &mut self.next_pressures[i],
                );
            }
        }
    }

    fn init_with_boundaries(&mut self, _boundaries: &[Boundary]) {}

    fn predict_advection(
        &mut self,
        timestep: &TimestepManager,
        kernel_radius: Real,
        contact_manager: &ContactManager,
        gravity: &Vector<Real>,
        fluids: &mut [Fluid],
        boundaries: &[Boundary],
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
        kernel_radius: Real,
        contact_manager: &mut ContactManager,
        fluids: &[Fluid],
        boundaries: &[Boundary],
    ) {
        helper::update_fluid_contacts::<KernelDensity, KernelGradient>(
            kernel_radius,
            &mut contact_manager.fluid_fluid_contacts,
            &mut contact_manager.fluid_boundary_contacts,
            fluids,
            boundaries,
        );

        helper::update_boundary_contacts::<KernelDensity, KernelGradient>(
            kernel_radius,
            &mut contact_manager.boundary_boundary_contacts,
            boundaries,
        );
    }

    fn compute_densities(
        &mut self,
        contact_manager: &ContactManager,
        fluids: &[Fluid],
        boundaries: &mut [Boundary],
    ) {
        self.compute_boundary_volumes(&contact_manager.boundary_boundary_contacts, boundaries);

        for fluid_id in 0..fluids.len() {
            par_iter_mut!(self.densities[fluid_id])
                .enumerate()
                .for_each(|(i, density)| {
                    *density = na::zero::<Real>();

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
        timestep: &mut TimestepManager,
        gravity: &Vector<Real>,
        contact_manager: &mut ContactManager,
        kernel_radius: Real,
        fluids: &mut [Fluid],
        boundaries: &[Boundary],
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

        let _0_5: Real = na::convert::<_, Real>(0.5);
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
            .for_each(|vs| vs.iter_mut().for_each(|v| v.fill(na::zero::<Real>())));
        counters.solver.pressure_resolution_time.pause();
    }
}
