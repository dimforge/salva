use crate::coupling::CouplingManager;
use crate::geometry::{self, ContactManager, HGrid, HGridEntry};
use crate::math::Vector;
use crate::object::{Boundary, BoundaryHandle};
use crate::object::{Fluid, FluidHandle};
use crate::solver::{DFSPHSolver, PBFSolver};
use crate::TimestepManager;
use na::RealField;

#[cfg(feature = "nphysics")]
use {
    crate::coupling::ColliderCouplingManager,
    nphysics::object::{BodySet, ColliderSet},
    nphysics::world::GeometricalWorld,
};

/// The physics world for simulating fluids with boundaries.
pub struct LiquidWorld<N: RealField> {
    particle_radius: N,
    h: N,
    fluids: Vec<Fluid<N>>,
    boundaries: Vec<Boundary<N>>,
    solver: DFSPHSolver<N>,
    contact_manager: ContactManager<N>,
    timestep_manager: TimestepManager<N>,
    hgrid: HGrid<N, HGridEntry>,
}

impl<N: RealField> LiquidWorld<N> {
    /// Initialize a new liquid world.
    ///
    /// # Parameters
    ///
    /// - `particle_radius`: the radius of every particle on this world.
    /// - `smoothing_factor`: the smoothing factor used to compute the SPH kernel radius.
    ///    The kernel radius will be computed as `particle_radius * smoothing_factor * 2.0.
    pub fn new(particle_radius: N, smoothing_factor: N) -> Self {
        let h = particle_radius * smoothing_factor * na::convert(2.0);
        Self {
            particle_radius,
            h,
            fluids: Vec::new(),
            boundaries: Vec::new(),
            solver: DFSPHSolver::new(),
            contact_manager: ContactManager::new(),
            timestep_manager: TimestepManager::new(),
            hgrid: HGrid::new(h),
        }
    }

    /// Advances the simulation by `dt` milliseconds.
    ///
    /// All the fluid particles will be affected by an acceleration equal to `gravity`.
    pub fn step(&mut self, dt: N, gravity: &Vector<N>) {
        self.step_with_coupling(dt, gravity, &mut ())
    }

    /// Advances the simulation by `dt` milliseconds, taking into account coupling with an external rigid-body engine.
    pub fn step_with_coupling(
        &mut self,
        dt: N,
        gravity: &Vector<N>,
        coupling: &mut impl CouplingManager<N>,
    ) {
        let mut remaining_time = dt;

        // Perform substeps.
        while remaining_time > N::zero() {
            // Substep length.
            let substep_dt = self.timestep_manager.compute_substep(
                dt,
                remaining_time,
                self.particle_radius,
                &self.fluids,
            );

            self.solver.init_with_fluids(&self.fluids);
            self.solver
                .predict_advection(substep_dt, gravity, &self.fluids);

            self.hgrid.clear();
            geometry::insert_fluids_to_grid(
                substep_dt,
                &self.fluids,
                Some(self.solver.velocity_changes()),
                &mut self.hgrid,
            );

            coupling.update_boundaries(
                substep_dt,
                self.h,
                &self.hgrid,
                &mut self.fluids,
                self.solver.velocity_changes_mut(),
                &mut self.boundaries,
            );

            geometry::insert_boundaries_to_grid(&self.boundaries, &mut self.hgrid);
            self.solver.init_with_boundaries(&self.boundaries);

            self.contact_manager.update_contacts(
                substep_dt,
                self.h,
                &self.fluids,
                &self.boundaries,
                Some(self.solver.velocity_changes()),
                &self.hgrid,
            );

            self.solver.step(
                substep_dt,
                &mut self.contact_manager,
                self.h,
                &mut self.fluids,
                &self.boundaries,
            );

            coupling.transmit_forces(&self.boundaries);

            remaining_time -= substep_dt;
        }
    }

    /// Add a fluid to the liquid world.
    pub fn add_fluid(&mut self, fluid: Fluid<N>) -> FluidHandle {
        let handle = self.fluids.len();
        self.fluids.push(fluid);
        handle
    }

    /// Add a boundary to the liquid world.
    pub fn add_boundary(&mut self, boundary: Boundary<N>) -> BoundaryHandle {
        let handle = self.boundaries.len();
        self.boundaries.push(boundary);
        handle
    }

    /// The set of fluids on this liquid world.
    pub fn fluids(&self) -> &[Fluid<N>] {
        &self.fluids
    }

    /// The set of boundaries on this liquid world.
    pub fn boundaries(&self) -> &[Boundary<N>] {
        &self.boundaries
    }

    /// The SPH kernel radius.
    pub fn h(&self) -> N {
        self.h
    }

    /// The radius of every particle on this liquid world.
    pub fn particle_radius(&self) -> N {
        self.particle_radius
    }
}
