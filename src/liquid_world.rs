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
    solver: DFSPHSolver<N>, // , crate::kernel::Poly6Kernel, crate::kernel::SpikyKernel>,
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
                &self.fluids,
                Some(self.solver.position_changes()),
                &mut self.hgrid,
            );

            geometry::insert_boundaries_to_grid(&self.boundaries, &mut self.hgrid);
            self.solver.init_with_boundaries(&self.boundaries);

            self.contact_manager.update_contacts(
                self.h,
                &self.fluids,
                &self.boundaries,
                Some(self.solver.position_changes()),
                &self.hgrid,
            );

            self.solver.step(
                dt,
                &mut self.contact_manager,
                self.h,
                &mut self.fluids,
                &self.boundaries,
            );

            remaining_time -= substep_dt;
        }
    }

    /// Advances the simulation by `dt` milliseconds, taking into account coupling with nphysic's colliders.
    ///
    /// All the fluid particles will be affected by an acceleration equal to `gravity`.
    #[cfg(feature = "nphysics")]
    pub fn step_with_coupling<Bodies, Colliders>(
        &mut self,
        dt: N,
        gravity: &Vector<N>,
        // We keep this here because it is very likely to become useful in the future.
        _geometrical_world: &GeometricalWorld<N, Bodies::Handle, Colliders::Handle>,
        coupling: &mut ColliderCouplingManager<N, Colliders::Handle>,
        bodies: &mut Bodies,
        colliders: &mut Colliders,
    ) where
        Bodies: BodySet<N>,
        Colliders: ColliderSet<N, Bodies::Handle>,
    {
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
                &self.fluids,
                Some(self.solver.position_changes()),
                &mut self.hgrid,
            );

            coupling.update_boundaries(
                self.h,
                colliders,
                &mut self.boundaries,
                &self.fluids,
                self.solver.position_changes_mut(),
                &self.hgrid,
            );

            geometry::insert_boundaries_to_grid(&self.boundaries, &mut self.hgrid);
            self.solver.init_with_boundaries(&self.boundaries);

            self.contact_manager.update_contacts(
                self.h,
                &self.fluids,
                &self.boundaries,
                Some(self.solver.position_changes()),
                &self.hgrid,
            );

            self.solver.step(
                dt,
                &mut self.contact_manager,
                self.h,
                &mut self.fluids,
                &self.boundaries,
            );

            coupling.transmit_forces(&mut self.boundaries, bodies, colliders);

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
