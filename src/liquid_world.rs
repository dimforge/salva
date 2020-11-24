use crate::counters::Counters;
use crate::coupling::CouplingManager;
use crate::geometry::{self, ContactManager, HGrid, HGridEntry};
use crate::math::{Point, Real, Vector};
use crate::object::{Boundary, BoundaryHandle, BoundarySet};
use crate::object::{Fluid, FluidHandle, FluidSet};
use crate::solver::PressureSolver;
use crate::TimestepManager;

use ncollide::bounding_volume::AABB;

/// The physics world for simulating fluids with boundaries.
pub struct LiquidWorld {
    /// Performance counters of the whole fluid simulation engine.
    pub counters: Counters,
    nsubsteps_since_sort: usize,
    particle_radius: Real,
    h: Real,
    fluids: FluidSet,
    boundaries: BoundarySet,
    solver: Box<dyn PressureSolver>,
    contact_manager: ContactManager,
    timestep_manager: TimestepManager,
    hgrid: HGrid<HGridEntry>,
}

impl LiquidWorld {
    /// Initialize a new liquid world.
    ///
    /// # Parameters
    ///
    /// - `particle_radius`: the radius of every particle on this world.
    /// - `smoothing_factor`: the smoothing factor used to compute the SPH kernel radius.
    ///    The kernel radius will be computed as `particle_radius * smoothing_factor * 2.0.
    pub fn new(
        solver: impl PressureSolver + 'static,
        particle_radius: Real,
        smoothing_factor: Real,
    ) -> Self {
        let h = particle_radius * smoothing_factor * na::convert::<_, Real>(2.0);
        Self {
            counters: Counters::new(),
            nsubsteps_since_sort: 0,
            particle_radius,
            h,
            fluids: FluidSet::new(),
            boundaries: BoundarySet::new(),
            solver: Box::new(solver),
            contact_manager: ContactManager::new(),
            timestep_manager: TimestepManager::new(particle_radius),
            hgrid: HGrid::new(h),
        }
    }

    /// Advances the simulation by `dt` milliseconds.
    ///
    /// All the fluid particles will be affected by an acceleration equal to `gravity`.
    pub fn step(&mut self, dt: Real, gravity: &Vector<Real>) {
        self.step_with_coupling(dt, gravity, &mut ())
    }

    /// Advances the simulation by `dt` milliseconds, taking into account coupling with an external rigid-body engine.
    pub fn step_with_coupling(
        &mut self,
        dt: Real,
        gravity: &Vector<Real>,
        coupling: &mut impl CouplingManager,
    ) {
        self.counters.reset();
        self.counters.step_time.start();
        self.timestep_manager.reset(dt);

        self.solver.init_with_fluids(self.fluids.as_slice());

        for fluid in self.fluids.as_mut_slice() {
            fluid.apply_particles_removal();
        }

        // Perform substeps.
        while !self.timestep_manager.is_done() {
            self.nsubsteps_since_sort += 1;
            self.counters.nsubsteps += 1;

            self.counters.stages.collision_detection_time.resume();
            self.counters.cd.grid_insertion_time.resume();
            self.hgrid.clear();
            geometry::insert_fluids_to_grid(self.fluids.as_slice(), &mut self.hgrid);
            self.counters.cd.grid_insertion_time.pause();

            self.counters.cd.boundary_update_time.resume();
            coupling.update_boundaries(
                &self.timestep_manager,
                self.h,
                self.particle_radius,
                &self.hgrid,
                self.fluids.as_mut_slice(),
                &mut self.boundaries,
            );
            self.counters.cd.boundary_update_time.pause();

            self.counters.cd.grid_insertion_time.resume();
            geometry::insert_boundaries_to_grid(self.boundaries.as_slice(), &mut self.hgrid);
            self.counters.cd.grid_insertion_time.pause();

            self.solver.init_with_boundaries(self.boundaries.as_slice());

            self.contact_manager.update_contacts(
                &mut self.counters,
                self.h,
                self.fluids.as_slice(),
                self.boundaries.as_slice(),
                &self.hgrid,
            );

            self.counters.cd.ncontacts = self.contact_manager.ncontacts();
            self.counters.stages.collision_detection_time.pause();

            self.counters.stages.solver_time.resume();
            self.solver.evaluate_kernels(
                self.h,
                &mut self.contact_manager,
                self.fluids.as_slice(),
                self.boundaries.as_slice(),
            );

            self.solver.compute_densities(
                &self.contact_manager,
                self.fluids.as_slice(),
                self.boundaries.as_mut_slice(),
            );

            self.solver.step(
                &mut self.counters,
                &mut self.timestep_manager,
                gravity,
                &mut self.contact_manager,
                self.h,
                self.fluids.as_mut_slice(),
                self.boundaries.as_slice(),
            );

            coupling.transmit_forces(&self.boundaries);
            self.counters.stages.solver_time.pause();
        }

        //        if self.nsubsteps_since_sort >= 100 {
        //            self.nsubsteps_since_sort = 0;
        //            println!("Performing z-sort of particles.");
        //            par_iter_mut!(self.fluids.as_mut_slice()).for_each(|fluid| fluid.z_sort())
        //        }

        self.counters.step_time.pause();
        //        println!("Counters: {}", self.counters);
    }

    /// Add a fluid to the liquid world.
    pub fn add_fluid(&mut self, fluid: Fluid) -> FluidHandle {
        self.fluids.insert(fluid)
    }

    /// Add a boundary to the liquid world.
    pub fn add_boundary(&mut self, boundary: Boundary) -> BoundaryHandle {
        self.boundaries.insert(boundary)
    }

    /// Add a fluid to the liquid world.
    pub fn remove_fluid(&mut self, handle: FluidHandle) -> Option<Fluid> {
        self.fluids.remove(handle)
    }

    /// Add a boundary to the liquid world.
    pub fn remove_boundary(&mut self, handle: BoundaryHandle) -> Option<Boundary> {
        self.boundaries.remove(handle)
    }

    // #[cfg(feature = "dim3")]
    pub fn particles_intersecting_aabb(&self, aabb: &AABB<Real>) -> Vec<Point<Real>> {
        self.fluids
            .iter()
            .flat_map(|(_, fluid)| fluid.particles_intersecting_aabb(aabb))
            .collect()
    }

    /// The set of fluids on this liquid world.
    pub fn fluids(&self) -> &FluidSet {
        &self.fluids
    }

    /// The mutable set of fluids on this liquid world.
    pub fn fluids_mut(&mut self) -> &mut FluidSet {
        &mut self.fluids
    }

    /// The set of boundaries on this liquid world.
    pub fn boundaries(&self) -> &BoundarySet {
        &self.boundaries
    }

    /// The mutable set of boundaries on this liquid world.
    pub fn boundaries_mut(&mut self) -> &mut BoundarySet {
        &mut self.boundaries
    }

    /// The SPH kernel radius.
    pub fn h(&self) -> Real {
        self.h
    }

    /// The radius of every particle on this liquid world.
    pub fn particle_radius(&self) -> Real {
        self.particle_radius
    }
}
