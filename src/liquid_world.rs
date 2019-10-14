use crate::boundary::{Boundary, BoundaryHandle};
use crate::fluid::Fluid;
use crate::geometry::{self, ContactManager, HGrid, HGridEntry, ParticlesContacts};
use crate::math::Vector;
use crate::solver::PBFSolver;
use crate::TimestepManager;
use na::RealField;

#[cfg(feature = "nphysics")]
use {
    crate::coupling::{ColliderCollisionDetector, ColliderCouplingManager},
    nphysics::object::{BodySet, ColliderSet},
    nphysics::world::GeometricalWorld,
};

pub struct LiquidWorld<N: RealField> {
    particle_radius: N,
    h: N,
    fluids: Vec<Fluid<N>>,
    boundaries: Vec<Boundary<N>>,
    solver: PBFSolver<N>,
    contact_manager: ContactManager<N>,
    timestep_manager: TimestepManager<N>,
    hgrid: HGrid<N, HGridEntry>,
}

impl<N: RealField> LiquidWorld<N> {
    pub fn new(particle_radius: N, smoothing_factor: N) -> Self {
        let h = particle_radius * smoothing_factor * na::convert(2.0);
        Self {
            particle_radius,
            h,
            fluids: Vec::new(),
            boundaries: Vec::new(),
            solver: PBFSolver::new(),
            contact_manager: ContactManager::new(),
            timestep_manager: TimestepManager::new(),
            hgrid: HGrid::new(h),
        }
    }

    pub fn step(&mut self, dt: N, gravity: &Vector<N>) {
        let step_start_time = instant::now();
        let mut remaining_time = dt;

        self.solver.step(
            dt,
            &self.timestep_manager,
            &mut self.contact_manager,
            gravity,
            self.h,
            self.particle_radius,
            &mut self.fluids,
            &self.boundaries,
            &mut self.hgrid,
        );

        println!("Total step time: {}ms", instant::now() - step_start_time);
    }

    #[cfg(feature = "nphysics")]
    pub fn step_with_coupling<Bodies, Colliders>(
        &mut self,
        dt: N,
        gravity: &Vector<N>,
        geometrical_world: &GeometricalWorld<N, Bodies::Handle, Colliders::Handle>,
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
                &self.contact_manager.fluid_fluid_contacts,
                &self.contact_manager.fluid_boundary_contacts,
            );

            let substep_inv_dt = N::one() / substep_dt;

            self.solver.init_with_fluids(&self.fluids);
            self.solver
                .predict_advection(substep_dt, gravity, &self.fluids);

            self.hgrid.clear();
            geometry::insert_fluids_to_grid(
                &self.fluids,
                Some(self.solver.position_changes()),
                &mut self.hgrid,
            );

            //            let mut detector = ColliderCollisionDetector::new();
            //            detector.detect_contacts(
            //                substep_dt,
            //                substep_inv_dt,
            //                gravity,
            //                &self.fluids,
            //                self.particle_radius,
            //                &self.hgrid,
            //                geometrical_world,
            //                bodies,
            //                colliders,
            //            );

            coupling.update_boundaries(
                self.h,
                colliders,
                &mut self.boundaries,
                &self.fluids,
                self.solver.position_changes_mut(),
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
                &self.timestep_manager,
                &mut self.contact_manager,
                gravity,
                self.h,
                self.particle_radius,
                &mut self.fluids,
                &self.boundaries,
                &mut self.hgrid,
            );

            remaining_time -= substep_dt;
            println!("Performed substep: {}", substep_dt);
        }

        coupling.transmit_forces(&mut self.boundaries, &self.fluids, bodies, colliders);
    }

    pub fn add_fluid(&mut self, fluid: Fluid<N>) {
        self.fluids.push(fluid)
    }
    pub fn add_boundary(&mut self, boundary: Boundary<N>) -> BoundaryHandle {
        let handle = self.boundaries.len();
        self.boundaries.push(boundary);
        handle
    }

    pub fn fluids(&self) -> &[Fluid<N>] {
        &self.fluids
    }
    pub fn boundaries(&self) -> &[Boundary<N>] {
        &self.boundaries
    }

    pub fn h(&self) -> N {
        self.h
    }
    pub fn particle_radius(&self) -> N {
        self.particle_radius
    }
}
