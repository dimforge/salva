use crate::geometry::{HGrid, HGridEntry};
use crate::math::Real;
use crate::object::{BoundarySet, Fluid};
use crate::TimestepManager;

/// Trait that needs to be implemented by middlewares responsible for
/// coupling bodies from a rigid-body physic framework (nphysics, bullet, PhysX, etc.)
/// with boundary objects of salva.
pub trait CouplingManager {
    /// Updates the boundary objects from the coupled bodies.
    ///
    /// The goal of this method is to update the particles composing the boundary
    /// objects so they reflect the state of the coupled body. Those particles
    /// will generally be samplings of the boundary of the object.
    /// This also updates the velocity at those particles.
    fn update_boundaries(
        &mut self,
        timestep: &TimestepManager,
        h: Real,
        particle_radius: Real,
        hgrid: &HGrid<HGridEntry>,
        fluids: &mut [Fluid],
        boundaries: &mut BoundarySet,
    );

    /// Transmit forces from salva's boundary objects to the coupled bodies.
    fn transmit_forces(&mut self, timestep: &TimestepManager, boundaries: &BoundarySet);
}

impl CouplingManager for () {
    fn update_boundaries(
        &mut self,
        _: &TimestepManager,
        _: Real,
        _: Real,
        _: &HGrid<HGridEntry>,
        _: &mut [Fluid],
        _: &mut BoundarySet,
    ) {
    }

    fn transmit_forces(&mut self, _: &TimestepManager, _: &BoundarySet) {}
}
