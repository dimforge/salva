use na::RealField;

use crate::geometry::{HGrid, HGridEntry};
use crate::object::{BoundarySet, Fluid};
use crate::TimestepManager;

/// Trait that needs to be implemented by middlewares responsible for
/// coupling bodies from a rigid-body physic framework (nphysics, bullet, PhysX, etc.)
/// with boundary objects of salva.
pub trait CouplingManager<N: RealField> {
    /// Updates the boundary objects from the coupled bodies.
    ///
    /// The goal of this method is to update the particles composing the boundary
    /// objects so they reflect the state of the coupled body. Those particles
    /// will generally be samplings of the boundary of the object.
    /// This also updates the velocity at those particles.
    fn update_boundaries(
        &mut self,
        timestep: &TimestepManager<N>,
        h: N,
        particle_radius: N,
        hgrid: &HGrid<N, HGridEntry>,
        fluids: &mut [Fluid<N>],
        boundaries: &mut BoundarySet<N>,
    );

    /// Transmit forces from salva's boundary objects to the coupled bodies.
    fn transmit_forces(&mut self, boundaries: &BoundarySet<N>);
}

impl<N: RealField> CouplingManager<N> for () {
    fn update_boundaries(
        &mut self,
        _: &TimestepManager<N>,
        _: N,
        _: N,
        _: &HGrid<N, HGridEntry>,
        _: &mut [Fluid<N>],
        _: &mut BoundarySet<N>,
    ) {
    }

    fn transmit_forces(&mut self, _: &BoundarySet<N>) {}
}
