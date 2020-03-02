use crate::geometry::ParticlesContacts;
use crate::object::{Boundary, Fluid};
use crate::TimestepManager;
use na::RealField;

/// Trait implemented by non-pressure forces.
///
/// This includes all non-pressure forces internal to a same fluid, or acting
/// between a fluid and a boundary.
pub trait NonPressureForce<N: RealField>: Send + Sync {
    /// Compute and applies the non-pressure forces to the given fluid.
    ///
    /// The force application should result in adding accelerations to the
    /// `fluid.accelerations` field.
    fn solve(
        &mut self,
        timestep: &TimestepManager<N>,
        kernel_radius: N,
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid_boundaries_contacts: &ParticlesContacts<N>,
        fluid: &mut Fluid<N>,
        boundaries: &[Boundary<N>],
        densities: &[N],
    );

    /// Apply the given permutation to all relevant field of this non-pressure force.
    ///
    /// This is currently not used so it can be left empty.
    fn apply_permutation(&mut self, _permutation: &[usize]) {}
}
