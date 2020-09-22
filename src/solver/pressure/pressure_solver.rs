use crate::counters::Counters;
use crate::geometry::ContactManager;
use crate::math::{Real, Vector};
use crate::object::{Boundary, Fluid};
use crate::TimestepManager;

/// Trait implemented by pressure solvers.
pub trait PressureSolver {
    /// Initialize this solver with the given fluids.
    fn init_with_fluids(&mut self, fluids: &[Fluid]);

    /// Initialize this solver with the given boundaries.
    fn init_with_boundaries(&mut self, boundaries: &[Boundary]);

    /// Predicts advection with the given gravity.
    fn predict_advection(
        &mut self,
        timestep: &TimestepManager,
        kernel_radius: Real,
        contact_manager: &ContactManager,
        gravity: &Vector<Real>,
        fluids: &mut [Fluid],
        boundaries: &[Boundary],
    );

    /// Evaluate the SPH kernels for all the contacts in `contact_manager`.
    fn evaluate_kernels(
        &mut self,
        kernel_radius: Real,
        contact_manager: &mut ContactManager,
        fluids: &[Fluid],
        boundaries: &[Boundary],
    );

    /// Compute the densities of all the boundary and fluid particles.
    fn compute_densities(
        &mut self,
        contact_manager: &ContactManager,
        fluids: &[Fluid],
        boundaries: &mut [Boundary],
    );

    /// Solves pressure and non-pressure force for the given fluids and boundaries.
    ///
    /// Both `self.init_with_fluids` and `self.init_with_boundaries` must be called before this
    /// method.
    fn step(
        &mut self,
        counters: &mut Counters,
        timestep: &mut TimestepManager,
        gravity: &Vector<Real>,
        contact_manager: &mut ContactManager,
        kernel_radius: Real,
        fluids: &mut [Fluid],
        boundaries: &[Boundary],
    );
}
