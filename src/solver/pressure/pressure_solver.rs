use na::RealField;

use crate::counters::Counters;
use crate::geometry::ContactManager;
use crate::math::Vector;
use crate::object::{Boundary, Fluid};
use crate::TimestepManager;

/// Trait implemented by pressure solvers.
pub trait PressureSolver<N: RealField> {
    /// Initialize this solver with the given fluids.
    fn init_with_fluids(&mut self, fluids: &[Fluid<N>]);

    /// Initialize this solver with the given boundaries.
    fn init_with_boundaries(&mut self, boundaries: &[Boundary<N>]);

    /// Predicts advection with the given gravity.
    fn predict_advection(
        &mut self,
        timestep: &TimestepManager<N>,
        kernel_radius: N,
        contact_manager: &ContactManager<N>,
        gravity: &Vector<N>,
        fluids: &mut [Fluid<N>],
        boundaries: &[Boundary<N>],
    );

    /// Evaluate the SPH kernels for all the contacts in `contact_manager`.
    fn evaluate_kernels(
        &mut self,
        kernel_radius: N,
        contact_manager: &mut ContactManager<N>,
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    );

    /// Compute the densities of all the boundary and fluid particles.
    fn compute_densities(
        &mut self,
        contact_manager: &ContactManager<N>,
        fluids: &[Fluid<N>],
        boundaries: &mut [Boundary<N>],
    );

    /// Solves pressure and non-pressure force for the given fluids and boundaries.
    ///
    /// Both `self.init_with_fluids` and `self.init_with_boundaries` must be called before this
    /// method.
    fn step(
        &mut self,
        counters: &mut Counters,
        timestep: &mut TimestepManager<N>,
        gravity: &Vector<N>,
        contact_manager: &mut ContactManager<N>,
        kernel_radius: N,
        fluids: &mut [Fluid<N>],
        boundaries: &[Boundary<N>],
    );
}
