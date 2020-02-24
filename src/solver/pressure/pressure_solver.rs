use na::RealField;

use crate::counters::Counters;
use crate::geometry::ContactManager;
use crate::math::Vector;
use crate::object::{Boundary, Fluid};

pub trait PressureSolver<N: RealField> {
    /// Initialize this solver with the given fluids.
    fn init_with_fluids(&mut self, fluids: &[Fluid<N>]);

    /// Initialize this solver with the given boundaries.
    fn init_with_boundaries(&mut self, boundaries: &[Boundary<N>]);

    /// Predicts advection with the given gravity.
    fn predict_advection(
        &mut self,
        dt: N,
        inv_dt: N,
        kernel_radius: N,
        contact_manager: &ContactManager<N>,
        gravity: &Vector<N>,
        fluids: &mut [Fluid<N>],
    );

    fn evaluate_kernels(
        &mut self,
        kernel_radius: N,
        contact_manager: &mut ContactManager<N>,
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    );

    fn compute_densities(
        &mut self,
        contact_manager: &ContactManager<N>,
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
    );

    /// Solves pressure and non-pressure force for the given fluids and boundaries.
    ///
    /// Both `self.init_with_fluids` and `self.init_with_boundaries` must be called before this
    /// method.
    fn step(
        &mut self,
        counters: &mut Counters,
        dt: N,
        contact_manager: &mut ContactManager<N>,
        kernel_radius: N,
        fluids: &mut [Fluid<N>],
        boundaries: &[Boundary<N>],
    );
}
