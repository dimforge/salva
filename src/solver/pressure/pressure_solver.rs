use na::RealField;

use crate::geometry::ContactManager;
use crate::math::Vector;
use crate::object::{Boundary, Fluid};

pub trait PressureSolver<N: RealField> {
    /// Gets the set of fluid particle velocity changes resulting from pressure resolution.
    fn velocity_changes(&self) -> &[Vec<Vector<N>>];

    /// Gets a mutable reference to the set of fluid particle velocity changes resulting from
    /// pressure resolution.
    fn velocity_changes_mut(&mut self) -> &mut [Vec<Vector<N>>];

    /// Initialize this solver with the given fluids.
    fn init_with_fluids(&mut self, fluids: &[Fluid<N>]);

    /// Initialize this solver with the given boundaries.
    fn init_with_boundaries(&mut self, boundaries: &[Boundary<N>]);

    /// Predicts advection with the given gravity.
    fn predict_advection(&mut self, dt: N, gravity: &Vector<N>, fluids: &[Fluid<N>]);

    /// Solves pressure and non-pressure force for the given fluids and boundaries.
    ///
    /// Both `self.init_with_fluids` and `self.init_with_boundaries` must be called before this
    /// method.
    fn step(
        &mut self,
        dt: N,
        contact_manager: &mut ContactManager<N>,
        kernel_radius: N,
        fluids: &mut [Fluid<N>],
        boundaries: &[Boundary<N>],
    );
}
