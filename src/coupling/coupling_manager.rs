use na::RealField;

use crate::geometry::{HGrid, HGridEntry};
use crate::math::Vector;
use crate::object::{Boundary, BoundarySet, Fluid};

pub trait CouplingManager<N: RealField> {
    fn update_boundaries(
        &mut self,
        dt: N,
        h: N,
        hgrid: &HGrid<N, HGridEntry>,
        fluids: &mut [Fluid<N>],
        fluids_delta_vel: &mut [Vec<Vector<N>>],
        boundaries: &mut BoundarySet<N>,
    );

    fn transmit_forces(&mut self, boundaries: &BoundarySet<N>);
}

impl<N: RealField> CouplingManager<N> for () {
    fn update_boundaries(
        &mut self,
        _: N,
        _: N,
        _: &HGrid<N, HGridEntry>,
        _: &mut [Fluid<N>],
        _: &mut [Vec<Vector<N>>],
        _: &mut BoundarySet<N>,
    ) {
    }

    fn transmit_forces(&mut self, _: &BoundarySet<N>) {}
}
