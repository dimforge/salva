use crate::math::Vector;
use crate::object::Fluid;
use na::RealField;

pub trait NonPressureForce<N: RealField>: Sync {
    fn solve(
        &mut self,
        dt: N,
        kernel_radius: N,
        fluid: &Fluid<N>,
        velocity_changes: &mut [Vector<N>],
    );
}
