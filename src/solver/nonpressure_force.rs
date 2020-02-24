use crate::geometry::ParticlesContacts;
use crate::math::Vector;
use crate::object::Fluid;
use na::RealField;

pub trait NonPressureForce<N: RealField>: Send + Sync {
    fn solve(
        &mut self,
        dt: N,
        kernel_radius: N,
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid: &Fluid<N>,
        densities: &[N],
        velocity_changes: &mut [Vector<N>],
    );

    fn apply_permutation(&mut self, permutation: &[usize]);
}
