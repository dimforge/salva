use crate::geometry::ParticlesContacts;
use crate::object::{Boundary, Fluid};
use crate::TimestepManager;
use na::RealField;

pub trait NonPressureForce<N: RealField>: Send + Sync {
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

    fn apply_permutation(&mut self, permutation: &[usize]);
}
