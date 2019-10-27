use na::RealField;

use crate::fluid::Fluid;
use crate::geometry::ParticlesContacts;

pub struct TimestepManager<N: RealField> {
    cfl_coeff: N,
    min_substep_coeff: N,
}

impl<N: RealField> TimestepManager<N> {
    pub fn new() -> Self {
        Self {
            cfl_coeff: na::convert(0.4),
            min_substep_coeff: na::convert(1.0), // 0.2),
        }
    }

    fn max_substep(
        &self,
        particle_radius: N,
        fluids: &[Fluid<N>],
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
    ) -> N
    {
        let mut max_sq_vel = N::zero();
        for v in fluids.iter().flat_map(|f| f.velocities.iter()) {
            max_sq_vel = max_sq_vel.max(v.norm_squared());
        }

        particle_radius * na::convert(2.0) / max_sq_vel.sqrt() * self.cfl_coeff
    }

    pub fn compute_substep(
        &self,
        total_step_size: N,
        remaining_time: N,
        particle_radius: N,
        fluids: &[Fluid<N>],
        fluid_fluid_contacts: &[ParticlesContacts<N>],
        fluid_boundary_contacts: &[ParticlesContacts<N>],
    ) -> N
    {
        let min_substep = total_step_size * self.min_substep_coeff;
        let max_substep = self.max_substep(
            particle_radius,
            fluids,
            fluid_fluid_contacts,
            fluid_boundary_contacts,
        );

        if remaining_time - max_substep < min_substep {
            remaining_time
        } else {
            max_substep
        }
    }
}
