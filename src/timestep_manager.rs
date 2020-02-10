use na::RealField;

use crate::object::Fluid;

/// Structure responsible for regulatin the timestep length of the simulation.
pub struct TimestepManager<N: RealField> {
    cfl_coeff: N,
    min_substep_coeff: N,
}

impl<N: RealField> TimestepManager<N> {
    /// Initialize a new timestep manager with default parameters.
    pub fn new() -> Self {
        Self {
            cfl_coeff: na::convert(1.0),
            min_substep_coeff: na::convert(1.0), // 0.2),
        }
    }

    fn max_substep(&self, particle_radius: N, fluids: &[Fluid<N>]) -> N {
        let mut max_sq_vel = N::zero();
        for v in fluids.iter().flat_map(|f| f.velocities.iter()) {
            max_sq_vel = max_sq_vel.max(v.norm_squared());
        }

        particle_radius * na::convert(2.0) / max_sq_vel.sqrt() * self.cfl_coeff
    }

    pub(crate) fn compute_substep(
        &self,
        total_step_size: N,
        remaining_time: N,
        particle_radius: N,
        fluids: &[Fluid<N>],
    ) -> N {
        let min_substep = total_step_size * self.min_substep_coeff;
        let max_substep = self.max_substep(particle_radius, fluids);

        if remaining_time - max_substep < min_substep {
            remaining_time
        } else {
            max_substep
        }
    }
}
