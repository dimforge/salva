use na::RealField;

use crate::object::Fluid;

/// Structure responsible for regulating the timestep length of the simulation.
pub struct TimestepManager<N: RealField> {
    cfl_coeff: N,
    min_num_substeps: u32,
    max_num_substeps: u32,
}

impl<N: RealField> TimestepManager<N> {
    /// Initialize a new timestep manager with default parameters.
    pub fn new() -> Self {
        Self {
            cfl_coeff: na::convert(1.0),
            min_num_substeps: 1,
            max_num_substeps: 10,
        }
    }

    fn max_substep(&self, remaining_time: N, particle_radius: N, fluids: &[Fluid<N>]) -> N {
        let mut max_sq_vel = N::zero();
        for (v, a) in fluids
            .iter()
            .flat_map(|f| f.velocities.iter().zip(f.accelerations.iter()))
        {
            max_sq_vel = max_sq_vel.max((v + a * remaining_time).norm_squared());
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
        return remaining_time;
        if remaining_time < N::default_epsilon() {
            return N::zero();
        }

        let min_substep = total_step_size / na::convert(self.max_num_substeps as f64);
        let max_substep = total_step_size / na::convert(self.min_num_substeps as f64);
        let computed_substep = self.max_substep(remaining_time, particle_radius, fluids);
        let substep = na::clamp(computed_substep, min_substep, max_substep);

        // NOTE: we use the condition `remaining_time - substep < min_substep`
        // instead of the condition `remaining_time < substep`.
        // This avoid having a final substep of length smaller than `min_substep`.
        // Indeed, very large timestep variations can cause significant instabilities.
        if remaining_time - substep < min_substep - N::default_epsilon() {
            remaining_time
        } else {
            substep
        }
    }
}
