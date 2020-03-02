use crate::counters::Timer;
use std::fmt::{Display, Formatter, Result};

/// Performance counters related to each stage of the time step.
#[derive(Default, Clone, Copy)]
pub struct StagesCounters {
    /// Total time spent for the collision detection.
    pub collision_detection_time: Timer,
    /// Total time spent for the computation and integration of forces.
    pub solver_time: Timer,
}

impl StagesCounters {
    /// Create a new counter initialized to zero.
    pub fn new() -> Self {
        StagesCounters {
            collision_detection_time: Timer::new(),
            solver_time: Timer::new(),
        }
    }

    /// Enables all the counters for the simulation stages.
    pub fn enable(&mut self) {
        self.collision_detection_time.enable();
        self.solver_time.enable();
    }

    /// Disables all the counters for the simulation stages.
    pub fn disable(&mut self) {
        self.collision_detection_time.disable();
        self.solver_time.disable();
    }

    /// Resets to zero all the counters for the simulation stages.
    pub fn reset(&mut self) {
        self.collision_detection_time.reset();
        self.solver_time.reset();
    }
}

impl Display for StagesCounters {
    fn fmt(&self, f: &mut Formatter) -> Result {
        writeln!(
            f,
            "Collision detection time: {}",
            self.collision_detection_time
        )?;
        writeln!(f, "Solver time: {}", self.solver_time)
    }
}
