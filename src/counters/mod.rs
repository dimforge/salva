//! Counters for benchmarking various parts of the physics engine.

use std::fmt::{Display, Formatter, Result};

pub use self::collision_detection_counters::CollisionDetectionCounters;
pub use self::solver_counters::SolverCounters;
pub use self::stages_counters::StagesCounters;
pub use self::timer::Timer;

mod collision_detection_counters;
mod solver_counters;
mod stages_counters;
mod timer;

/// Aggregation of all the performances counters tracked by nphysics.
#[derive(Clone, Copy)]
pub struct Counters {
    /// Total number of substeps performed.
    pub nsubsteps: usize,
    /// Timer for a whole timestep.
    pub step_time: Timer,
    /// Timer used for debugging.
    pub custom: Timer,
    /// Counters of every stage of one time step.
    pub stages: StagesCounters,
    /// Counters of the collision-detection stage.
    pub cd: CollisionDetectionCounters,
    /// Counters of the constraints resolution and force computation stage.
    pub solver: SolverCounters,
}

impl Counters {
    /// Create a new set of counters initialized to wero.
    pub fn new() -> Self {
        Counters {
            nsubsteps: 0,
            step_time: Timer::new(),
            custom: Timer::new(),
            stages: StagesCounters::new(),
            cd: CollisionDetectionCounters::new(),
            solver: SolverCounters::new(),
        }
    }

    /// Resets to zero all the counters.
    pub fn reset(&mut self) {
        self.nsubsteps = 0;
        self.step_time.reset();
        self.custom.reset();
        self.stages.reset();
        self.cd.reset();
        self.solver.reset();
    }

    /// Enable all the counters.
    pub fn enable(&mut self) {
        self.step_time.enable();
        self.custom.enable();
        self.stages.enable();
        self.cd.enable();
        self.solver.enable();
    }

    /// Disable all the counters.
    pub fn disable(&mut self) {
        self.step_time.disable();
        self.custom.disable();
        self.stages.disable();
        self.cd.disable();
        self.solver.disable();
    }
}

impl Display for Counters {
    fn fmt(&self, f: &mut Formatter) -> Result {
        writeln!(f, "Total timestep time: {}", self.step_time)?;
        writeln!(f, "Num substeps: {}", self.nsubsteps)?;
        self.stages.fmt(f)?;
        self.cd.fmt(f)?;
        self.solver.fmt(f)?;
        writeln!(f, "Custom timer: {}", self.custom)
    }
}
