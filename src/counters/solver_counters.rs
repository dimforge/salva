use crate::counters::Timer;
use std::fmt::{Display, Formatter, Result};

/// Performance counters related to constraints resolution.
#[derive(Default, Clone, Copy)]
pub struct SolverCounters {
    /// Time spent for the resolution of non-pressure forces.
    pub non_pressure_resolution_time: Timer,
    /// Time spent for the resolution of pressure forces.
    pub pressure_resolution_time: Timer,
}

impl SolverCounters {
    /// Creates a new counter initialized to zero.
    pub fn new() -> Self {
        SolverCounters {
            non_pressure_resolution_time: Timer::new(),
            pressure_resolution_time: Timer::new(),
        }
    }

    /// Enables all the counters for the solver.
    pub fn enable(&mut self) {
        self.non_pressure_resolution_time.enable();
        self.pressure_resolution_time.enable();
    }

    /// Disables all the counters for the solver.
    pub fn disable(&mut self) {
        self.non_pressure_resolution_time.disable();
        self.pressure_resolution_time.disable();
    }

    /// Resets to zero all the counters for the solver.
    pub fn reset(&mut self) {
        self.non_pressure_resolution_time.reset();
        self.pressure_resolution_time.reset();
    }
}

impl Display for SolverCounters {
    fn fmt(&self, f: &mut Formatter) -> Result {
        writeln!(
            f,
            "Non-pressure resolution time: {}",
            self.non_pressure_resolution_time
        )?;
        writeln!(
            f,
            "Pressure resolution time: {}",
            self.pressure_resolution_time
        )
    }
}
