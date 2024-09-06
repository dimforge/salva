use std::fmt::{Display, Error, Formatter};
use web_time::Instant;

/// A timer.
#[derive(Copy, Clone, Debug, Default)]
pub struct Timer {
    enabled: bool,
    time: f64,
    start: Option<Instant>,
}

impl Timer {
    /// Creates a new timer initialized to zero and not started.
    pub fn new() -> Self {
        Timer {
            enabled: false,
            time: 0.0,
            start: None,
        }
    }

    /// Enables this timer.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disables this timer.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Resets the timer to 0.
    pub fn reset(&mut self) {
        self.time = 0.0
    }

    /// Start the timer.
    pub fn start(&mut self) {
        if self.enabled {
            self.time = 0.0;
            self.start = Some(Instant::now());
        }
    }

    /// Pause the timer.
    pub fn pause(&mut self) {
        if self.enabled {
            if let Some(start) = self.start {
                self.time += start.elapsed().as_secs_f64();
            }
            self.start = None;
        }
    }

    /// Resume the timer.
    pub fn resume(&mut self) {
        if self.enabled {
            self.start = Some(Instant::now());
        }
    }

    /// The measured time between the last `.start()` and `.pause()` calls.
    pub fn time(&self) -> f64 {
        self.time
    }
}

impl Display for Timer {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "{}s", self.time)
    }
}
