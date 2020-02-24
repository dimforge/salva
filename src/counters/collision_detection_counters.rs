use crate::counters::Timer;
use std::fmt::{Display, Formatter, Result};

/// Performance counters related to collision detection.
#[derive(Default, Clone, Copy)]
pub struct CollisionDetectionCounters {
    /// Number of contacts detected.
    pub ncontacts: usize,
    /// Time spent updating the boundary particles.
    pub boundary_update_time: Timer,
    /// Time spent for the broad-phase of the collision detection.
    pub grid_insertion_time: Timer,
    /// Time spent for the narrow-phase of the collision detection.
    pub neighborhood_search_time: Timer,
    /// Time spent to sort the contacts.
    pub contact_sorting_time: Timer,
}

impl CollisionDetectionCounters {
    /// Creates a new counter initialized to zero.
    pub fn new() -> Self {
        CollisionDetectionCounters {
            ncontacts: 0,
            boundary_update_time: Timer::new(),
            grid_insertion_time: Timer::new(),
            neighborhood_search_time: Timer::new(),
            contact_sorting_time: Timer::new(),
        }
    }

    pub fn enable(&mut self) {
        self.boundary_update_time.enable();
        self.grid_insertion_time.enable();
        self.neighborhood_search_time.enable();
        self.contact_sorting_time.enable();
    }

    pub fn disable(&mut self) {
        self.boundary_update_time.disable();
        self.grid_insertion_time.disable();
        self.neighborhood_search_time.disable();
        self.contact_sorting_time.disable();
    }

    pub fn reset(&mut self) {
        self.ncontacts = 0;
        self.boundary_update_time.reset();
        self.grid_insertion_time.reset();
        self.neighborhood_search_time.reset();
        self.contact_sorting_time.reset();
    }
}

impl Display for CollisionDetectionCounters {
    fn fmt(&self, f: &mut Formatter) -> Result {
        writeln!(f, "Number of contacts: {}", self.ncontacts)?;
        writeln!(f, "Boundary update time: {}", self.boundary_update_time)?;
        writeln!(f, "Grid insertion time: {}", self.grid_insertion_time)?;
        writeln!(
            f,
            "Neighborhood search time: {}",
            self.neighborhood_search_time
        );
        writeln!(f, "Contact sorting time: {}", self.contact_sorting_time)
    }
}
