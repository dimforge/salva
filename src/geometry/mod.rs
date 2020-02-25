//! Acceleration data structures for collision detection.

pub use self::contact_manager::ContactManager;
pub use self::contacts::{
    compute_contacts, compute_self_contacts, insert_boundaries_to_grid, insert_fluids_to_grid,
    HGridEntry, ParticlesContacts,
};
pub use self::hgrid::HGrid;

mod contact_manager;
mod contacts;
mod hgrid;
