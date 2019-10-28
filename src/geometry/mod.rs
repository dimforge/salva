//! Acceleration data structures for collision detection.

pub(crate) use self::contact_manager::ContactManager;
pub(crate) use self::contacts::{
    compute_contacts, insert_boundaries_to_grid, insert_fluids_to_grid, HGridEntry,
    ParticlesContacts,
};
pub(crate) use self::hgrid::HGrid;

mod contact_manager;
mod contacts;
mod hgrid;
