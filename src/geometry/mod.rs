pub use self::contact_manager::ContactManager;
pub use self::contacts::{
    compute_contacts, insert_boundaries_to_grid, insert_fluids_to_grid, Contact, HGridEntry,
    ParticlesContacts,
};
pub use self::hgrid::HGrid;

mod contact_manager;
mod contacts;
mod hgrid;
