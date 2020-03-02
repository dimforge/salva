use crate::counters::Counters;
use crate::geometry::{self, HGrid, HGridEntry, ParticlesContacts};
use crate::object::Boundary;
use crate::object::Fluid;
use na::RealField;

/// Structure responsible for computing and grouping all the contact between fluid and boundary particles.
pub struct ContactManager<N: RealField> {
    /// All contacts detected between pairs of fluid partices.
    pub fluid_fluid_contacts: Vec<ParticlesContacts<N>>,
    /// All contacts detected between a fluid particle and a boundary particle.
    pub fluid_boundary_contacts: Vec<ParticlesContacts<N>>,
    /// All contacts detected between two boundary particles.
    pub boundary_boundary_contacts: Vec<ParticlesContacts<N>>,
}

impl<N: RealField> ContactManager<N> {
    /// Create a new contact manager.
    pub fn new() -> Self {
        Self {
            fluid_fluid_contacts: Vec::new(),
            fluid_boundary_contacts: Vec::new(),
            boundary_boundary_contacts: Vec::new(),
        }
    }

    /// The total number of contacts detected by this manager.
    ///
    /// Note that there will be two contact for each pair of distinct particles.
    pub fn ncontacts(&self) -> usize {
        self.fluid_fluid_contacts
            .iter()
            .map(|c| c.len())
            .sum::<usize>()
            + self
                .fluid_boundary_contacts
                .iter()
                .map(|c| c.len())
                .sum::<usize>()
            + self
                .boundary_boundary_contacts
                .iter()
                .map(|c| c.len())
                .sum::<usize>()
    }

    /// Computes all the contacts between the particles inserted on the provided spacial grid.
    pub fn update_contacts(
        &mut self,
        counters: &mut Counters,
        h: N,
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
        hgrid: &HGrid<N, HGridEntry>,
    ) {
        geometry::compute_contacts(
            counters,
            h,
            &fluids,
            &boundaries,
            &mut self.fluid_fluid_contacts,
            &mut self.fluid_boundary_contacts,
            &mut self.boundary_boundary_contacts,
            hgrid,
        );
    }
}
