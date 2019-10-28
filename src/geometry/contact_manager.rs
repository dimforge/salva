use crate::geometry::{self, HGrid, HGridEntry, ParticlesContacts};
use crate::math::Vector;
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
    pub fn new() -> Self {
        Self {
            fluid_fluid_contacts: Vec::new(),
            fluid_boundary_contacts: Vec::new(),
            boundary_boundary_contacts: Vec::new(),
        }
    }

    pub fn update_contacts(
        &mut self,
        h: N,
        fluids: &[Fluid<N>],
        boundaries: &[Boundary<N>],
        fluids_delta_pos: Option<&[Vec<Vector<N>>]>,
        hgrid: &HGrid<N, HGridEntry>,
    ) {
        geometry::compute_contacts(
            h,
            &fluids,
            &boundaries,
            fluids_delta_pos,
            &mut self.fluid_fluid_contacts,
            &mut self.fluid_boundary_contacts,
            &mut self.boundary_boundary_contacts,
            hgrid,
        );
    }
}
