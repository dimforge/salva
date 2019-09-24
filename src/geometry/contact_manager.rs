use crate::boundary::Boundary;
use crate::fluid::Fluid;
use crate::geometry::{self, ParticlesContacts};
use crate::math::Vector;
use na::RealField;

pub struct ContactManager<N: RealField> {
    pub fluid_fluid_contacts: Vec<ParticlesContacts<N>>,
    pub fluid_boundary_contacts: Vec<ParticlesContacts<N>>,
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
        fluid_delta_pos: Option<&[Vec<Vector<N>>]>,
    )
    {
        self.fluid_fluid_contacts.clear();
        self.fluid_boundary_contacts.clear();
        self.boundary_boundary_contacts.clear();

        geometry::compute_contacts(
            h,
            &fluids,
            &boundaries,
            fluid_delta_pos,
            &mut self.fluid_fluid_contacts,
            &mut self.fluid_boundary_contacts,
            &mut self.boundary_boundary_contacts,
        );
    }
}
