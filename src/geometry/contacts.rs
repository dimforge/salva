use crate::math::Vector;
use na::RealField;
use std::ops::Range;

pub struct Contact<N: RealField> {
    pub i: usize,
    pub i_model: usize,
    pub j: usize,
    pub j_model: usize,
    pub weight: N,
    pub gradient: Vector<N>,
}

pub struct ParticlesContacts<N: RealField> {
    contacts: Vec<Contact<N>>,
    contact_ranges: Vec<Range<usize>>,
}

impl<N: RealField> ParticlesContacts<N> {
    pub fn particle_contacts(&self, i: usize) -> &[Contact<N>] {
        &self.contacts[self.contact_ranges[i].clone()]
    }

    pub fn particle_contacts_mut(&mut self, i: usize) -> &mut [Contact<N>] {
        &mut self.contacts[self.contact_ranges[i].clone()]
    }

    pub fn contacts(&self) -> &[Contact<N>] {
        &self.contacts[..]
    }

    pub fn contacts_mut(&mut self) -> &mut [Contact<N>] {
        &mut self.contacts[..]
    }
}
