use crate::counters::Counters;
use crate::geometry::HGrid;
use crate::math::{Point, Real, Vector};
use crate::object::Boundary;
use crate::object::Fluid;

use std::sync::RwLock;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Copy, Clone, Debug)]
/// A particle inserted on a spacial grid.
pub enum HGridEntry {
    /// A fluid particle with its fluid ID and particle ID.
    FluidParticle(usize, usize),
    /// A fluid particle with its boundary ID and particle ID.
    BoundaryParticle(usize, usize),
}

impl HGridEntry {
    /// Returns (object ID, particle ID, is_boundary).
    ///
    /// The last tuple entry is `true` if this is a fluid particle, or `false` if it is a boundary particle.
    pub fn into_tuple(self) -> (usize, usize, bool) {
        match self {
            HGridEntry::FluidParticle(a, b) => (a, b, false),
            HGridEntry::BoundaryParticle(a, b) => (a, b, true),
        }
    }
}

#[derive(Copy, Clone, Debug)]
/// A contact between two particles.
///
/// If the contact is between two fluid particles, it is assumed "one-way", i.e., this contact can
/// only result in a force applied by the particle `j` to the particle `i`. The force applied by
/// `i` on `j` will result from another contacts.
/// In other words, for each par of distinct fluid particles, there will be be two symmetric contacts.
pub struct Contact {
    /// The index of the first particle involved in this contact.
    pub i: usize,
    /// The index of the first fluid involved in this contact.
    pub i_model: usize,
    /// The index of the second particle involved in this contact.
    pub j: usize,
    /// The index of the second fluid boundary involved in this contact.
    pub j_model: usize,
    /// The kernel evaluated at `xi - xj` where `xi` is the position of the
    /// particle `i`, and `xj` is the position of the particle `j`.
    pub weight: Real,
    /// The kernel gradient evaluated at `xi - xj` where `xi` is the position of the
    /// particle `i`, and `xj` is the position of the particle `j`.
    pub gradient: Vector<Real>,
}

impl Contact {
    /// Flips this contact by swapping `i` with `j`, `i_model` with `j_model`, and by negating the gradient.
    pub fn flip(&self) -> Self {
        Self {
            i: self.j,
            i_model: self.j_model,
            j: self.i,
            j_model: self.i_model,
            weight: self.weight,
            gradient: -self.gradient,
        }
    }

    /// Returns `true` if this contact involves a single particle with itself.
    pub fn is_same_particle_contact(&self) -> bool {
        self.i_model == self.j_model && self.i == self.j
    }

    /// Returns `true` if this contact involves two particles from the same fluid.
    pub fn is_same_model_contact(&self) -> bool {
        self.i_model == self.j_model
    }
}

#[derive(Debug)]
/// The set of contacts affecting the particles of a single fluid.
pub struct ParticlesContacts {
    // All the particle contact for one model.
    // `self.contacts[i]` contains all the contacts involving the particle `i`.
    contacts: Vec<RwLock<Vec<Contact>>>,
}

impl ParticlesContacts {
    /// Creates an empty set of contacts.
    pub fn new() -> Self {
        Self {
            contacts: Vec::new(),
        }
    }

    /// The set of contacts affecting the particle `i`.
    pub fn particle_contacts(&self, i: usize) -> &RwLock<Vec<Contact>> {
        &self.contacts[i]
    }

    /// The set of mutable contacts affecting the particle `i`.
    pub fn particle_contacts_mut(&mut self, i: usize) -> &mut RwLock<Vec<Contact>> {
        &mut self.contacts[i]
    }

    /// All the contacts in this set.
    ///
    /// The `self.contacts()[i]` contains all the contact affecting the particle `i`.
    pub fn contacts(&self) -> &[RwLock<Vec<Contact>>] {
        &self.contacts[..]
    }

    /// All the mutable contacts in this set.
    ///
    /// The `self.contacts()[i]` contains all the contact affecting the particle `i`.
    pub fn contacts_mut(&mut self) -> &mut [RwLock<Vec<Contact>>] {
        &mut self.contacts[..]
    }

    /// The total number of contacts in this set.
    pub fn len(&self) -> usize {
        self.contacts.iter().map(|c| c.read().unwrap().len()).sum()
    }

    /// Apply a permutation to this set of contacts.
    pub fn apply_permutation(&mut self, _permutation: &[usize]) {
        unimplemented!()
    }
}

/// Insert all the particles from the given fluids into the `grid`.
pub fn insert_fluids_to_grid(fluids: &[Fluid], grid: &mut HGrid<HGridEntry>) {
    for (fluid_id, fluid) in fluids.iter().enumerate() {
        for (particle_id, point) in fluid.positions.iter().enumerate() {
            grid.insert(&point, HGridEntry::FluidParticle(fluid_id, particle_id));
        }
    }
}

/// Insert all the particles from the given boundaries into the `grid`.
pub fn insert_boundaries_to_grid(boundaries: &[Boundary], grid: &mut HGrid<HGridEntry>) {
    for (boundary_id, boundary) in boundaries.iter().enumerate() {
        for (particle_id, point) in boundary.positions.iter().enumerate() {
            grid.insert(
                &point,
                HGridEntry::BoundaryParticle(boundary_id, particle_id),
            );
        }
    }
}

/// Compute all the contacts between the particles inserted in `grid`.
pub fn compute_contacts(
    counters: &mut Counters,
    h: Real,
    fluids: &[Fluid],
    boundaries: &[Boundary],
    fluid_fluid_contacts: &mut Vec<ParticlesContacts>,
    fluid_boundary_contacts: &mut Vec<ParticlesContacts>,
    boundary_boundary_contacts: &mut Vec<ParticlesContacts>,
    grid: &HGrid<HGridEntry>,
) {
    // Needed so the loop in -1..=1 bellow works.
    assert_eq!(h, grid.cell_width());
    counters.cd.neighborhood_search_time.resume();

    fluid_fluid_contacts.resize_with(fluids.len(), || ParticlesContacts::new());
    fluid_boundary_contacts.resize_with(fluids.len(), || ParticlesContacts::new());
    boundary_boundary_contacts.resize_with(boundaries.len(), || ParticlesContacts::new());

    for (fluid, contacts) in fluids.iter().zip(fluid_fluid_contacts.iter_mut()) {
        contacts
            .contacts
            .iter_mut()
            .for_each(|c| c.write().unwrap().clear());
        contacts
            .contacts
            .resize_with(fluid.num_particles(), || RwLock::new(Vec::new()))
    }

    for (fluid, contacts) in fluids.iter().zip(fluid_boundary_contacts.iter_mut()) {
        contacts
            .contacts
            .iter_mut()
            .for_each(|c| c.write().unwrap().clear());
        contacts
            .contacts
            .resize_with(fluid.num_particles(), || RwLock::new(Vec::new()))
    }

    for (boundary, contacts) in boundaries.iter().zip(boundary_boundary_contacts.iter_mut()) {
        contacts
            .contacts
            .iter_mut()
            .for_each(|c| c.write().unwrap().clear());
        contacts
            .contacts
            .resize_with(boundary.num_particles(), || RwLock::new(Vec::new()))
    }

    #[cfg(feature = "dim2")]
    let neighbours: [(i64, i64); 5] = [(0, 0), (0, 1), (1, -1), (1, 0), (1, 1)];
    #[cfg(feature = "dim3")]
    let neighbours: [(i64, i64, i64); 14] = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, -1),
        (0, 1, 0),
        (0, 1, 1),
        (1, -1, -1),
        (1, -1, 0),
        (1, -1, 1),
        (1, 0, -1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, -1),
        (1, 1, 0),
        (1, 1, 1),
    ];

    par_iter!(grid.inner_table()).for_each(|(curr_cell, curr_particles)| {
        for &val in neighbours.iter() {
            #[cfg(feature = "dim2")]
            let neighbor_cell = {
                let (i, j) = val;
                curr_cell + Vector::new(i, j)
            };
            #[cfg(feature = "dim3")]
            let neighbor_cell = {
                let (i, j, k) = val;
                curr_cell + Vector::new(i, j, k)
            };
            if let Some(neighbor_particles) = grid.cell(&neighbor_cell) {
                compute_contacts_for_pair_of_cells(
                    h,
                    fluids,
                    boundaries,
                    fluid_fluid_contacts,
                    fluid_boundary_contacts,
                    boundary_boundary_contacts,
                    curr_cell,
                    curr_particles,
                    &neighbor_cell,
                    neighbor_particles,
                );
            }
        }
    });

    counters.cd.neighborhood_search_time.pause();
}

fn compute_contacts_for_pair_of_cells(
    h: Real,
    fluids: &[Fluid],
    boundaries: &[Boundary],
    fluid_fluid_contacts: &[ParticlesContacts],
    fluid_boundary_contacts: &[ParticlesContacts],
    boundary_boundary_contacts: &[ParticlesContacts],
    curr_cell: &Point<i64>,
    curr_particles: &[HGridEntry],
    neighbor_cell: &Point<i64>,
    neighbor_particles: &[HGridEntry],
) {
    for entry in curr_particles {
        match entry {
            HGridEntry::BoundaryParticle(boundary_i, particle_i) => {
                for entry in neighbor_particles {
                    // NOTE: we are not interested by boundary-fluid contacts.
                    // Those will already be detected as fluid-boundary contacts instead.
                    match entry {
                        HGridEntry::BoundaryParticle(boundary_j, particle_j) => {
                            let bi = &boundaries[*boundary_i];
                            let bj = &boundaries[*boundary_j];
                            if !bi.fluid_interaction.test(bj.fluid_interaction) {
                                continue;
                            }

                            let pi = &bi.positions[*particle_i];
                            let pj = &bj.positions[*particle_j];

                            if na::distance_squared(pi, pj) <= h * h {
                                let contact = Contact {
                                    i_model: *boundary_i,
                                    j_model: *boundary_j,
                                    i: *particle_i,
                                    j: *particle_j,
                                    weight: na::zero::<Real>(),
                                    gradient: Vector::zeros(),
                                };

                                boundary_boundary_contacts[*boundary_i].contacts[*particle_i]
                                    .write()
                                    .unwrap()
                                    .push(contact);

                                if *curr_cell != *neighbor_cell {
                                    boundary_boundary_contacts[*boundary_j].contacts[*particle_j]
                                        .write()
                                        .unwrap()
                                        .push(contact.flip());
                                }
                            }
                        }
                        HGridEntry::FluidParticle(fluid_j, particle_j) => {
                            if *curr_cell == *neighbor_cell {
                                // This pair will already be handled by the case where particle_i is a
                                // fluid particle.
                                continue;
                            }
                            let bi = &boundaries[*boundary_i];
                            let fj = &fluids[*fluid_j];
                            if !bi.fluid_interaction.test(fj.boundary_interaction) {
                                continue;
                            }
                            let pi = &boundaries[*boundary_i].positions[*particle_i];
                            let pj = &fluids[*fluid_j].positions[*particle_j];

                            if na::distance_squared(pi, pj) <= h * h {
                                let contact = Contact {
                                    i_model: *fluid_j,
                                    j_model: *boundary_i,
                                    i: *particle_j,
                                    j: *particle_i,
                                    weight: na::zero::<Real>(),
                                    gradient: Vector::zeros(),
                                };

                                fluid_boundary_contacts[*fluid_j].contacts[*particle_j]
                                    .write()
                                    .unwrap()
                                    .push(contact);
                            }
                        }
                    }
                }
            }
            HGridEntry::FluidParticle(fluid_i, particle_i) => {
                for entry in neighbor_particles {
                    let (fluid_j, particle_j, is_boundary_j) = entry.into_tuple();
                    let pi = fluids[*fluid_i].positions[*particle_i];
                    let pj = if is_boundary_j {
                        let bj = &boundaries[fluid_j];
                        if !fluids[*fluid_i]
                            .boundary_interaction
                            .test(bj.fluid_interaction)
                        {
                            continue;
                        }
                        boundaries[fluid_j].positions[particle_j]
                    } else {
                        fluids[fluid_j].positions[particle_j]
                    };

                    if na::distance_squared(&pi, &pj) <= h * h {
                        assert!(na::distance_squared(&pj, &pi) <= h * h);
                        let contact = Contact {
                            i_model: *fluid_i,
                            j_model: fluid_j,
                            i: *particle_i,
                            j: particle_j,
                            weight: na::zero::<Real>(),
                            gradient: Vector::zeros(),
                        };

                        if is_boundary_j {
                            fluid_boundary_contacts[*fluid_i].contacts[*particle_i]
                                .write()
                                .unwrap()
                                .push(contact);
                        } else {
                            fluid_fluid_contacts[*fluid_i].contacts[*particle_i]
                                .write()
                                .unwrap()
                                .push(contact);

                            if *curr_cell != *neighbor_cell {
                                fluid_fluid_contacts[fluid_j].contacts[particle_j]
                                    .write()
                                    .unwrap()
                                    .push(contact.flip());
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Compute all the contacts between the particles of a single fluid object.
pub fn compute_self_contacts(h: Real, fluid: &Fluid, contacts: &mut ParticlesContacts) {
    contacts
        .contacts
        .iter_mut()
        .for_each(|c| c.write().unwrap().clear());

    contacts
        .contacts
        .resize_with(fluid.num_particles(), || RwLock::new(Vec::new()));

    let mut grid = HGrid::new(h);
    for (i, particle) in fluid.positions.iter().enumerate() {
        grid.insert(particle, i);
    }

    for (cell, curr_particles) in grid.cells() {
        let neighbors: Vec<_> = grid.neighbor_cells(cell, h).collect();

        for particle_i in curr_particles {
            for (_, nbh_particles) in &neighbors {
                for particle_j in *nbh_particles {
                    let pi = fluid.positions[*particle_i];
                    let pj = fluid.positions[*particle_j];

                    if na::distance_squared(&pi, &pj) <= h * h {
                        let contact = Contact {
                            i_model: 0,
                            j_model: 0,
                            i: *particle_i,
                            j: *particle_j,
                            weight: na::zero::<Real>(),
                            gradient: Vector::zeros(),
                        };

                        contacts.contacts[*particle_i]
                            .write()
                            .unwrap()
                            .push(contact);
                    }
                }
            }
        }
    }
}
