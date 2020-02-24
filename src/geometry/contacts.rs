use crate::counters::Counters;
use crate::geometry::HGrid;
use crate::math::{Point, Vector};
use crate::object::Boundary;
use crate::object::Fluid;
use na::RealField;
use std::collections::HashSet;
use std::ops::Range;
use std::sync::RwLock;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Copy, Clone, Debug)]
pub enum HGridEntry {
    FluidParticle(usize, usize),
    BoundaryParticle(usize, usize),
}

impl HGridEntry {
    // The last tuple entry is `true` if this is a fluid particle, or `false` if it is a boundary particle.
    pub fn into_tuple(self) -> (usize, usize, bool) {
        match self {
            HGridEntry::FluidParticle(a, b) => (a, b, false),
            HGridEntry::BoundaryParticle(a, b) => (a, b, true),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Contact<N: RealField> {
    pub i: usize,
    pub i_model: usize,
    pub j: usize,
    pub j_model: usize,
    pub weight: N,
    pub gradient: Vector<N>,
}

impl<N: RealField> Contact<N> {
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

    pub fn is_same_particle_contact(&self) -> bool {
        self.i_model == self.j_model && self.i == self.j
    }

    pub fn is_same_model_contact(&self) -> bool {
        self.i_model == self.j_model
    }
}

#[derive(Debug)]
pub struct ParticlesContacts<N: RealField> {
    // All the particle contact for one model.
    // There is one vec per particle.
    contacts: Vec<RwLock<Vec<Contact<N>>>>,
}

impl<N: RealField> ParticlesContacts<N> {
    pub fn new() -> Self {
        Self {
            contacts: Vec::new(),
        }
    }

    pub fn particle_contacts(&self, i: usize) -> &RwLock<Vec<Contact<N>>> {
        &self.contacts[i]
    }

    pub fn particle_contacts_mut(&mut self, i: usize) -> &mut RwLock<Vec<Contact<N>>> {
        &mut self.contacts[i]
    }

    pub fn contacts(&self) -> &[RwLock<Vec<Contact<N>>>] {
        &self.contacts[..]
    }

    pub fn contacts_mut(&mut self) -> &mut [RwLock<Vec<Contact<N>>>] {
        &mut self.contacts[..]
    }

    pub fn len(&self) -> usize {
        self.contacts.iter().map(|c| c.read().unwrap().len()).sum()
    }

    pub fn apply_permutation(&mut self, permutation: &[usize]) {
        unimplemented!()
    }
}

pub fn insert_fluids_to_grid<N: RealField>(
    dt: N,
    fluids: &[Fluid<N>],
    grid: &mut HGrid<N, HGridEntry>,
) {
    for (fluid_id, fluid) in fluids.iter().enumerate() {
        for (particle_id, point) in fluid.positions.iter().enumerate() {
            grid.insert(&point, HGridEntry::FluidParticle(fluid_id, particle_id));
        }
    }
}

pub fn insert_boundaries_to_grid<N: RealField>(
    boundaries: &[Boundary<N>],
    grid: &mut HGrid<N, HGridEntry>,
) {
    for (boundary_id, boundary) in boundaries.iter().enumerate() {
        for (particle_id, point) in boundary.positions.iter().enumerate() {
            grid.insert(
                &point,
                HGridEntry::BoundaryParticle(boundary_id, particle_id),
            );
        }
    }
}

pub fn compute_contacts<N: RealField>(
    counters: &mut Counters,
    h: N,
    fluids: &[Fluid<N>],
    boundaries: &[Boundary<N>],
    fluid_fluid_contacts: &mut Vec<ParticlesContacts<N>>,
    fluid_boundary_contacts: &mut Vec<ParticlesContacts<N>>,
    boundary_boundary_contacts: &mut Vec<ParticlesContacts<N>>,
    grid: &HGrid<N, HGridEntry>,
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

    par_iter!(grid.inner_table()).for_each(|(curr_cell, curr_particles)| {
        for i in 0i64..=1 {
            for j in -1..=1 {
                for k in -1..=1 {
                    // Avoid visiting the same pair of cells twice.
                    if i == 0 {
                        if j == 0 {
                            if k < 0 {
                                continue;
                            }
                        } else if j < 0 {
                            continue;
                        }
                    }

                    let neighbor_cell = curr_cell + Vector::new(i, j, k);
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
            }
        }
    });

    counters.cd.neighborhood_search_time.pause();
}

pub fn compute_contacts_for_pair_of_cells<N: RealField>(
    h: N,
    fluids: &[Fluid<N>],
    boundaries: &[Boundary<N>],
    fluid_fluid_contacts: &[ParticlesContacts<N>],
    fluid_boundary_contacts: &[ParticlesContacts<N>],
    boundary_boundary_contacts: &[ParticlesContacts<N>],
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
                            let pi = &boundaries[*boundary_i].positions[*particle_i];
                            let pj = &boundaries[*boundary_j].positions[*particle_j];

                            if na::distance_squared(pi, pj) <= h * h {
                                let contact = Contact {
                                    i_model: *boundary_i,
                                    j_model: *boundary_j,
                                    i: *particle_i,
                                    j: *particle_j,
                                    weight: N::zero(),
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
                            let pi = &boundaries[*boundary_i].positions[*particle_i];
                            let pj = &fluids[*fluid_j].positions[*particle_j];

                            if na::distance_squared(pi, pj) <= h * h {
                                let contact = Contact {
                                    i_model: *fluid_j,
                                    j_model: *boundary_i,
                                    i: *particle_j,
                                    j: *particle_i,
                                    weight: N::zero(),
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
                            weight: N::zero(),
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

pub fn compute_self_contacts<N: RealField>(
    h: N,
    fluid: &Fluid<N>,
    contacts: &mut ParticlesContacts<N>,
) {
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
                            weight: N::zero(),
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
