use crate::geometry::HGrid;
use crate::math::Vector;
use crate::object::Boundary;
use crate::object::Fluid;
use na::RealField;
use std::ops::Range;

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

#[derive(Clone, Debug)]
pub struct Contact<N: RealField> {
    pub i: usize,
    pub i_model: usize,
    pub j: usize,
    pub j_model: usize,
    pub weight: N,
    pub gradient: Vector<N>,
}

#[derive(Clone, Debug)]
pub struct ParticlesContacts<N: RealField> {
    // All the particle contact for one model.
    // Contacts involving the same particle `i` are adjascent.
    contacts: Vec<Contact<N>>,
    // There is one range per perticle.
    contact_ranges: Vec<Range<usize>>,
}

impl<N: RealField> ParticlesContacts<N> {
    pub fn new() -> Self {
        Self {
            contacts: Vec::new(),
            contact_ranges: Vec::new(),
        }
    }

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

pub fn insert_fluids_to_grid<N: RealField>(
    dt: N,
    fluids: &[Fluid<N>],
    fluids_delta_vel: Option<&[Vec<Vector<N>>]>,
    grid: &mut HGrid<N, HGridEntry>,
) {
    for (fluid_id, fluid) in fluids.iter().enumerate() {
        if let Some(deltas) = fluids_delta_vel {
            let fluid_deltas = &deltas[fluid_id];

            for (particle_id, (point, vel)) in fluid
                .positions
                .iter()
                .zip(fluid.velocities.iter())
                .enumerate()
            {
                grid.insert(
                    &(point + (vel + fluid_deltas[particle_id]) * dt),
                    HGridEntry::FluidParticle(fluid_id, particle_id),
                );
            }
        } else {
            for (particle_id, point) in fluid.positions.iter().enumerate() {
                grid.insert(&point, HGridEntry::FluidParticle(fluid_id, particle_id));
            }
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
    dt: N,
    h: N,
    fluids: &[Fluid<N>],
    boundaries: &[Boundary<N>],
    fluids_delta_vel: Option<&[Vec<Vector<N>>]>,
    fluid_fluid_contacts: &mut Vec<ParticlesContacts<N>>,
    fluid_boundary_contacts: &mut Vec<ParticlesContacts<N>>,
    boundary_boundary_contacts: &mut Vec<ParticlesContacts<N>>,
    grid: &HGrid<N, HGridEntry>,
) {
    fluid_fluid_contacts.clear();
    fluid_boundary_contacts.clear();
    boundary_boundary_contacts.clear();

    fluid_fluid_contacts.resize(fluids.len(), ParticlesContacts::new());
    fluid_boundary_contacts.resize(fluids.len(), ParticlesContacts::new());
    boundary_boundary_contacts.resize(boundaries.len(), ParticlesContacts::new());

    for (fluid, contacts) in fluids.iter().zip(fluid_fluid_contacts.iter_mut()) {
        contacts.contact_ranges.resize(fluid.num_particles(), 0..0)
    }

    for (fluid, contacts) in fluids.iter().zip(fluid_boundary_contacts.iter_mut()) {
        contacts.contact_ranges.resize(fluid.num_particles(), 0..0)
    }

    for (boundary, contacts) in boundaries.iter().zip(boundary_boundary_contacts.iter_mut()) {
        contacts
            .contact_ranges
            .resize(boundary.num_particles(), 0..0)
    }

    for (cell, curr_particles) in grid.cells() {
        let neighbors: Vec<_> = grid.neighbor_cells(cell, h).collect();

        for entry in curr_particles {
            match entry {
                HGridEntry::BoundaryParticle(boundary_i, particle_i) => {
                    let bb_contacts = &mut boundary_boundary_contacts[*boundary_i];
                    let bb_start = bb_contacts.contacts.len();
                    bb_contacts.contact_ranges[*particle_i] = bb_start..bb_start;

                    for (_, nbh_particles) in &neighbors {
                        for entry in *nbh_particles {
                            // NOTE: we are not interested by boundary-fluid contacts.
                            // Those will already be detected as fluid-boundary contacts instead.
                            if let HGridEntry::BoundaryParticle(boundary_j, particle_j) = entry {
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

                                    bb_contacts.contacts.push(contact);
                                    bb_contacts.contact_ranges[*particle_i].end += 1;
                                }
                            }
                        }
                    }
                }
                HGridEntry::FluidParticle(fluid_i, particle_i) => {
                    let ff_contacts = &mut fluid_fluid_contacts[*fluid_i];
                    let fb_contacts = &mut fluid_boundary_contacts[*fluid_i];
                    let ff_start = ff_contacts.contacts.len();
                    let fb_start = fb_contacts.contacts.len();

                    ff_contacts.contact_ranges[*particle_i] = ff_start..ff_start;
                    fb_contacts.contact_ranges[*particle_i] = fb_start..fb_start;

                    for (_, nbh_particles) in &neighbors {
                        for entry in *nbh_particles {
                            let (fluid_j, particle_j, is_boundary_j) = entry.into_tuple();
                            let mut pi = fluids[*fluid_i].positions[*particle_i];
                            let mut vi = fluids[*fluid_i].velocities[*particle_i];
                            let (mut pj, mut vj) = if is_boundary_j {
                                (boundaries[fluid_j].positions[particle_j], Vector::zeros())
                            } else {
                                (
                                    fluids[fluid_j].positions[particle_j],
                                    fluids[fluid_j].velocities[particle_j],
                                )
                            };

                            if let Some(deltas) = fluids_delta_vel {
                                vi += deltas[*fluid_i][*particle_i];

                                if !is_boundary_j {
                                    vj += deltas[fluid_j][particle_j];
                                }
                            }

                            pi += vi * dt;
                            pj += vj * dt;

                            if na::distance_squared(&pi, &pj) <= h * h {
                                let contact = Contact {
                                    i_model: *fluid_i,
                                    j_model: fluid_j,
                                    i: *particle_i,
                                    j: particle_j,
                                    weight: N::zero(),
                                    gradient: Vector::zeros(),
                                };

                                if is_boundary_j {
                                    fb_contacts.contacts.push(contact);
                                    fb_contacts.contact_ranges[*particle_i].end += 1;
                                } else {
                                    ff_contacts.contacts.push(contact);
                                    ff_contacts.contact_ranges[*particle_i].end += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
