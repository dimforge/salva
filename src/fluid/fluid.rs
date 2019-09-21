use crate::math::{Point, Vector, DIM};
use na::{self, DVector, RealField};

pub struct Fluid<N: RealField> {
    pub positions: Vec<Point<N>>,
    pub velocities: Vec<Vector<N>>,
    pub volumes: DVector<N>,
    pub density0: N,
    assembly_id: usize,
}

impl<N: RealField> Fluid<N> {
    pub fn new(particle_positions: Vec<Point<N>>, particle_volume: N, density0: N) -> Self {
        let num_particles = particle_positions.len();
        let velocities = std::iter::repeat(Vector::zeros())
            .take(num_particles)
            .collect();

        Self {
            positions: particle_positions,
            velocities,
            volumes: DVector::repeat(num_particles, particle_volume),
            density0,
            assembly_id: 0,
        }
    }

    pub fn assembly_id(&self) -> usize {
        self.assembly_id
    }

    pub fn set_assembly_id(&mut self, id: usize) {
        self.assembly_id = id
    }

    pub fn num_particles(&self) -> usize {
        self.positions.len()
    }
}
