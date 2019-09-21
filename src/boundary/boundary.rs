use crate::math::{Point, Vector};
use na::{self, DVector, RealField};

pub struct Boundary<N: RealField> {
    pub positions: Vec<Point<N>>,
    pub velocities: Vec<Vector<N>>,
    pub densities: DVector<N>,
    pub volumes: DVector<N>,
    assembly_id: usize,
}

impl<N: RealField> Boundary<N> {
    pub fn new(particle_positions: Vec<Point<N>>) -> Self {
        let num_particles = particle_positions.len();
        let velocities = std::iter::repeat(Vector::zeros())
            .take(num_particles)
            .collect();

        Self {
            positions: particle_positions,
            velocities,
            volumes: DVector::zeros(num_particles),
            densities: DVector::zeros(num_particles),
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
