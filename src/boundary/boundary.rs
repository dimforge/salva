use crate::math::{AngularVector, Point, Vector};
use na::{self, DVector, RealField};
use std::collections::HashMap;
use std::sync::RwLock;

pub type BoundaryHandle = usize;

pub struct Boundary<N: RealField> {
    pub positions: Vec<Point<N>>,
    pub velocities: Vec<Vector<N>>,
    pub forces: RwLock<Vec<Vector<N>>>,
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
            forces: RwLock::new(Vec::new()),
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

    pub fn apply_force(&self, i: usize, f: Vector<N>) {
        let mut forces = self.forces.write().unwrap();
        forces[i] += f;
    }

    pub fn clear_forces(&mut self, resize_buffer: bool) {
        let forces = self.forces.get_mut().unwrap();

        if resize_buffer {
            forces.resize(self.positions.len(), Vector::zeros());
        }

        for f in forces {
            f.fill(N::zero())
        }
    }
}
