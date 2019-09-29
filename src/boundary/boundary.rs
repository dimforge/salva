use crate::math::{AngularVector, Point, Vector};
use na::{self, DVector, RealField};
use std::collections::HashMap;
use std::sync::RwLock;

pub type BoundaryHandle = usize;

pub struct Boundary<N: RealField> {
    pub positions: Vec<Point<N>>,
    pub velocities: Vec<Vector<N>>,
    assembly_id: usize,
    force: RwLock<(Vector<N>, AngularVector<N>)>,
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
            assembly_id: 0,
            force: RwLock::new((Vector::zeros(), AngularVector::zeros())),
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
        let dpos = self.positions[i] - self.positions[0];
        let mut total_f = self.force.write().unwrap();
        total_f.0 += f;

        #[cfg(feature = "dim2")]
        {
            total_f.1 += AngularVector::new(dpos.perp(&f));
        }
        #[cfg(feature = "dim3")]
        {
            total_f.1 += dpos.cross(&f);
        }
    }

    pub fn clear_forces(&mut self) {
        let f = self.force.get_mut().unwrap();
        f.0.fill(N::zero());
        f.1.fill(N::zero());
    }

    pub fn force(&self) -> (Vector<N>, AngularVector<N>) {
        self.force.read().unwrap().clone()
    }
}
