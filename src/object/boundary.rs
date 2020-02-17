use crate::math::{Isometry, Point, Vector};
use na::{self, RealField};
use std::sync::RwLock;

/// The unique identifiant of a boundary object.
pub type BoundaryHandle = usize;

/// A boundary object.
///
/// A boundary object is composed of static particles, or of particles coupled with non-fluid bodies.
pub struct Boundary<N: RealField> {
    /// The world-space position of the boundary particles.
    pub positions: Vec<Point<N>>,
    /// The artificial velocities of each boundary particle.
    pub velocities: Vec<Vector<N>>,
    /// The forces applied to each particle of this boundary object.
    pub forces: RwLock<Vec<Vector<N>>>,
}

impl<N: RealField> Boundary<N> {
    /// Initialize a boundary object with the given particles.
    pub fn new(particle_positions: Vec<Point<N>>) -> Self {
        let num_particles = particle_positions.len();
        let velocities = std::iter::repeat(Vector::zeros())
            .take(num_particles)
            .collect();

        Self {
            positions: particle_positions,
            velocities,
            forces: RwLock::new(Vec::new()),
        }
    }

    /// The number of particles of this boundary object.
    pub fn num_particles(&self) -> usize {
        self.positions.len()
    }

    pub fn transform_by(&mut self, pose: &Isometry<N>) {
        self.positions.iter_mut().for_each(|p| *p = pose * *p);
    }

    /// Apply a force `f` to the `i`-th particle of this boundary object.
    ///
    /// This call relies on thread-safe interior mutability.
    pub fn apply_force(&self, i: usize, f: Vector<N>) {
        let mut forces = self.forces.write().unwrap();
        forces[i] += f;
    }

    /// Clears all the forces applied to this boundary object's particles.
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
