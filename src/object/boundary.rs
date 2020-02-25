use crate::math::{Isometry, Point, Vector};
use crate::object::{ContiguousArena, ContiguousArenaIndex};
use na::{self, RealField};
use std::sync::RwLock;

/// A boundary object.
///
/// A boundary object is composed of static particles, or of particles coupled with non-fluid bodies.
pub struct Boundary<N: RealField> {
    /// The world-space position of the boundary particles.
    pub positions: Vec<Point<N>>,
    /// The artificial velocities of each boundary particle.
    pub velocities: Vec<Vector<N>>,
    /// The volume computed for each boundary particle.
    pub volumes: Vec<N>,
    /// The forces applied to each particle of this boundary object.
    /// If this is set to `None` (which is the default), the boundary won't receive any
    /// force for fluids.
    pub forces: Option<RwLock<Vec<Vector<N>>>>,
}

impl<N: RealField> Boundary<N> {
    /// Initialize a boundary object with the given particles.
    pub fn new(particle_positions: Vec<Point<N>>) -> Self {
        let num_particles = particle_positions.len();
        let velocities = std::iter::repeat(Vector::zeros())
            .take(num_particles)
            .collect();
        let volumes = std::iter::repeat(N::zero()).take(num_particles).collect();

        Self {
            positions: particle_positions,
            velocities,
            volumes,
            forces: None,
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
        if let Some(forces) = &self.forces {
            let mut forces = forces.write().unwrap();
            forces[i] += f;
        }
    }

    /// Clears all the forces applied to this boundary object's particles.
    pub fn clear_forces(&mut self, resize_buffer: bool) {
        if let Some(forces) = &mut self.forces {
            let forces = forces.get_mut().unwrap();

            if resize_buffer {
                forces.resize(self.positions.len(), Vector::zeros());
            }

            for f in forces {
                f.fill(N::zero())
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct BoundaryHandle(ContiguousArenaIndex);
pub type BoundarySet<N> = ContiguousArena<BoundaryHandle, Boundary<N>>;

impl From<ContiguousArenaIndex> for BoundaryHandle {
    #[inline]
    fn from(i: ContiguousArenaIndex) -> Self {
        BoundaryHandle(i)
    }
}

impl Into<ContiguousArenaIndex> for BoundaryHandle {
    #[inline]
    fn into(self) -> ContiguousArenaIndex {
        self.0
    }
}
