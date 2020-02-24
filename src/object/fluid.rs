use crate::math::{Isometry, Point, Vector};
use crate::object::{ContiguousArena, ContiguousArenaIndex};
use crate::solver::NonPressureForce;
use na::{self, DVector, RealField};

/// A fluid object.
///
/// A fluid object is composed of movable particles with additional properties like viscosity.
pub struct Fluid<N: RealField> {
    /// Nonpressure forces this fluid is subject to.
    pub nonpressure_forces: Vec<Box<dyn NonPressureForce<N>>>,
    /// The world-space position of the fluid particles.
    pub positions: Vec<Point<N>>,
    /// The velocities of the fluid particles.
    pub velocities: Vec<Vector<N>>,
    /// The volume of the fluid particles.
    pub volumes: DVector<N>,
    /// The rest density of this fluid.
    pub density0: N,
}

impl<N: RealField> Fluid<N> {
    /// Initializes a new fluid object with the given particle positions, particle radius, density, and viscosity.
    ///
    /// The particle radius should be the same as the radius used to initialize the liquid world.
    pub fn new(
        particle_positions: Vec<Point<N>>,
        particle_radius: N, // XXX: remove this parameter since it is already defined by the liquid world.
        density0: N,
    ) -> Self {
        let num_particles = particle_positions.len();
        let velocities = std::iter::repeat(Vector::zeros())
            .take(num_particles)
            .collect();
        // The volume of a fluid is computed as the volume of a cuboid of half-width equal to particle_radius.
        // It is multiplied by 0.8 so that there is no pressure when the cuboids are aligned on a grid.
        // This mass computation method is inspired from the SplishSplash project.
        #[cfg(feature = "dim2")]
        let particle_volume = particle_radius * particle_radius * na::convert(4.0 * 0.8);
        #[cfg(feature = "dim3")]
        let particle_volume =
            particle_radius * particle_radius * particle_radius * na::convert(8.0 * 0.8);

        Self {
            nonpressure_forces: Vec::new(),
            positions: particle_positions,
            velocities,
            volumes: DVector::repeat(num_particles, particle_volume),
            density0,
        }
    }

    pub fn z_sort(&mut self) {
        let order = crate::z_order::compute_points_z_order(&self.positions);
        self.positions = crate::z_order::apply_permutation(&order, &self.positions);
        self.velocities = crate::z_order::apply_permutation(&order, &self.velocities);
        self.volumes = DVector::from_vec(crate::z_order::apply_permutation(
            &order,
            self.volumes.as_slice(),
        ));

        for forces in &mut self.nonpressure_forces {
            forces.apply_permutation(&order);
        }
    }

    pub fn transform_by(&mut self, t: &Isometry<N>) {
        self.positions.iter_mut().for_each(|p| *p = t * *p)
    }

    /// The number of particles on this fluid.
    pub fn num_particles(&self) -> usize {
        self.positions.len()
    }

    /// Computes the AABB of this fluid.
    #[cfg(feature = "nphysics")]
    pub fn compute_aabb(&self, particle_radius: N) -> ncollide::bounding_volume::AABB<N> {
        use ncollide::bounding_volume::{self, BoundingVolume};
        bounding_volume::local_point_cloud_aabb(&self.positions).loosened(particle_radius)
    }

    /// The mass of the `i`-th particle of this fluid.
    pub fn particle_mass(&self, i: usize) -> N {
        self.volumes[i] * self.density0
    }

    /// The inverse mass of the `i`-th particle of this fluid.
    ///
    /// Returns 0 if the `i`-th particle has a zero mass.
    pub fn particle_inv_mass(&self, i: usize) -> N {
        if self.volumes[i].is_zero() {
            N::zero()
        } else {
            N::one() / (self.volumes[i] * self.density0)
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct FluidHandle(ContiguousArenaIndex);
pub type FluidSet<N> = ContiguousArena<FluidHandle, Fluid<N>>;

impl From<ContiguousArenaIndex> for FluidHandle {
    #[inline]
    fn from(i: ContiguousArenaIndex) -> Self {
        FluidHandle(i)
    }
}

impl Into<ContiguousArenaIndex> for FluidHandle {
    #[inline]
    fn into(self) -> ContiguousArenaIndex {
        self.0
    }
}
