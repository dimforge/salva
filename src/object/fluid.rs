use crate::math::{Isometry, Point, Real, Vector};
use crate::object::{ContiguousArena, ContiguousArenaIndex};
use crate::solver::NonPressureForce;

use num::Zero;

use super::interaction_groups::InteractionGroups;

/// A fluid object.
///
/// A fluid object is composed of movable particles with additional properties like viscosity.
pub struct Fluid {
    /// Nonpressure forces this fluid is subject to.
    pub nonpressure_forces: Vec<Box<dyn NonPressureForce>>,
    /// The world-space position of the fluid particles.
    pub positions: Vec<Point<Real>>,
    /// The velocities of the fluid particles.
    pub velocities: Vec<Vector<Real>>,
    /// The accelerations of the fluid particles.
    pub accelerations: Vec<Vector<Real>>,
    /// The volume of the fluid particles.
    pub volumes: Vec<Real>,
    /// The rest density of this fluid.
    pub density0: Real,
    /// Mask indicating what particles have been deleted.
    deleted_particles: Vec<bool>,
    /// Indicates if a bit of the `deleted_particles` mask has been set.
    num_deleted_particles: usize,
    /// The particles radius.
    particle_radius: Real,
    /// The groups controlling which static particles ([`Boundary`]) can interact with this fluid.
    pub boundary_interaction: InteractionGroups,
}

impl Fluid {
    /// Initializes a new fluid object with the given particle positions, particle radius, density, and viscosity.
    ///
    /// The particle radius should be the same as the radius used to initialize the liquid world.
    pub fn new(
        particle_positions: Vec<Point<Real>>,
        particle_radius: Real, // XXX: remove this parameter since it is already defined by the liquid world.
        density0: Real,
        boundary_interaction: InteractionGroups,
    ) -> Self {
        let num_particles = particle_positions.len();
        let velocities: Vec<_> = std::iter::repeat(Vector::zeros())
            .take(num_particles)
            .collect();
        let accelerations: Vec<_> = velocities.clone();

        let particle_volume = Self::particle_volume(particle_radius);

        Self {
            nonpressure_forces: Vec::new(),
            positions: particle_positions,
            velocities,
            accelerations,
            volumes: std::iter::repeat(particle_volume)
                .take(num_particles)
                .collect(),
            boundary_interaction,
            deleted_particles: std::iter::repeat(false).take(num_particles).collect(),
            num_deleted_particles: 0,
            density0,
            particle_radius,
        }
    }

    /// Mark the given particle to be deleted at the next timestep.
    pub fn delete_particle_at_next_timestep(&mut self, particle: usize) {
        if !self.deleted_particles[particle] {
            self.deleted_particles[particle] = true;
            self.num_deleted_particles += 1;
        }
    }

    /// The number of particles that will be deleted at the next timestep.
    pub fn num_deleted_particles(&self) -> usize {
        self.num_deleted_particles
    }

    /// The mask of particles that will be deleted at the next timestep.
    pub fn deleted_particles_mask(&self) -> &[bool] {
        &self.deleted_particles
    }

    pub(crate) fn apply_particles_removal(&mut self) {
        if self.num_deleted_particles != 0 {
            crate::helper::filter_from_mask(&self.deleted_particles, &mut self.positions);
            crate::helper::filter_from_mask(&self.deleted_particles, &mut self.velocities);
            crate::helper::filter_from_mask(&self.deleted_particles, &mut self.accelerations);
            crate::helper::filter_from_mask(&self.deleted_particles, &mut self.volumes);
            self.deleted_particles.truncate(self.positions.len());
            self.deleted_particles.iter_mut().for_each(|i| *i = false);
            self.num_deleted_particles = 0;
        }
    }

    /// The radius of this fluid's particles.
    pub fn particle_radius(&self) -> Real {
        self.particle_radius
    }

    /// The default volume given to each of this fluid's particles.
    pub fn default_particle_volume(&self) -> Real {
        Self::particle_volume(self.particle_radius)
    }

    fn particle_volume(particle_radius: Real) -> Real {
        // The volume of a fluid is computed as the volume of a cuboid of half-width equal to particle_radius.
        // It is multiplied by 0.8 so that there is no pressure when the cuboids are aligned on a grid.
        // This mass computation method is inspired from the SplishSplash project.
        #[cfg(feature = "dim2")]
        let particle_volume = particle_radius * particle_radius * na::convert::<_, Real>(4.0 * 0.8);
        #[cfg(feature = "dim3")]
        let particle_volume =
            particle_radius * particle_radius * particle_radius * na::convert::<_, Real>(8.0 * 0.8);
        particle_volume
    }

    /// Add a set of particles to this fluid.
    ///
    /// If `velocities` is `None` the velocity of each particle will be initialized at zero.
    /// If it is not `None`, then it must be a slice with the same length than `positions`.
    pub fn add_particles(
        &mut self,
        positions: &[Point<Real>],
        velocities: Option<&[Vector<Real>]>,
    ) {
        let nparticles = self.positions.len() + positions.len();
        let particle_volume = self.default_particle_volume();

        self.positions.extend_from_slice(positions);

        if let Some(vels) = velocities {
            assert_eq!(
                positions.len(),
                vels.len(),
                "The provided positions and velocities arrays must have the same length."
            );
            self.velocities.extend_from_slice(vels);
        } else {
            self.velocities.resize(nparticles, Vector::zeros());
        }

        self.accelerations.resize(nparticles, Vector::zeros());
        self.volumes.resize(nparticles, particle_volume);
        self.deleted_particles.resize(nparticles, false);
    }

    /// Sorts all the particles of this fluids according to morton order.
    pub fn z_sort(&mut self) {
        let order = crate::z_order::compute_points_z_order(&self.positions);
        self.positions = crate::z_order::apply_permutation(&order, &self.positions);
        self.velocities = crate::z_order::apply_permutation(&order, &self.velocities);
        self.accelerations = crate::z_order::apply_permutation(&order, &self.accelerations);
        self.volumes = crate::z_order::apply_permutation(&order, self.volumes.as_slice());

        for forces in &mut self.nonpressure_forces {
            forces.apply_permutation(&order);
        }
    }

    /// Apply the given transformation to each particle of this fluid.
    pub fn transform_by(&mut self, t: &Isometry<Real>) {
        self.positions.iter_mut().for_each(|p| *p = t * *p)
    }

    /// The number of particles on this fluid.
    pub fn num_particles(&self) -> usize {
        self.positions.len()
    }

    /// Computes the AABB of this fluid.
    #[cfg(feature = "parry")]
    pub fn compute_aabb(&self, particle_radius: Real) -> parry::bounding_volume::Aabb {
        use parry::bounding_volume::{details::local_point_cloud_aabb, BoundingVolume};
        local_point_cloud_aabb(&self.positions).loosened(particle_radius)
    }

    /// The mass of the `i`-th particle of this fluid.
    pub fn particle_mass(&self, i: usize) -> Real {
        self.volumes[i] * self.density0
    }

    /// The inverse mass of the `i`-th particle of this fluid.
    ///
    /// Returns 0 if the `i`-th particle has a zero mass.
    pub fn particle_inv_mass(&self, i: usize) -> Real {
        if self.volumes[i].is_zero() {
            na::zero::<Real>()
        } else {
            na::one::<Real>() / (self.volumes[i] * self.density0)
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// The unique identifier of a boundary object.
pub struct FluidHandle(ContiguousArenaIndex);
/// The set of all fluid objects.
pub type FluidSet = ContiguousArena<FluidHandle, Fluid>;

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
