use crate::coupling::CouplingManager;
use crate::geometry::{DeterministicState, HGrid, HGridEntry};
use crate::object::{BoundaryHandle, BoundarySet, Fluid};
use crate::solver::{DFSPHSolver, DfsphParameters, PressureSolver};
use crate::TimestepManager;
use crate::{math, LiquidWorld};
use approx::AbsDiffEq;
use na::Unit;
use rapier::dynamics::RigidBodySet;
use rapier::geometry::{ColliderHandle, ColliderSet};
use rapier::math::Vector as RapierVector;
use rapier::parry::bounding_volume::BoundingVolume;
use rapier::parry::shape::FeatureId;
use std::collections::HashMap;
use std::sync::RwLock;

#[cfg(feature = "dim2")]
fn salva_to_rapier_vector(v: math::Vector<math::Real>) -> RapierVector {
    RapierVector::new(v.x, v.y)
}

#[cfg(feature = "dim3")]
fn salva_to_rapier_vector(v: math::Vector<math::Real>) -> RapierVector {
    RapierVector::new(v.x, v.y, v.z)
}

#[cfg(feature = "dim2")]
fn rapier_to_salva_vector(v: RapierVector) -> math::Vector<math::Real> {
    math::Vector::new(v.x, v.y)
}

#[cfg(feature = "dim3")]
fn rapier_to_salva_vector(v: RapierVector) -> math::Vector<math::Real> {
    math::Vector::new(v.x, v.y, v.z)
}

/// Pipeline for particle-based fluid simulation.
pub struct FluidsPipeline {
    // FIXME: keep these public?
    /// The liquid world responsible for the actual fluid simulation.
    pub liquid_world: LiquidWorld,
    /// Set of couplings between fluids and rigid-bodies.
    pub coupling: ColliderCouplingSet,
}

impl FluidsPipeline {
    /// Initialize a new pipeline for fluids simulation.
    ///
    /// # Parameters
    ///
    /// - `particle_radius`: the radius of every particle for the fluid simulation.
    /// - `smoothing_factor`: the smoothing factor used to compute the SPH kernel radius.
    ///    The kernel radius will be computed as `particle_radius * smoothing_factor * 2.0.
    pub fn new(particle_radius: math::Real, smoothing_factor: math::Real) -> Self {
        Self::new_with_boundary_coef(particle_radius, smoothing_factor, na::one::<math::Real>())
    }

    /// Initialize a new pipeline with custom DFSPH solver parameters.
    pub fn new_with_dfsph_parameters(
        particle_radius: math::Real,
        smoothing_factor: math::Real,
        dfsph_parameters: DfsphParameters,
    ) -> Self {
        let dfsph: DFSPHSolver = DFSPHSolver::with_parameters(dfsph_parameters);
        Self::new_with_solver(dfsph, particle_radius, smoothing_factor)
    }

    /// Initialize a new pipeline for fluids simulation with a custom boundary force coefficient.
    ///
    /// # Parameters
    ///
    /// - `particle_radius`: the radius of every particle for the fluid simulation.
    /// - `smoothing_factor`: the smoothing factor used to compute the SPH kernel radius.
    ///    The kernel radius will be computed as `particle_radius * smoothing_factor * 2.0.
    /// - `boundary_force_coefficient`: coefficient applied when transmitting forces from fluids to boundaries.
    ///    Use 1.0 for full force, 0.5 for half force, etc.
    pub fn new_with_boundary_coef(
        particle_radius: math::Real,
        smoothing_factor: math::Real,
        boundary_force_coefficient: math::Real,
    ) -> Self {
        let dfsph: DFSPHSolver = DFSPHSolver::new();
        Self::new_with_solver_and_boundary_coef(
            dfsph,
            particle_radius,
            smoothing_factor,
            boundary_force_coefficient,
        )
    }

    /// Initialize a new pipeline with a custom pressure solver.
    pub fn new_with_solver(
        solver: impl PressureSolver + Send + Sync + 'static,
        particle_radius: math::Real,
        smoothing_factor: math::Real,
    ) -> Self {
        Self::new_with_solver_and_boundary_coef(
            solver,
            particle_radius,
            smoothing_factor,
            na::one::<math::Real>(),
        )
    }

    /// Initialize a new pipeline with a custom pressure solver and boundary force coefficient.
    pub fn new_with_solver_and_boundary_coef(
        solver: impl PressureSolver + Send + Sync + 'static,
        particle_radius: math::Real,
        smoothing_factor: math::Real,
        boundary_force_coefficient: math::Real,
    ) -> Self {
        Self {
            liquid_world: LiquidWorld::new(
                solver,
                particle_radius,
                smoothing_factor,
                boundary_force_coefficient,
            ),
            coupling: ColliderCouplingSet::new(),
        }
    }

    /// Current DFSPH tuning parameters, if this pipeline uses DFSPH.
    pub fn dfsph_parameters(&self) -> Option<DfsphParameters> {
        self.liquid_world.dfsph_parameters()
    }

    /// Set DFSPH tuning parameters if this pipeline uses DFSPH.
    pub fn set_dfsph_parameters(&mut self, parameters: DfsphParameters) -> bool {
        self.liquid_world.set_dfsph_parameters(parameters)
    }

    /// Advances the fluid simulation by `dt` seconds.
    ///
    /// All the fluid particles will be affected by an acceleration equal to `gravity`.
    /// This `step` function may apply forces to some rigid-bodies that interact with fluids.
    /// However, it will not integrate these forces. Use the `PhysicsPipeline` for this integration.
    pub fn step(
        &mut self,
        gravity: &RapierVector,
        dt: math::Real,
        colliders: &ColliderSet,
        bodies: &mut RigidBodySet,
    ) {
        let gravity = rapier_to_salva_vector(*gravity);
        self.liquid_world.step_with_coupling(
            dt,
            &gravity,
            &mut self.coupling.as_manager_mut(colliders, bodies),
        )
    }
}

/// The way a collider is coupled to a boundary object.
pub enum ColliderSampling {
    /// The collider shape is approximated with the given sample points in local-space.
    ///
    /// It is recommended that those points are separated by a distance smaller or equal to twice
    /// the particle radius used to initialize the LiquidWorld.
    StaticSampling(Vec<math::Vector<math::Real>>),
    /// The collider shape is approximated by a dynamic set of points automatically computed based on contacts with fluid particles.
    DynamicContactSampling,
}

pub struct ColliderCouplingEntry {
    pub sampling_method: ColliderSampling,
    pub boundary: BoundaryHandle,
    features: Vec<FeatureId>,
}

/// Structure managing all the coupling between colliders from rapier with boundaries and fluids from salva.
pub struct ColliderCouplingSet {
    /// The hashmap containing mappings from ColliderHandle to ColliderCouplingEntry
    pub entries: HashMap<ColliderHandle, ColliderCouplingEntry, DeterministicState>,
}

impl ColliderCouplingSet {
    /// Create a new collider coupling manager.
    pub fn new() -> Self {
        Self {
            entries: HashMap::with_hasher(DeterministicState),
        }
    }

    /// Register a coupling between a boundary and a collider.
    /// There can be only up to one coupling between a collider and a boundary object. If a coupling
    /// already exists for this collider when calling this function, the handle of the previously coupled
    /// boundary is returned.
    pub fn register_coupling(
        &mut self,
        boundary: BoundaryHandle,
        collider: ColliderHandle,
        sampling_method: ColliderSampling,
    ) -> Option<BoundaryHandle> {
        let old = self.entries.insert(
            collider,
            ColliderCouplingEntry {
                sampling_method,
                boundary,
                features: Vec::new(),
            },
        );

        old.map(|e| e.boundary)
    }

    /// Unregister a coupling between a boundary and a collider.
    /// Note that this does not remove the boundary itself from the liquid world.
    /// Returns the handle of the boundary this collider was coupled with.
    pub fn unregister_coupling(&mut self, collider: ColliderHandle) -> Option<BoundaryHandle> {
        let deleted = self.entries.remove(&collider);
        deleted.map(|e| e.boundary)
    }

    /// Use this collider coupling set as a coupling manager.
    pub fn as_manager_mut<'a>(
        &'a mut self,
        colliders: &'a ColliderSet,
        bodies: &'a mut RigidBodySet,
    ) -> ColliderCouplingManager<'a> {
        ColliderCouplingManager {
            coupling: self,
            colliders,
            bodies,
        }
    }
}

/// A manager for coupling colliders from rapier2d/rapier3D with the boundary
/// objects from salva.
pub struct ColliderCouplingManager<'a> {
    coupling: &'a mut ColliderCouplingSet,
    colliders: &'a ColliderSet,
    bodies: &'a mut RigidBodySet,
}

impl<'a> CouplingManager for ColliderCouplingManager<'a> {
    fn update_boundaries(
        &mut self,
        timestep: &TimestepManager,
        h: math::Real,
        particle_radius: math::Real,
        hgrid: &HGrid<HGridEntry>,
        fluids: &mut [Fluid],
        boundaries: &mut BoundarySet,
    ) {
        let bodies = &self.bodies;
        for (collider, coupling) in &mut self.coupling.entries {
            if let (Some(collider), Some(boundary)) = (
                self.colliders.get(*collider),
                boundaries.get_mut(coupling.boundary),
            ) {
                // Update the boundary's ability to receive forces.
                let body = collider.parent().and_then(|p| bodies.get(p));
                if let Some(body) = body {
                    if !body.is_dynamic() {
                        boundary.forces = None;
                    } else {
                        boundary.forces = Some(RwLock::new(Vec::new()));
                        boundary.clear_forces(true);
                    }
                }

                // Update positions and velocities.
                boundary.positions.clear();
                boundary.velocities.clear();
                boundary.volumes.clear();
                coupling.features.clear();

                match &coupling.sampling_method {
                    ColliderSampling::StaticSampling(points) => {
                        for pt in points {
                            let rapier_pt = salva_to_rapier_vector(*pt);
                            let world_pt = rapier_to_salva_vector(collider.position() * rapier_pt);
                            boundary.positions.push(world_pt);
                            let velocity = body.map(|b| b.velocity_at_point(rapier_pt));

                            boundary.velocities.push(rapier_to_salva_vector(
                                velocity.unwrap_or(RapierVector::ZERO),
                            ));
                        }

                        boundary
                            .volumes
                            .resize(points.len(), na::zero::<math::Real>());
                    }
                    ColliderSampling::DynamicContactSampling => {
                        let prediction = h * na::convert::<_, math::Real>(0.5);
                        let margin = particle_radius * na::convert::<_, math::Real>(0.1);
                        let collider_pos = collider.position();
                        let aabb = collider
                            .shape()
                            .compute_aabb(&collider_pos)
                            .loosened(h + prediction);
                        let mins = rapier_to_salva_vector(aabb.mins);
                        let maxs = rapier_to_salva_vector(aabb.maxs);

                        for particle in hgrid
                            .cells_intersecting_aabb(&mins, &maxs)
                            .flat_map(|e| e.1)
                        {
                            match particle {
                                HGridEntry::FluidParticle(fluid_id, particle_id) => {
                                    let fluid = &mut fluids[*fluid_id];

                                    // Check interaction groups between this fluid and the boundary.
                                    // Use the fluid's interaction groups explicitly to mirror checks
                                    // performed elsewhere in the codebase.
                                    let fluid_groups = fluid.interaction_groups;
                                    let boundary_groups = boundary.interaction_groups;

                                    if !fluid_groups.test(boundary_groups) {
                                        continue;
                                    }

                                    let particle_pos = fluid.positions[*particle_id]
                                        + fluid.velocities[*particle_id] * timestep.dt();
                                    let rapier_particle_pos = salva_to_rapier_vector(particle_pos);

                                    if aabb.contains_local_point(rapier_particle_pos) {
                                        let (proj, feature) =
                                            collider.shape().project_point_and_get_feature(
                                                &collider_pos,
                                                rapier_particle_pos,
                                            );

                                        let dpt = particle_pos - rapier_to_salva_vector(proj.point);

                                        if let Some((normal, depth)) = Unit::try_new_and_get(
                                            dpt,
                                            math::Real::default_epsilon(),
                                        ) {
                                            if proj.is_inside {
                                                fluid.positions[*particle_id] -=
                                                    *normal * (depth + margin);

                                                let vel_err =
                                                    normal.dot(&fluid.velocities[*particle_id]);

                                                if vel_err > na::zero::<math::Real>() {
                                                    fluid.velocities[*particle_id] -=
                                                        *normal * vel_err;
                                                }
                                            } else if depth > h + prediction {
                                                continue;
                                            }
                                        }

                                        let velocity =
                                            body.map(|b| b.velocity_at_point(proj.point));

                                        boundary.velocities.push(rapier_to_salva_vector(
                                            velocity.unwrap_or(RapierVector::ZERO),
                                        ));
                                        boundary.positions.push(rapier_to_salva_vector(proj.point));
                                        boundary.volumes.push(na::zero::<math::Real>());
                                        coupling.features.push(feature);
                                    }
                                }
                                HGridEntry::BoundaryParticle(..) => {
                                    // Not yet implemented.
                                }
                            }
                        }
                    }
                }

                boundary.clear_forces(true);
            }
        }
    }

    fn transmit_forces(
        &mut self,
        timestep: &TimestepManager,
        boundaries: &BoundarySet,
        force_coefficient: math::Real,
    ) {
        for (collider, coupling) in &self.coupling.entries {
            if let (Some(collider), Some(boundary)) = (
                self.colliders.get(*collider),
                boundaries.get(coupling.boundary),
            ) {
                if boundary.positions.is_empty() {
                    continue;
                }

                if let Some(forces) = &boundary.forces {
                    let forces = forces.read().unwrap();
                    if let Some(parent) = collider.parent() {
                        if let Some(body) = self.bodies.get_mut(parent) {
                            for (pos, force) in
                                boundary.positions.iter().zip(forces.iter().cloned())
                            {
                                body.apply_impulse_at_point(
                                    salva_to_rapier_vector(
                                        force * timestep.dt() * force_coefficient,
                                    ),
                                    salva_to_rapier_vector(*pos),
                                    true,
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}
