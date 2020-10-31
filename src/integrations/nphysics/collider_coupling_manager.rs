use crate::coupling::CouplingManager;
use crate::geometry::{HGrid, HGridEntry};
use crate::math::{Point, Real, Vector};
use crate::object::Fluid;
use crate::object::{BoundaryHandle, BoundarySet};
use crate::TimestepManager;
use na::{RealField, Unit};
use ncollide::bounding_volume::BoundingVolume;
use ncollide::query::PointQuery;
use ncollide::shape::FeatureId;
use nphysics::math::ForceType;
use nphysics::object::{BodySet, ColliderAnchor, ColliderHandle, ColliderSet};
use std::collections::HashMap;
use std::sync::RwLock;

/// The way a collider is coupled to a boundary object.
pub enum CouplingMethod {
    /// The collider shape is approximated with the given sample points in local-space.
    ///
    /// It is recommended that those points are separated by a distance smaller or equal to twice
    /// the particle radius used to initialize the LiquidWorld.
    StaticSampling(Vec<Point<Real>>),
    /// The collider shape is approximated by a dynamic set of points automatically computed based on contacts with fluid particles.
    DynamicContactSampling,
}

struct ColliderCouplingEntry {
    coupling_method: CouplingMethod,
    boundary: BoundaryHandle,
    features: Vec<FeatureId>,
}

/// Structure managing all the coupling between colliders from nphysics with boundaries and fluids from salva.
pub struct ColliderCouplingSet<CollHandle: ColliderHandle> {
    entries: HashMap<CollHandle, ColliderCouplingEntry>,
}

impl<CollHandle: ColliderHandle> ColliderCouplingSet<CollHandle> {
    /// Create a new collider coupling manager.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Register a coupling between a boundary and a collider.
    /// There can be only up to one coupling between a collider and a boundary object. If a coupling
    /// already exists for this collider when calling this function, the handle of the previously coupled
    /// boundary is returned.
    pub fn register_coupling(
        &mut self,
        boundary: BoundaryHandle,
        collider: CollHandle,
        coupling_method: CouplingMethod,
    ) -> Option<BoundaryHandle> {
        let old = self.entries.insert(
            collider,
            ColliderCouplingEntry {
                coupling_method,
                boundary,
                features: Vec::new(),
            },
        );

        old.map(|e| e.boundary)
    }

    /// Unregister a coupling between a boundary and a collider.
    /// Note that this does not remove the boundary itself from the liquid world.
    /// Returns the handle of the boundary this collider was coupled with.
    pub fn unregister_coupling(&mut self, collider: CollHandle) -> Option<BoundaryHandle> {
        let deleted = self.entries.remove(&collider);
        deleted.map(|e| e.boundary)
    }

    /// Use this collider coupling set as a coupling manager.
    pub fn as_manager_mut<'a, Colliders, Bodies>(
        &'a mut self,
        colliders: &'a Colliders,
        bodies: &'a mut Bodies,
    ) -> ColliderCouplingManager<Colliders, Bodies>
    where
        Colliders: ColliderSet<Bodies::Handle, Handle = CollHandle>,
        Bodies: BodySet<Real>,
    {
        ColliderCouplingManager {
            coupling: self,
            colliders,
            bodies,
        }
    }
}

/// A manager for coupling colliders from nphysics2d/nphysics3D with the boundary
/// objects from salva.
pub struct ColliderCouplingManager<'a, Colliders, Bodies>
where
    Colliders: ColliderSet<Real, Bodies::Handle>,
    Bodies: BodySet<Real>,
{
    coupling: &'a mut ColliderCouplingSet<Colliders::Handle>,
    colliders: &'a Colliders,
    bodies: &'a mut Bodies,
}

impl<'a, Colliders, Bodies> CouplingManager for ColliderCouplingManager<'a, Colliders, Bodies>
where
    Colliders: ColliderSet<Real, Bodies::Handle>,
    Bodies: BodySet<Real>,
{
    fn update_boundaries(
        &mut self,
        timestep: &TimestepManager,
        h: Real,
        particle_radius: Real,
        hgrid: &HGrid<HGridEntry>,
        fluids: &mut [Fluid],
        boundaries: &mut BoundarySet,
    ) {
        for (collider, coupling) in &mut self.coupling.entries {
            if let (Some(collider), Some(boundary)) = (
                self.colliders.get(*collider),
                boundaries.get_mut(coupling.boundary),
            ) {
                // Update the boundary's ability to receive forces.
                let body = self.bodies.get(collider.body());
                if let Some(body) = body {
                    if body.status_dependent_ndofs() == 0 {
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

                match &coupling.coupling_method {
                    CouplingMethod::StaticSampling(points) => {
                        for pt in points {
                            boundary.positions.push(collider.position() * pt);
                            // FIXME: how do we get the point-velocity of deformable bodies correctly?
                            let velocity = body.map(|b| {
                                if let ColliderAnchor::OnBodyPart { body_part, .. } =
                                    collider.anchor()
                                {
                                    b.velocity_at_point(body_part.1, pt).linear
                                } else {
                                    Vector::zeros()
                                }
                            });

                            boundary
                                .velocities
                                .push(velocity.unwrap_or(Vector::zeros()));
                        }

                        boundary.volumes.resize(points.len(), na::zero::<Real>());
                    }
                    CouplingMethod::DynamicContactSampling => {
                        let prediction = h * na::convert::<_, Real>(0.5);
                        let margin = particle_radius * na::convert::<_, Real>(0.1);
                        let collider_pos = collider.position();
                        let aabb = collider
                            .shape()
                            .aabb(&collider_pos)
                            .loosened(h + prediction);

                        for particle in hgrid
                            .cells_intersecting_aabb(aabb.mins(), aabb.maxs())
                            .flat_map(|e| e.1)
                        {
                            match particle {
                                HGridEntry::FluidParticle(fluid_id, particle_id) => {
                                    let fluid = &mut fluids[*fluid_id];
                                    let particle_pos = fluid.positions[*particle_id]
                                        + fluid.velocities[*particle_id] * timestep.dt();

                                    if aabb.contains_local_point(&particle_pos) {
                                        let (proj, feature) =
                                            collider.shape().project_point_with_feature(
                                                &collider_pos,
                                                &particle_pos,
                                            );

                                        let dpt = particle_pos - proj.point;

                                        if let Some((normal, depth)) =
                                            Unit::try_new_and_get(dpt, Real::default_epsilon())
                                        {
                                            if proj.is_inside {
                                                fluid.positions[*particle_id] -=
                                                    *normal * (depth + margin);

                                                let vel_err =
                                                    normal.dot(&fluid.velocities[*particle_id]);

                                                if vel_err > na::zero::<Real>() {
                                                    fluid.velocities[*particle_id] -=
                                                        *normal * vel_err;
                                                }
                                            } else if depth > h + prediction {
                                                continue;
                                            }
                                        }

                                        let velocity = body.map(|b| {
                                            if let ColliderAnchor::OnBodyPart {
                                                body_part, ..
                                            } = collider.anchor()
                                            {
                                                b.velocity_at_point(body_part.1, &proj.point).linear
                                            } else {
                                                Vector::zeros()
                                            }
                                        });

                                        boundary
                                            .velocities
                                            .push(velocity.unwrap_or(Vector::zeros()));
                                        boundary.positions.push(proj.point);
                                        boundary.volumes.push(na::zero::<Real>());
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

    fn transmit_forces(&mut self, boundaries: &BoundarySet) {
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
                    match collider.anchor() {
                        ColliderAnchor::OnBodyPart { body_part, .. } => {
                            if let Some(body) = self.bodies.get_mut(body_part.0) {
                                for (pos, force) in
                                    boundary.positions.iter().zip(forces.iter().cloned())
                                {
                                    // FIXME: how do we deal with large density ratio?
                                    // Is it only an issue with PBF?
                                    // The following commented code was an attempt to limit the force applied
                                    // to the bodies in order to avoid large forces.
                                    //
                                    //                                let ratio = na::convert::<_, Real>(3.0)
                                    //                                    * body.part(body_part.1).unwrap().inertia().mass();
                                    //
                                    //                                if ratio < na::convert::<_, Real>(1.0) {
                                    //                                    force *= ratio;
                                    //                                }

                                    body.apply_force_at_point(
                                        body_part.1,
                                        &force,
                                        pos,
                                        ForceType::Force,
                                        true,
                                    )
                                }
                            }
                        }
                        ColliderAnchor::OnDeformableBody { body, body_parts } => {
                            if let Some(body) = self.bodies.get_mut(*body) {
                                for (feature, pos, force) in itertools::multizip((
                                    coupling.features.iter(),
                                    boundary.positions.iter(),
                                    forces.iter(),
                                )) {
                                    let subshape_id =
                                        collider.shape().subshape_containing_feature(*feature);
                                    let part_id = if let Some(body_parts) = body_parts {
                                        body_parts[subshape_id]
                                    } else {
                                        subshape_id
                                    };

                                    body.apply_force_at_point(
                                        part_id,
                                        &force,
                                        pos,
                                        ForceType::Force,
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
}
