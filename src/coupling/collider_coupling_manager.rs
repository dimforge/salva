use crate::boundary::{Boundary, BoundaryHandle};
use crate::fluid::Fluid;
use crate::geometry::{HGrid, HGridEntry};
use crate::math::{Point, Vector};
use na::{RealField, Unit};
use ncollide::bounding_volume::BoundingVolume;
use ncollide::query::PointQuery;
use ncollide::shape::FeatureId;
use nphysics::math::{Force, ForceType};
use nphysics::object::{Body, BodyHandle, BodySet, ColliderHandle, ColliderSet};
use std::collections::HashMap;

pub enum CouplingMethod<N: RealField> {
    StaticSampling(Vec<Point<N>>),
    DynamicFeatureSampling(HashMap<FeatureId, Vec<Point<N>>>),
    DynamicContactSampling,
}

pub struct ColliderCoupling<N: RealField> {
    coupling_method: CouplingMethod<N>,
    boundary: BoundaryHandle,
}

pub struct ColliderCouplingManager<N: RealField, CollHandle: ColliderHandle> {
    entries: HashMap<CollHandle, ColliderCoupling<N>>,
}

impl<N: RealField, CollHandle: ColliderHandle> ColliderCouplingManager<N, CollHandle> {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn register_coupling(
        &mut self,
        boundary: BoundaryHandle,
        collider: CollHandle,
        coupling_method: CouplingMethod<N>,
    )
    {
        let _ = self.entries.insert(
            collider,
            ColliderCoupling {
                coupling_method,
                boundary,
            },
        );
    }

    pub fn update_boundaries<Handle, Colliders>(
        &mut self,
        h: N,
        colliders: &Colliders,
        boundaries: &mut [Boundary<N>],
        fluids: &[Fluid<N>],
        fluids_delta_pos: &mut [Vec<Vector<N>>],
        hgrid: &mut HGrid<N, HGridEntry>,
    ) where
        Handle: BodyHandle,
        Colliders: ColliderSet<N, Handle, Handle = CollHandle>,
    {
        self.entries.retain(|collider, coupling| {
            if let (Some(collider), Some(boundary)) = (
                colliders.get(*collider),
                boundaries.get_mut(coupling.boundary),
            ) {
                boundary.positions.clear();
                boundary.velocities.clear();

                match &coupling.coupling_method {
                    CouplingMethod::StaticSampling(points) => {
                        for pt in points {
                            boundary.positions.push(collider.position() * pt);
                            // XXX: actually set the velocity of this point.
                            boundary.velocities.push(Vector::zeros());
                        }
                    }
                    CouplingMethod::DynamicFeatureSampling(_) => unimplemented!(),
                    CouplingMethod::DynamicContactSampling => {
                        let collider_pos = collider.position();
                        let aabb = collider.shape().aabb(&collider_pos).loosened(h);

                        for particle in hgrid
                            .cells_intersecting_aabb(aabb.mins(), aabb.maxs())
                            .flat_map(|e| e.1)
                        {
                            match particle {
                                HGridEntry::FluidParticle(fluid_id, particle_id) => {
                                    let fluid = &fluids[*fluid_id];
                                    let particle_delta =
                                        &mut fluids_delta_pos[*fluid_id][*particle_id];
                                    let mut particle_pos =
                                        fluid.positions[*particle_id] + *particle_delta;

                                    if aabb.contains_local_point(&particle_pos) {
                                        let proj = collider.shape().project_point(
                                            &collider_pos,
                                            &particle_pos,
                                            false,
                                        );

                                        let dpt = particle_pos - proj.point;
                                        boundary.positions.push(proj.point);

                                        if let Some((mut normal, mut depth)) =
                                            Unit::try_new_and_get(dpt, N::default_epsilon())
                                        {
                                            if proj.is_inside {
                                                *particle_delta -=
                                                    *normal * (depth + na::convert(0.0001));
                                            }
                                        }
                                    }
                                }
                                HGridEntry::BoundaryParticle(..) => {
                                    // Not yet implemented.
                                }
                            }
                        }
                    }
                }

                true
            } else {
                false
            }
        })
    }

    pub fn transmit_forces<Bodies, Colliders>(
        &self,
        boundaries: &mut [Boundary<N>],
        fluids: &[Fluid<N>],
        bodies: &mut Bodies,
        colliders: &Colliders,
    ) where
        Colliders: ColliderSet<N, Bodies::Handle, Handle = CollHandle>,
        Bodies: BodySet<N>,
    {
        for (collider, coupling) in &self.entries {
            if let (Some(collider), Some(boundary)) = (
                colliders.get(*collider),
                boundaries.get_mut(coupling.boundary),
            ) {
                if boundary.positions.is_empty() {
                    continue;
                }

                if let Some(body) = bodies.get_mut(collider.body()) {
                    let (mut linear, mut angular) = boundary.force();
                    let ratio = na::convert::<_, N>(3.0) * body.part(0).unwrap().inertia().mass();

                    if ratio < na::convert(1.0) {
                        linear *= ratio;
                        angular *= ratio;
                    }

                    let boundary_ref_point = boundary.positions[0];

                    // FIXME: the part_id should not be zero.
                    body.apply_force_at_point(
                        0,
                        &linear,
                        &boundary_ref_point,
                        ForceType::Force,
                        true,
                    );

                    let torque = Force::torque_from_vector(angular);
                    body.apply_force(0, &torque, ForceType::Force, true);
                }

                boundary.clear_forces();
            }
        }
    }
}
