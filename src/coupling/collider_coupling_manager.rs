use crate::boundary::{Boundary, BoundaryHandle};
use crate::fluid::Fluid;
use crate::math::{Point, Vector};
use na::RealField;
use ncollide::shape::FeatureId;
use nphysics::math::{Force, ForceType};
use nphysics::object::{Body, BodyHandle, BodySet, ColliderHandle, ColliderSet};
use std::collections::HashMap;

enum CouplingMethod<N: RealField> {
    StaticSampling(Vec<Point<N>>),
    DynamicFeatureSampling(HashMap<FeatureId, Vec<Point<N>>>),
    DynamicContactSampling,
}

pub struct ColliderCoupling<N: RealField, CollHandle: ColliderHandle> {
    coupling_method: CouplingMethod<N>,
    collider: CollHandle,
    boundary: BoundaryHandle,
}

pub struct ColliderCouplingManager<N: RealField, CollHandle: ColliderHandle> {
    entries: Vec<ColliderCoupling<N, CollHandle>>,
}

impl<N: RealField, CollHandle: ColliderHandle> ColliderCouplingManager<N, CollHandle> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn register_static_coupling(
        &mut self,
        boundary: BoundaryHandle,
        collider: CollHandle,
        local_particles: Vec<Point<N>>,
    )
    {
        self.entries.push(ColliderCoupling {
            coupling_method: CouplingMethod::StaticSampling(local_particles),
            collider,
            boundary,
        })
    }

    pub fn update_boundaries<Handle, Colliders>(
        &mut self,
        boundaries: &mut [Boundary<N>],
        fluids: &[Fluid<N>],
        colliders: &Colliders,
    ) where
        Handle: BodyHandle,
        Colliders: ColliderSet<N, Handle, Handle = CollHandle>,
    {
        self.entries.retain(|entry| {
            if let (Some(collider), Some(boundary)) = (
                colliders.get(entry.collider),
                boundaries.get_mut(entry.boundary),
            ) {
                match &entry.coupling_method {
                    CouplingMethod::StaticSampling(points) => {
                        boundary.positions.clear();
                        boundary.velocities.clear();

                        for pt in points {
                            boundary.positions.push(collider.position() * pt);
                            // XXX: actually set the velocity of this point.
                            boundary.velocities.push(Vector::zeros());
                        }
                    }
                    CouplingMethod::DynamicFeatureSampling(_) => unimplemented!(),
                    CouplingMethod::DynamicContactSampling => unimplemented!(),
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
        for entry in &self.entries {
            if let (Some(collider), Some(boundary)) = (
                colliders.get(entry.collider),
                boundaries.get_mut(entry.boundary),
            ) {
                if let Some(body) = bodies.get_mut(collider.body()) {
                    let (linear, angular) = boundary.force();
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
