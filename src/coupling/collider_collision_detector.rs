use crate::boundary::{Boundary, BoundaryHandle};
use crate::coupling::{ContactData, DirectForcing, RigidFluidContactPair};
use crate::fluid::Fluid;
use crate::geometry::{HGrid, HGridEntry};
use crate::math::{Point, Vector};
use na::{RealField, Unit};
use ncollide::bounding_volume::BoundingVolume;
use ncollide::pipeline::CollisionGroups;
use ncollide::query::PointQuery;
use ncollide::shape::FeatureId;
use nphysics::math::{AngularVector, Force, ForceType, Translation, Velocity};
use nphysics::object::{
    Body, BodyHandle, BodyPartHandle, BodySet, ColliderAnchor, ColliderHandle, ColliderSet,
};
use nphysics::world::GeometricalWorld;
use std::collections::{HashMap, HashSet};
use std::process::id;

pub struct ColliderCollisionDetector<N: RealField, Handle: BodyHandle> {
    rigid_body_contacts: Vec<RigidFluidContactPair<N>>,
    rigid_bodies: Vec<BodyPartHandle<Handle>>,
    rigid_bodies_extra_velocities: HashMap<BodyPartHandle<Handle>, Velocity<N>>,
}

impl<N: RealField, Handle: BodyHandle> ColliderCollisionDetector<N, Handle> {
    pub fn new() -> Self {
        Self {
            rigid_body_contacts: Vec::new(),
            rigid_bodies: Vec::new(),
            rigid_bodies_extra_velocities: HashMap::new(),
        }
    }

    // FIXME: this should be done automatically.
    pub fn clear_forces(&mut self) {
        self.rigid_bodies_extra_velocities.clear();
    }

    pub fn detect_contacts<Bodies, Colliders>(
        &mut self,
        dt: N,
        inv_dt: N,
        gravity: &Vector<N>,
        fluids: &[Fluid<N>],
        particle_radius: N,
        hgrid: &HGrid<N, HGridEntry>,
        geometrical_world: &GeometricalWorld<N, Handle, Colliders::Handle>,
        bodies: &Bodies,
        colliders: &Colliders,
    ) where
        Bodies: BodySet<N, Handle = Handle>,
        Colliders: ColliderSet<N, Handle>,
    {
        self.rigid_body_contacts.clear();
        self.rigid_bodies.clear();

        let mut colliders_to_check = HashSet::new();
        // FIXME: avoid the hash-map?
        let mut body_part_to_contacts = HashMap::new();
        let mut num_contacts = 0;
        let friction_coeff: N = na::convert(0.1);

        for fluid in fluids {
            let all_groups = CollisionGroups::new();
            let fluid_aabb = fluid.compute_aabb(particle_radius);
            let interferences =
                geometrical_world.interferences_with_aabb(colliders, &fluid_aabb, &all_groups);

            for (handle, _) in interferences {
                let _ = colliders_to_check.insert(handle);
            }
        }

        for collider_handle in colliders_to_check {
            if let Some(collider) = colliders.get(collider_handle) {
                let bf = geometrical_world.broad_phase();
                let aabb = collider
                    .proxy_handle()
                    .and_then(|h| bf.proxy(h))
                    .map(|p| p.0)
                    .unwrap()
                    .loosened(particle_radius);

                for particle in hgrid
                    .cells_intersecting_aabb(aabb.mins(), aabb.maxs())
                    .flat_map(|e| e.1)
                {
                    match particle {
                        HGridEntry::FluidParticle(fluid_id, particle_id) => {
                            let fluid = &fluids[*fluid_id];
                            let particle_pos = &fluid.positions[*particle_id];

                            if aabb.contains_local_point(particle_pos) {
                                match collider.anchor() {
                                    ColliderAnchor::OnBodyPart { body_part, .. } => {
                                        let body = bodies.get(body_part.0).unwrap();
                                        let part = body.part(body_part.1).unwrap();
                                        let com = part.center_of_mass();
                                        let body_gravity = if body.status_dependent_ndofs() != 0 {
                                            *gravity
                                        } else {
                                            Vector::zeros()
                                        };

                                        let collider_pos = if let Some(vel) =
                                            self.rigid_bodies_extra_velocities.get(body_part)
                                        {
                                            let total_vel = *vel
                                                + Velocity::from_vectors(
                                                    body_gravity * dt,
                                                    AngularVector::zeros(),
                                                );
                                            let shift = Translation::from(com.coords);
                                            let disp =
                                                shift * total_vel.integrate(dt) * shift.inverse();
                                            disp * collider.position()
                                        } else {
                                            *collider.position()
                                        };

                                        let proj = collider.shape().project_point(
                                            &collider_pos,
                                            particle_pos,
                                            false,
                                        );

                                        let dpt = particle_pos - proj.point;

                                        if let Some((mut normal, mut depth)) =
                                            Unit::try_new_and_get(dpt, N::default_epsilon())
                                        {
                                            if proj.is_inside {
                                                normal = -normal;
                                            } else {
                                                depth = -depth;
                                            }

                                            depth += particle_radius;

                                            let r = proj.point - com;

                                            let cp_vel = part.velocity().shift(&r).linear
                                                + body_gravity * dt;
                                            let dvel = fluid.velocities[*particle_id] - cp_vel;
                                            let normal_dvel = *normal * dvel.dot(&normal);
                                            let tangent_dvel = dvel - normal_dvel;

                                            // Correction for predictive contact.
                                            let predictive_dvel = if depth < N::zero() {
                                                *normal * (-depth * inv_dt)
                                            } else {
                                                Vector::zeros()
                                            };

                                            let contact = ContactData {
                                                fluid: *fluid_id,
                                                particle: *particle_id,
                                                particle_mass: fluid.particle_mass(*particle_id),
                                                normal,
                                                depth,
                                                point: proj.point,
                                                r,
                                                vel: normal_dvel
                                                    + predictive_dvel
                                                    + (tangent_dvel * friction_coeff),
                                                result_particle_force: Vector::zeros(),
                                                result_particle_shift: Vector::zeros(),
                                            };

                                            let id = body_part_to_contacts
                                                .entry(*body_part)
                                                .or_insert_with(|| {
                                                    let inertia = part.inertia();
                                                    let pair = RigidFluidContactPair {
                                                        one_way_coupling: body
                                                            .status_dependent_ndofs()
                                                            == 0,
                                                        rigid_m: inertia.linear,
                                                        rigid_i: *inertia.angular_matrix(),
                                                        contacts: Vec::new(),
                                                        result_linear_acc: Vector::zeros(),
                                                        result_angular_acc: AngularVector::zeros(),
                                                    };

                                                    let id = self.rigid_body_contacts.len();
                                                    self.rigid_body_contacts.push(pair);
                                                    self.rigid_bodies.push(*body_part);
                                                    id
                                                });

                                            self.rigid_body_contacts[*id].contacts.push(contact);
                                            num_contacts += 1;
                                        }
                                    }
                                    _ => {
                                        // Not yet implemented.
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
    }

    //    pub fn solve_positions<Bodies>(
    //        &mut self,
    //        dt: N,
    //        inv_dt: N,
    //        fluids: &mut [Fluid<N>],
    //        bodies: &mut Bodies,
    //    ) where
    //        Bodies: BodySet<N, Handle = Handle>,
    //    {
    //        DirectForcing::solve_positions(dt, inv_dt, &mut self.rigid_body_contacts);
    //
    //        for pair in &self.rigid_body_contacts {
    //            for contact in &pair.contacts {
    //                let fluid = &mut fluids[contact.fluid];
    //                fluid.positions[contact.particle] += contact.result_particle_shift;
    //            }
    //        }
    //    }
}
