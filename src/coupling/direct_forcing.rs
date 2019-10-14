use crate::math::{
    AngularDim, AngularInertia, AngularVector, Dim, Point, SpatialDim, SpatialMatrix, Vector, DIM,
};
use na::{MatrixMN, MatrixN, RealField, Unit, VectorN};
use std::marker::PhantomData;

pub struct ContactData<N: RealField> {
    pub fluid: usize,
    pub particle: usize,
    pub particle_mass: N,
    pub normal: Unit<Vector<N>>,
    pub point: Point<N>,
    pub r: Vector<N>,
    pub depth: N,
    pub vel: Vector<N>,
    pub result_particle_force: Vector<N>,
    pub result_particle_shift: Vector<N>,
}

impl<N: RealField> ContactData<N> {
    fn vel_active(&self) -> bool {
        self.vel.dot(&self.normal) < N::zero()
    }
}

pub struct RigidFluidContactPair<N: RealField> {
    pub one_way_coupling: bool,
    pub rigid_m: N,
    pub rigid_i: AngularInertia<N>,
    pub contacts: Vec<ContactData<N>>,
    pub result_linear_acc: Vector<N>,
    pub result_angular_acc: AngularVector<N>,
}

pub struct DirectForcing {}

impl DirectForcing {
    pub fn solve_velocities<N: RealField>(
        dt: N,
        inv_dt: N,
        contacts: &mut [RigidFluidContactPair<N>],
    )
    {
        for pair in contacts {
            pair.result_linear_acc.fill(N::zero());
            pair.result_angular_acc.fill(N::zero());

            if pair.one_way_coupling {
                // Non need to solve a system for one-way coupling.
                for contact in &mut pair.contacts {
                    // FIXME: this filtering should be done when the contacts are generated.
                    if contact.vel_active() {
                        contact.result_particle_force =
                            contact.vel * (-contact.particle_mass * inv_dt);
                    } else {
                        contact.result_particle_force.fill(N::zero());
                    }
                }
            } else {
                let mut a11 = pair.rigid_m;
                let mut a21 = MatrixMN::<N, AngularDim, Dim>::zeros();
                let mut a22 = pair.rigid_i;
                let mut rhs = VectorN::<N, SpatialDim>::zeros();

                for contact in &mut pair.contacts {
                    contact.result_particle_force.fill(N::zero());

                    if !contact.vel_active() {
                        // FIXME: this filtering should be done when the contacts are generated.
                        continue;
                    }

                    let rmat = crate::math::gcross_matrix(&contact.r);
                    let weighted_rmat = rmat * contact.particle_mass;
                    a11 += contact.particle_mass;
                    a21 += weighted_rmat;
                    a22 += weighted_rmat * rmat.transpose();

                    rhs.fixed_rows_mut::<Dim>(0).axpy(
                        contact.particle_mass * inv_dt,
                        &contact.vel,
                        N::one(),
                    );

                    rhs.fixed_rows_mut::<AngularDim>(DIM).gemv(
                        inv_dt,
                        &weighted_rmat,
                        &contact.vel,
                        N::one(),
                    );
                }

                let mut a = SpatialMatrix::zeros();
                a.fixed_slice_mut::<Dim, Dim>(0, 0).fill_diagonal(a11);
                a.fixed_slice_mut::<AngularDim, Dim>(DIM, 0).copy_from(&a21);
                a.fixed_slice_mut::<Dim, AngularDim>(0, DIM)
                    .tr_copy_from(&a21);
                a.fixed_slice_mut::<AngularDim, AngularDim>(DIM, DIM)
                    .copy_from(&a22);

                if let Some(inv_a) = a.cholesky() {
                    // Compute the resultant force.
                    inv_a.solve_mut(&mut rhs);

                    let lin_accel = rhs.fixed_rows::<Dim>(0).into_owned();
                    let ang_accel = rhs.fixed_rows::<AngularDim>(DIM).into_owned();
                    pair.result_linear_acc = lin_accel;
                    pair.result_angular_acc = ang_accel;

                    // Compute the corresponding forces on the particles.
                    for contact in &mut pair.contacts {
                        if !contact.vel_active() {
                            // FIXME: this filtering should be done when the contacts are generated.
                            continue;
                        }

                        let rmat_tr = crate::math::gcross_matrix(&contact.r).transpose();
                        contact.result_particle_force =
                            (-contact.vel * inv_dt + lin_accel + rmat_tr * ang_accel)
                                * contact.particle_mass;
                    }
                }
            }
        }
    }

    pub fn solve_positions<N: RealField>(
        dt: N,
        inv_dt: N,
        contacts: &mut [RigidFluidContactPair<N>],
    )
    {
        for pair in contacts {
            for contact in &mut pair.contacts {
                // FIXME: this filtering should be done when the contacts are generated.
                if contact.depth > N::zero() {
                    contact.result_particle_shift =
                        *contact.normal * (contact.depth * na::convert(0.9));
                }
            }
        }
    }
}
