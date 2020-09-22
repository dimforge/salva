#[cfg(feature = "parallel")]
use rayon::prelude::*;

use approx::AbsDiffEq;
use na::{self, RealField, Unit};

use crate::geometry::ParticlesContacts;

use crate::math::{Real, Vector};
use crate::object::{Boundary, Fluid};
use crate::solver::NonPressureForce;
use crate::TimestepManager;

#[derive(Clone)]
/// Surface tension method introduced by Akinci et al. 2013
///
/// This combines both cohesion forces as well as curvature minimization forces.
/// This also includes adhesion forces for fluid/boundary interactions.
pub struct Akinci2013SurfaceTension {
    fluid_tension_coefficient: Real,
    boundary_adhesion_coefficient: Real,
    normals: Vec<Vector<Real>>,
}

impl Akinci2013SurfaceTension {
    /// Initializes a surface tension with the given surface tension coefficient and boundary adhesion coefficients.
    ///
    /// Both those coefficients are typically in [0.0, 1.0].
    pub fn new(fluid_tension_coefficient: Real, boundary_adhesion_coefficient: Real) -> Self {
        Self {
            fluid_tension_coefficient,
            boundary_adhesion_coefficient,
            normals: Vec::new(),
        }
    }

    fn init(&mut self, fluid: &Fluid) {
        if self.normals.len() != fluid.num_particles() {
            self.normals.resize(fluid.num_particles(), Vector::zeros());
        }
    }

    fn compute_normals(
        &mut self,
        kernel_radius: Real,
        fluid_fluid_contacts: &ParticlesContacts,
        fluid: &Fluid,
        densities: &[Real],
    ) {
        par_iter_mut!(self.normals)
            .enumerate()
            .for_each(|(i, normal_i)| {
                let mut normal = Vector::zeros();

                for c in fluid_fluid_contacts
                    .particle_contacts(i)
                    .read()
                    .unwrap()
                    .iter()
                {
                    if c.i_model == c.j_model {
                        normal += c.gradient * (fluid.particle_mass(c.j) / densities[c.j]);
                    }
                }

                *normal_i = normal * kernel_radius;
            })
    }
}

fn cohesion_kernel(r: Real, h: Real) -> Real {
    #[cfg(feature = "dim3")]
    let normalizer = na::convert::<_, Real>(32.0f64) / (Real::pi() * h.powi(9));
    // FIXME: not sure this is the right formula for the 2D version.
    #[cfg(feature = "dim2")]
    let normalizer = na::convert::<_, Real>(32.0f64) / (Real::pi() * h.powi(8));
    let _2: Real = na::convert::<_, Real>(2.0f64);

    let coeff = if r <= h / _2 {
        _2 * (h - r).powi(3) * r.powi(3) - h.powi(6) / na::convert::<_, Real>(64.0f64)
    } else if r <= h {
        (h - r).powi(3) * r.powi(3)
    } else {
        na::zero::<Real>()
    };

    normalizer * coeff
}

fn adhesion_kernel(r: Real, h: Real) -> Real {
    let _2: Real = na::convert::<_, Real>(2.0f64);

    if r > h / _2 && r <= h {
        let _4: Real = na::convert::<_, Real>(4.0f64);
        let _6: Real = na::convert::<_, Real>(6.0f64);

        #[cfg(feature = "dim3")]
        let normalizer = na::convert::<_, Real>(0.007) / h.powf(na::convert::<_, Real>(3.25f64));
        // FIXME: not sure this is the right formula for the 2D version.
        #[cfg(feature = "dim2")]
        let normalizer = na::convert::<_, Real>(0.007) / h.powf(na::convert::<_, Real>(2.25f64));

        // NOTE: the .max(na::zero::<Real>()) prevents NaN due to float rounding errors.
        let coeff = ((-_4 * r * r / h + _6 * r - _2 * h).max(na::zero::<Real>()))
            .powf(na::convert::<_, Real>(0.25));

        normalizer * coeff
    } else {
        na::zero::<Real>()
    }
}

impl NonPressureForce for Akinci2013SurfaceTension {
    fn solve(
        &mut self,
        _timestep: &TimestepManager,
        kernel_radius: Real,
        fluid_fluid_contacts: &ParticlesContacts,
        fluid_boundaries_contacts: &ParticlesContacts,
        fluid: &mut Fluid,
        boundaries: &[Boundary],
        densities: &[Real],
    ) {
        self.init(fluid);
        let _2: Real = na::convert::<_, Real>(2.0f64);

        self.compute_normals(kernel_radius, fluid_fluid_contacts, fluid, densities);

        // Compute and apply forces.
        let normals = &self.normals;
        let fluid_tension_coefficient = self.fluid_tension_coefficient;
        let boundary_adhesion_coefficient = self.boundary_adhesion_coefficient;
        let volumes = &mut fluid.volumes;
        let density0 = fluid.density0;
        let positions = &fluid.positions;

        par_iter_mut!(fluid.accelerations)
            .enumerate()
            .for_each(|(i, acceleration_i)| {
                if self.fluid_tension_coefficient != na::zero::<Real>() {
                    for c in fluid_fluid_contacts
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        if c.i_model == c.j_model {
                            let dpos = positions[c.i] - positions[c.j];
                            let cohesion_vec = if let Some((dir, dist)) =
                                Unit::try_new_and_get(dpos, Real::default_epsilon())
                            {
                                *dir * cohesion_kernel(dist, kernel_radius)
                            } else {
                                Vector::zeros()
                            };

                            let cohesion_acc = cohesion_vec
                                * (-fluid_tension_coefficient * volumes[c.j] * density0);
                            let curvature_acc =
                                (normals[c.i] - normals[c.j]) * -fluid_tension_coefficient;
                            let kij = _2 * density0 / (densities[c.i] + densities[c.j]);
                            *acceleration_i += (curvature_acc + cohesion_acc) * kij;
                        }
                    }
                }

                if boundary_adhesion_coefficient != na::zero::<Real>() {
                    for c in fluid_boundaries_contacts
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        let dpos = positions[c.i] - boundaries[c.j_model].positions[c.j];
                        let adhesion_vec = if let Some((dir, dist)) =
                            Unit::try_new_and_get(dpos, Real::default_epsilon())
                        {
                            *dir * adhesion_kernel(dist, kernel_radius)
                        } else {
                            Vector::zeros()
                        };

                        let mi = volumes[c.i] * density0;
                        let mj = boundaries[c.j_model].volumes[c.j] * density0;
                        let adhesion_acc = adhesion_vec * (boundary_adhesion_coefficient * mj);
                        *acceleration_i -= adhesion_acc;

                        boundaries[c.j_model].apply_force(c.j, adhesion_acc * mi);
                    }
                }
            })
    }

    fn apply_permutation(&mut self, _: &[usize]) {}
}
