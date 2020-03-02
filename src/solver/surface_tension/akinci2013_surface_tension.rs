#[cfg(feature = "parallel")]
use rayon::prelude::*;

use na::{self, RealField, Unit};

use crate::geometry::ParticlesContacts;

use crate::math::Vector;
use crate::object::{Boundary, Fluid};
use crate::solver::NonPressureForce;
use crate::TimestepManager;

#[derive(Clone)]
/// Surface tension method introduced by Akinci et al. 2013
///
/// This combines both cohesion forces as well as curvature minimization forces.
/// This also includes adhesion forces for fluid/boundary interactions.
pub struct Akinci2013SurfaceTension<N: RealField> {
    fluid_tension_coefficient: N,
    boundary_adhesion_coefficient: N,
    normals: Vec<Vector<N>>,
}

impl<N: RealField> Akinci2013SurfaceTension<N> {
    /// Initializes a surface tension with the given surface tension coefficient and boundary adhesion coefficients.
    ///
    /// Both those coefficients are typically in [0.0, 1.0].
    pub fn new(fluid_tension_coefficient: N, boundary_adhesion_coefficient: N) -> Self {
        Self {
            fluid_tension_coefficient,
            boundary_adhesion_coefficient,
            normals: Vec::new(),
        }
    }

    fn init(&mut self, fluid: &Fluid<N>) {
        if self.normals.len() != fluid.num_particles() {
            self.normals.resize(fluid.num_particles(), Vector::zeros());
        }
    }

    fn compute_normals(
        &mut self,
        kernel_radius: N,
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid: &Fluid<N>,
        densities: &[N],
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

fn cohesion_kernel<N: RealField>(r: N, h: N) -> N {
    #[cfg(feature = "dim3")]
    let normalizer = na::convert::<_, N>(32.0f64) / (N::pi() * h.powi(9));
    // FIXME: not sure this is the right formula for the 2D version.
    #[cfg(feature = "dim2")]
    let normalizer = na::convert::<_, N>(32.0f64) / (N::pi() * h.powi(8));
    let _2: N = na::convert(2.0f64);

    let coeff = if r <= h / _2 {
        _2 * (h - r).powi(3) * r.powi(3) - h.powi(6) / na::convert(64.0f64)
    } else if r <= h {
        (h - r).powi(3) * r.powi(3)
    } else {
        N::zero()
    };

    normalizer * coeff
}

fn adhesion_kernel<N: RealField>(r: N, h: N) -> N {
    let _2: N = na::convert(2.0f64);

    if r > h / _2 && r <= h {
        let _4: N = na::convert(4.0f64);
        let _6: N = na::convert(6.0f64);

        #[cfg(feature = "dim3")]
        let normalizer = na::convert::<_, N>(0.007) / h.powf(na::convert(3.25f64));
        // FIXME: not sure this is the right formula for the 2D version.
        #[cfg(feature = "dim2")]
        let normalizer = na::convert::<_, N>(0.007) / h.powf(na::convert(2.25f64));

        // NOTE: the .max(N::zero()) prevents NaN due to float rounding errors.
        let coeff = ((-_4 * r * r / h + _6 * r - _2 * h).max(N::zero())).powf(na::convert(0.25));

        normalizer * coeff
    } else {
        N::zero()
    }
}

impl<N: RealField> NonPressureForce<N> for Akinci2013SurfaceTension<N> {
    fn solve(
        &mut self,
        _timestep: &TimestepManager<N>,
        kernel_radius: N,
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid_boundaries_contacts: &ParticlesContacts<N>,
        fluid: &mut Fluid<N>,
        boundaries: &[Boundary<N>],
        densities: &[N],
    ) {
        self.init(fluid);
        let _2: N = na::convert(2.0f64);

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
                if self.fluid_tension_coefficient != N::zero() {
                    for c in fluid_fluid_contacts
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        if c.i_model == c.j_model {
                            let dpos = positions[c.i] - positions[c.j];
                            let cohesion_vec = if let Some((dir, dist)) =
                                Unit::try_new_and_get(dpos, N::default_epsilon())
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

                if boundary_adhesion_coefficient != N::zero() {
                    for c in fluid_boundaries_contacts
                        .particle_contacts(i)
                        .read()
                        .unwrap()
                        .iter()
                    {
                        let dpos = positions[c.i] - boundaries[c.j_model].positions[c.j];
                        let adhesion_vec = if let Some((dir, dist)) =
                            Unit::try_new_and_get(dpos, N::default_epsilon())
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
