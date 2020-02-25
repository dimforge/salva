#[cfg(feature = "parallel")]
use rayon::prelude::*;

use na::{self, RealField, Unit};

use crate::geometry::ParticlesContacts;

use crate::math::Vector;
use crate::object::Fluid;
use crate::solver::NonPressureForce;
use crate::TimestepManager;

#[derive(Clone)]
pub struct Akinci2013SurfaceTension<N: RealField> {
    tension_coefficient: N,
    normals: Vec<Vector<N>>,
}

impl<N: RealField> Akinci2013SurfaceTension<N> {
    pub fn new(tension_coefficient: N) -> Self {
        Self {
            tension_coefficient,
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
    let normalizer = na::convert::<_, N>(32.0f64) / (N::pi() * h.powi(9));
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

impl<N: RealField> NonPressureForce<N> for Akinci2013SurfaceTension<N> {
    fn solve(
        &mut self,
        timestep: &TimestepManager<N>,
        kernel_radius: N,
        fluid_fluid_contacts: &ParticlesContacts<N>,
        fluid: &mut Fluid<N>,
        densities: &[N],
    ) {
        self.init(fluid);
        let _2: N = na::convert(2.0f64);

        self.compute_normals(kernel_radius, fluid_fluid_contacts, fluid, densities);

        // Compute and apply forces.
        let normals = &self.normals;
        let tension_coefficient = self.tension_coefficient;
        let volumes = &mut fluid.volumes;
        let density0 = fluid.density0;
        let positions = &fluid.positions;

        par_iter_mut!(fluid.accelerations)
            .enumerate()
            .for_each(|(i, acceleration_i)| {
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

                        let cohesion_acc =
                            cohesion_vec * (-tension_coefficient * volumes[c.j] * density0);
                        let curvature_acc = (normals[c.i] - normals[c.j]) * -tension_coefficient;
                        let kij = _2 * density0 / (densities[c.i] + densities[c.j]);
                        *acceleration_i += (curvature_acc + cohesion_acc) * kij;
                    }
                }
            })
    }

    fn apply_permutation(&mut self, _: &[usize]) {}
}
