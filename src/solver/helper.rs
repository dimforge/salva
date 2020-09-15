use crate::geometry::ParticlesContacts;
use crate::kernel::Kernel;
use crate::math::Real;
use crate::object::{Boundary, Fluid};
use na::RealField;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub fn update_fluid_contacts<N: RealField, KernelDensity: Kernel, KernelGradient: Kernel>(
    kernel_radius: Real,
    fluid_fluid_contacts: &mut [ParticlesContacts],
    fluid_boundary_contacts: &mut [ParticlesContacts],
    fluids: &[Fluid],
    boundaries: &[Boundary],
) {
    for contacts in fluid_fluid_contacts.iter_mut() {
        par_iter_mut!(contacts.contacts_mut()).for_each(|contacts| {
            for c in contacts.get_mut().unwrap() {
                let fluid1 = &fluids[c.i_model];
                let fluid2 = &fluids[c.j_model];
                let pi = fluid1.positions[c.i];
                let pj = fluid2.positions[c.j];

                c.weight = KernelDensity::points_apply(&pi, &pj, kernel_radius);
                c.gradient = KernelGradient::points_apply_diff1(&pi, &pj, kernel_radius);
            }
        })
    }

    for contacts in fluid_boundary_contacts.iter_mut() {
        par_iter_mut!(contacts.contacts_mut()).for_each(|contacts| {
            for c in contacts.get_mut().unwrap() {
                let fluid1 = &fluids[c.i_model];
                let bound2 = &boundaries[c.j_model];

                let pi = fluid1.positions[c.i];
                let pj = bound2.positions[c.j];

                c.weight = KernelDensity::points_apply(&pi, &pj, kernel_radius);
                c.gradient = KernelGradient::points_apply_diff1(&pi, &pj, kernel_radius);
            }
        })
    }
}

pub fn update_boundary_contacts<N: RealField, KernelDensity: Kernel, KernelGradient: Kernel>(
    kernel_radius: Real,
    boundary_boundary_contacts: &mut [ParticlesContacts],
    boundaries: &[Boundary],
) {
    for contacts in boundary_boundary_contacts.iter_mut() {
        par_iter_mut!(contacts.contacts_mut()).for_each(|contacts| {
            for c in contacts.get_mut().unwrap() {
                let bound1 = &boundaries[c.i_model];
                let bound2 = &boundaries[c.j_model];

                let pi = bound1.positions[c.i];
                let pj = bound2.positions[c.j];

                c.weight = KernelDensity::points_apply(&pi, &pj, kernel_radius);
                c.gradient = KernelGradient::points_apply_diff1(&pi, &pj, kernel_radius);
            }
        })
    }
}
