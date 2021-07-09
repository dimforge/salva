use super::na::{Point3, Vector3};
use salva3d::object::Fluid;

pub fn cube_fluid(ni: usize, nj: usize, nk: usize, particle_rad: f32, density: f32) -> Fluid {
    let mut points = Vec::new();
    let half_extents = Vector3::new(ni as f32, nj as f32, nk as f32) * particle_rad;

    for i in 0..ni {
        for j in 0..nj {
            for k in 0..nk {
                let x = (i as f32) * particle_rad * 2.0;
                let y = (j as f32) * particle_rad * 2.0;
                let z = (k as f32) * particle_rad * 2.0;
                points.push(Point3::new(x, y, z) + Vector3::repeat(particle_rad) - half_extents);
            }
        }
    }

    Fluid::new(points, particle_rad, density)
}
