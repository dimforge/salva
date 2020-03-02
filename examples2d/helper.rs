use na::{Point2, Vector2};
use salva2d::object::Fluid;

pub fn cube_fluid(ni: usize, nj: usize, particle_rad: f32, density: f32) -> Fluid<f32> {
    let mut points = Vec::new();
    let half_extents = Vector2::new(ni as f32, nj as f32) * particle_rad;

    for i in 0..ni {
        for j in 0..nj {
            let x = (i as f32) * particle_rad * 2.0;
            let y = (j as f32) * particle_rad * 2.0;
            points.push(Point2::new(x, y) + Vector2::repeat(particle_rad) - half_extents);
        }
    }

    Fluid::new(points, particle_rad, density)
}
