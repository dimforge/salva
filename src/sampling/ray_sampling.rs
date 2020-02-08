use crate::math::{Isometry, Point, Vector, DIM};
use na::RealField;
use ncollide::bounding_volume::{BoundingVolume, AABB};
use ncollide::query::{Ray, RayCast};
use std::collections::HashSet;

pub fn surface_ray_sample<N: RealField, S: ?Sized + RayCast<N>>(
    shape: &S,
    volume: &AABB<N>,
    subdivision_size: N,
) -> Vec<Point<N>> {
    let mut points = Vec::new();
    points
}

pub fn volume_ray_sample<N: RealField, S: ?Sized + RayCast<N>>(
    shape: &S,
    volume: &AABB<N>,
    subdivision_size: N,
) -> Vec<Point<N>> {
    let mut quantized_points = HashSet::new();

    let volume = volume.loosened(subdivision_size);
    let maxs = volume.maxs();
    let origin = volume.mins();
    let mut curr = *origin;

    for i in 0..DIM {
        let j = (i + 1) % DIM;
        let k = (i + 2) % DIM;

        while curr[j] < maxs[j] {
            while curr[k] < maxs[k] {
                // Cast a ray along `z`.
                let mut dir = Vector::zeros();
                dir[i] = N::one();
                let mut ray = Ray::new(curr, dir);
                let mut prev_impact = None;

                while let Some(toi) = shape.toi_with_ray(&Isometry::identity(), &ray, false) {
                    if let Some(prev) = prev_impact {
                        sample_segment(
                            origin,
                            &curr,
                            prev,
                            ray.origin[i] + toi,
                            subdivision_size,
                            i,
                            &mut quantized_points,
                        );
                        prev_impact = None;
                    } else {
                        prev_impact = Some(ray.origin[i] + toi);
                        ray.origin[i] += toi + subdivision_size / na::convert(10.0);
                    }
                }

                curr[k] += subdivision_size;
            }

            curr[j] += subdivision_size;
            curr[k] = origin[k];
        }
    }

    unquantize_points(origin, subdivision_size, &quantized_points)
}

fn sample_segment<N: RealField>(
    origin: &Point<N>,
    start: &Point<N>,
    a: N,
    b: N,
    subdivision_size: N,
    dimension: usize,
    out: &mut HashSet<Point<u32>>,
) {
    let mut quantized_pt = (start - origin).map(|e| {
        na::try_convert::<_, f64>(e / subdivision_size)
            .unwrap()
            .round() as u32
    });
    let start_index = na::try_convert::<_, f64>((a - origin[dimension]) / subdivision_size)
        .unwrap()
        .round() as u32;
    let end_index = na::try_convert::<_, f64>((b - origin[dimension]) / subdivision_size)
        .unwrap()
        .round() as u32;

    for i in start_index..=end_index {
        quantized_pt[dimension] = i;
        let _ = out.insert(quantized_pt.into());
    }
}

fn unquantize_points<N: RealField>(
    origin: &Point<N>,
    subdivision_size: N,
    quantized_points: &HashSet<Point<u32>>,
) -> Vec<Point<N>> {
    quantized_points
        .iter()
        .map(|qpt| {
            origin
                + qpt
                    .coords
                    .map(|e| na::convert::<_, N>(e as f64) * subdivision_size)
        })
        .collect()
}
