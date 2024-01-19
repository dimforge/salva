use crate::math::{Isometry, Point, Real, Vector, DIM};

use parry::bounding_volume::{Aabb, BoundingVolume};
use parry::query::{Ray, RayCast};
use rapier::geometry::Shape;
use std::collections::HashSet;

/// Samples the surface of `shape` with a method based on ray-casting.
pub fn shape_surface_ray_sample<S: ?Sized + Shape>(
    shape: &S,
    particle_rad: Real,
) -> Option<Vec<Point<Real>>> {
    let aabb = shape.compute_aabb(&Isometry::identity());
    Some(surface_ray_sample(shape, &aabb, particle_rad))
}

/// Samples the volume of `shape` with a method based on ray-casting.
pub fn shape_volume_ray_sample<S: ?Sized + Shape>(
    shape: &S,
    particle_rad: Real,
) -> Option<Vec<Point<Real>>> {
    let aabb = shape.compute_aabb(&Isometry::identity());
    Some(volume_ray_sample(shape, &aabb, particle_rad))
}

/// Samples the surface of `shape` with a method based on ray-casting.
pub fn surface_ray_sample<S: ?Sized + RayCast>(
    shape: &S,
    volume: &Aabb,
    particle_rad: Real,
) -> Vec<Point<Real>> {
    let mut quantized_points = HashSet::new();
    let subdivision_size = particle_rad * na::convert::<_, Real>(2.0);

    let volume = volume.loosened(subdivision_size);
    let maxs = volume.maxs;
    let origin = volume.mins + Vector::repeat(subdivision_size / na::convert::<_, Real>(2.0));
    let mut curr = origin;

    let mut perform_cast = |i, curr| {
        let mut dir = Vector::zeros();
        dir[i] = na::one::<Real>();
        let mut ray = Ray::new(curr, dir);
        let mut entry_point = true;

        while let Some(toi) = shape.cast_local_ray(&ray, Real::MAX, false) {
            let impact = ray.point_at(toi);
            let quantized_pt = quantize_point(&origin, &impact, subdivision_size, entry_point, i);
            let _ = quantized_points.insert(quantized_pt);
            ray.origin[i] += toi + subdivision_size / na::convert::<_, Real>(10.0);
            entry_point = !entry_point;
        }
    };

    #[cfg(feature = "dim3")]
    for i in 0..DIM {
        let j = (i + 1) % DIM;
        let k = (i + 2) % DIM;

        while curr[j] < maxs[j] {
            while curr[k] < maxs[k] {
                perform_cast(i, curr);
                curr[k] += subdivision_size;
            }

            curr[j] += subdivision_size;
            curr[k] = origin[k];
        }

        curr[i] += subdivision_size;
        curr[j] = origin[j];
    }

    #[cfg(feature = "dim2")]
    for i in 0..DIM {
        let j = (i + 1) % DIM;

        while curr[j] < maxs[j] {
            perform_cast(i, curr);
            curr[j] += subdivision_size;
        }

        curr[i] += subdivision_size;
        curr[j] = origin[j];
    }

    unquantize_points(&origin, subdivision_size, &quantized_points)
}

/// Samples the volume of `shape` with a method based on ray-casting.
pub fn volume_ray_sample<S: ?Sized + RayCast>(
    shape: &S,
    volume: &Aabb,
    particle_rad: Real,
) -> Vec<Point<Real>> {
    let mut quantized_points = HashSet::new();
    let subdivision_size = particle_rad * na::convert::<_, Real>(2.0);

    let volume = volume.loosened(subdivision_size);
    let maxs = volume.maxs;
    let origin = volume.mins + Vector::repeat(subdivision_size / na::convert::<_, Real>(2.0));

    let mut perform_cast = |i, curr| {
        let mut dir = Vector::zeros();
        dir[i] = na::one::<Real>();
        let mut ray = Ray::new(curr, dir);
        let mut prev_impact = None;

        while let Some(toi) = shape.cast_local_ray(&ray, Real::MAX, false) {
            if let Some(prev) = prev_impact {
                sample_segment(
                    &origin,
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
            }

            ray.origin[i] += toi + subdivision_size / na::convert::<_, Real>(10.0);
        }
    };

    let mut curr = origin;

    #[cfg(feature = "dim3")]
    for i in 0..DIM {
        let j = (i + 1) % DIM;
        let k = (i + 2) % DIM;

        while curr[j] < maxs[j] {
            while curr[k] < maxs[k] {
                perform_cast(i, curr);
                curr[k] += subdivision_size;
            }

            curr[j] += subdivision_size;
            curr[k] = origin[k];
        }

        curr[i] += subdivision_size;
        curr[j] = origin[j];
    }

    #[cfg(feature = "dim2")]
    for i in 0..DIM {
        let j = (i + 1) % DIM;

        while curr[j] < maxs[j] {
            perform_cast(i, curr);
            curr[j] += subdivision_size;
        }

        curr[i] += subdivision_size;
        curr[j] = origin[j];
    }

    unquantize_points(&origin, subdivision_size, &quantized_points)
}

fn sample_segment(
    origin: &Point<Real>,
    start: &Point<Real>,
    a: Real,
    b: Real,
    subdivision_size: Real,
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

fn unquantize_points(
    origin: &Point<Real>,
    subdivision_size: Real,
    quantized_points: &HashSet<Point<u32>>,
) -> Vec<Point<Real>> {
    quantized_points
        .iter()
        .map(|qpt| {
            origin
                + qpt
                    .coords
                    .map(|e| na::convert::<_, Real>(e as f64) * subdivision_size)
        })
        .collect()
}

fn quantize_point(
    origin: &Point<Real>,
    point: &Point<Real>,
    subdivision_size: Real,
    entry_point: bool,
    leading_dimension: usize,
) -> Point<u32> {
    let mut dpt = point - origin;
    for i in 0..DIM {
        if i == leading_dimension {
            if entry_point {
                dpt[i] = (dpt[i] / subdivision_size).ceil();
            } else {
                dpt[i] = (dpt[i] / subdivision_size).floor();
            }
        } else {
            dpt[i] = (dpt[i] / subdivision_size).round();
        }
    }

    dpt.map(|e| na::try_convert::<_, f64>(e).unwrap() as u32)
        .into()
}
