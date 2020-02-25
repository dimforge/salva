use crate::math::{Isometry, Point, Vector, DIM};
use na::RealField;
use ncollide::bounding_volume::{BoundingVolume, AABB};
use ncollide::query::{Ray, RayCast};
use ncollide::shape::Shape;
use std::collections::HashSet;

pub fn shape_surface_ray_sample<N: RealField, S: ?Sized + Shape<N>>(
    shape: &S,
    particle_rad: N,
) -> Option<Vec<Point<N>>> {
    let rc = shape.as_ray_cast()?;
    let aabb = shape.local_aabb();
    Some(surface_ray_sample(rc, &aabb, particle_rad))
}

pub fn shape_volume_ray_sample<N: RealField, S: ?Sized + Shape<N>>(
    shape: &S,
    particle_rad: N,
) -> Option<Vec<Point<N>>> {
    let rc = shape.as_ray_cast()?;
    let aabb = shape.local_aabb();
    Some(volume_ray_sample(rc, &aabb, particle_rad))
}

pub fn surface_ray_sample<N: RealField, S: ?Sized + RayCast<N>>(
    shape: &S,
    volume: &AABB<N>,
    particle_rad: N,
) -> Vec<Point<N>> {
    let mut quantized_points = HashSet::new();
    let subdivision_size = particle_rad * na::convert(2.0);

    let volume = volume.loosened(subdivision_size);
    let maxs = volume.maxs();
    let origin = volume.mins() + Vector::repeat(subdivision_size / na::convert(2.0));
    let mut curr = origin;

    let mut perform_cast = |i, curr| {
        let mut dir = Vector::zeros();
        dir[i] = N::one();
        let mut ray = Ray::new(curr, dir);
        let mut entry_point = true;

        while let Some(toi) = shape.toi_with_ray(&Isometry::identity(), &ray, false) {
            let impact = ray.point_at(toi);
            let quantized_pt = quantize_point(&origin, &impact, subdivision_size, entry_point, i);
            let _ = quantized_points.insert(quantized_pt);
            ray.origin[i] += toi + subdivision_size / na::convert(10.0);
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

pub fn volume_ray_sample<N: RealField, S: ?Sized + RayCast<N>>(
    shape: &S,
    volume: &AABB<N>,
    particle_rad: N,
) -> Vec<Point<N>> {
    let mut quantized_points = HashSet::new();
    let subdivision_size = particle_rad * na::convert(2.0);

    let volume = volume.loosened(subdivision_size);
    let maxs = volume.maxs();
    let origin = volume.mins() + Vector::repeat(subdivision_size / na::convert(2.0));

    let mut perform_cast = |i, curr| {
        let mut dir = Vector::zeros();
        dir[i] = N::one();
        let mut ray = Ray::new(curr, dir);
        let mut prev_impact = None;

        while let Some(toi) = shape.toi_with_ray(&Isometry::identity(), &ray, false) {
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

            ray.origin[i] += toi + subdivision_size / na::convert(10.0);
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

fn quantize_point<N: RealField>(
    origin: &Point<N>,
    point: &Point<N>,
    subdivision_size: N,
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
