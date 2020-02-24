use crate::math::Point;
use na::RealField;
use num_traits::float::FloatCore;
use std::cmp::Ordering;

pub fn apply_permutation<T: Clone>(permutation: &[usize], data: &[T]) -> Vec<T> {
    permutation.iter().map(|i| data[*i].clone()).collect()
}

pub fn compute_points_z_order<N: RealField>(points: &[Point<N>]) -> Vec<usize> {
    let mut indices: Vec<_> = (0..points.len()).collect();
    indices.sort_unstable_by(|i, j| {
        z_order_floats(points[*i].coords.as_slice(), points[*j].coords.as_slice())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}

// Fast construction of k-Nearest Neighbor Graphs for Point Clouds
// Michael Connor, Piyush Kumar
// Algorithm 1
//
// http://compgeom.com/~piyush/papers/tvcg_stann.pdf
pub fn z_order_floats<N: RealField>(p1: &[N], p2: &[N]) -> Option<Ordering> {
    assert_eq!(
        p1.len(),
        p2.len(),
        "Cannot compare array with different lengths."
    );
    let mut x = 0;
    let mut dim = 0;

    for j in 0..p1.len() {
        let y = xor_msb_float(p1[j], p2[j]);
        if x < y {
            x = y;
            dim = j;
        }
    }

    p1[dim].partial_cmp(&p2[dim])
}

fn xor_msb_float<N: RealField>(a: N, b: N) -> i16 {
    let fa = na::try_convert::<_, f64>(a).unwrap();
    let fb = na::try_convert::<_, f64>(a).unwrap();
    let (mantissa1, exponent1, sign1) = fa.integer_decode();
    let (mantissa2, exponent2, sign2) = fb.integer_decode();
    let x = exponent1; // To keep the same notation as the paper.
    let y = exponent2; // To keep the same notation as the paper.

    if x == y {
        let z = msdb(mantissa1, mantissa2);
        x - z
    } else if y < x {
        x
    } else {
        y
    }
}

fn msdb(x: u64, y: u64) -> i16 {
    64i16 - (x ^ y).leading_zeros() as i16
}
