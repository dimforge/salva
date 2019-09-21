use crate::math::{Point, Vector};
use na::{RealField, Unit};

/// Kernel functions for performing approximations within the PBF method.
pub trait Kernel: Send + Sync {
    /// Evaluates the kernel for the given scalar `r` and the reference support length `h`.
    fn scalar_apply<N: RealField>(r: N, h: N) -> N;
    /// Evaluates the kernel derivative for the given scalar `r` and the reference support length `h`.
    fn scalar_apply_diff<N: RealField>(r: N, h: N) -> N;

    /// Evaluate the kernel for the given vector.
    fn apply<N: RealField>(v: Vector<N>, h: N) -> N {
        Self::scalar_apply(v.norm(), h)
    }

    /// Differential wrt. the coordinates of `v`.
    fn apply_diff<N: RealField>(v: Vector<N>, h: N) -> Vector<N> {
        if let Some((dir, norm)) = Unit::try_new_and_get(v, N::default_epsilon()) {
            *dir * Self::scalar_apply_diff(norm, h)
        } else {
            Vector::zeros()
        }
    }

    /// Evaluate the kernel for the vector equal to `p1 - p2`.
    fn points_apply<N: RealField>(p1: &Point<N>, p2: &Point<N>, h: N) -> N {
        Self::apply(p1 - p2, h)
    }

    /// Differential wrt. the coordinates of `p1`.
    fn points_apply_diff1<N: RealField>(p1: &Point<N>, p2: &Point<N>, h: N) -> Vector<N> {
        Self::apply_diff(p1 - p2, h)
    }

    /// Differential wrt. the coordinates of `p2`.
    fn points_apply_diff2<N: RealField>(p1: &Point<N>, p2: &Point<N>, h: N) -> Vector<N> {
        -Self::apply_diff(p1 - p2, h)
    }
}
