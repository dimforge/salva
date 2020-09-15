use crate::math::{Point, Real, Vector};
use na::{RealField, Unit};

/// Kernel functions for performing approximations within the PBF/SPH methods.
pub trait Kernel: Send + Sync {
    /// Evaluates the kernel for the given scalar `r` and the reference support length `h`.
    fn scalar_apply(r: Real, h: Real) -> N;
    /// Evaluates the kernel derivative for the given scalar `r` and the reference support length `h`.
    fn scalar_apply_diff(r: Real, h: Real) -> N;

    /// Evaluate the kernel for the given vector.
    fn apply(v: Vector<Real>, h: Real) -> N {
        Self::scalar_apply(v.norm(), h)
    }

    /// Differential wrt. the coordinates of `v`.
    fn apply_diff(v: Vector<Real>, h: Real) -> Vector<Real> {
        if let Some((dir, norm)) = Unit::try_new_and_get(v, N::default_epsilon()) {
            *dir * Self::scalar_apply_diff(norm, h)
        } else {
            Vector::zeros()
        }
    }

    /// Evaluate the kernel for the vector equal to `p1 - p2`.
    fn points_apply(p1: &Point<Real>, p2: &Point<Real>, h: Real) -> N {
        Self::apply(p1 - p2, h)
    }

    /// Differential wrt. the coordinates of `p1`.
    fn points_apply_diff1(p1: &Point<Real>, p2: &Point<Real>, h: Real) -> Vector<Real> {
        Self::apply_diff(p1 - p2, h)
    }

    /// Differential wrt. the coordinates of `p2`.
    fn points_apply_diff2(p1: &Point<Real>, p2: &Point<Real>, h: Real) -> Vector<Real> {
        -Self::apply_diff(p1 - p2, h)
    }
}
