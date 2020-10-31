use crate::kernel::Kernel;
use crate::math::Real;
use na::RealField;

/// The cubic spline smoothing kernel.
///
/// See https://pysph.readthedocs.io/en/latest/reference/kernels.html
#[derive(Copy, Clone, Debug)]
pub struct CubicSplineKernel;

impl Kernel for CubicSplineKernel {
    fn scalar_apply(r: Real, h: Real) -> Real {
        assert!(r >= na::zero::<Real>());

        #[cfg(feature = "dim2")]
        let normalizer = na::convert::<_, Real>(40.0 / 7.0) / (Real::pi() * h * h);
        #[cfg(feature = "dim3")]
        let normalizer = na::convert::<_, Real>(8.0) / (Real::pi() * h * h * h);

        let _2: Real = na::convert::<_, Real>(2.0);
        let q = r / h;

        let rhs = if q <= na::convert::<_, Real>(0.5) {
            let q2 = q * q;
            na::one::<Real>() + (q2 * q - q2) * na::convert::<_, Real>(6.0)
        } else if q <= na::one::<Real>() {
            (na::one::<Real>() - q).powi(3) * _2
        } else {
            na::zero::<Real>()
        };

        normalizer * rhs

        /*
        let q = r / h;
        #[cfg(feature = "dim2")]
            let normalizer = na::convert::<_, Real>(10.0 / 7.0) / (Real::pi() * h * h);
        #[cfg(feature = "dim3")]
            let normalizer = na::one::<Real>() / (Real::pi() * h * h * h);

        let _2: Real = na::convert::<_, Real>(2.0);
        let _3: Real = na::convert::<_, Real>(3.0);
        let rhs = if q <= na::one::<Real>() {
            na::one::<Real>() - _3 / _2 * q * q * (na::one::<Real>() - q / _2)
        } else if q <= _2 {
            (_2 - q).powi(3) / na::convert::<_, Real>(4.0)
        } else {
            na::zero::<Real>()
        };

        normalizer * rhs
        */
    }

    fn scalar_apply_diff(r: Real, h: Real) -> Real {
        assert!(r >= na::zero::<Real>());

        #[cfg(feature = "dim2")]
        let normalizer = na::convert::<_, Real>(40.0 / 7.0) / (Real::pi() * h * h);
        #[cfg(feature = "dim3")]
        let normalizer = na::convert::<_, Real>(8.0) / (Real::pi() * h * h * h);

        let _1: Real = na::convert::<_, Real>(1.0);
        let _2: Real = na::convert::<_, Real>(2.0);
        let _3: Real = na::convert::<_, Real>(3.0);
        let _eps: Real = na::convert::<_, Real>(1.0e-5);
        let q = r / h;

        let rhs = if q > _1 || q <= _eps {
            na::zero::<Real>()
        } else if q <= na::convert::<_, Real>(0.5) {
            (q * _3 - _2) * q * na::convert::<_, Real>(6.0)
        } else {
            // 0.5 < q <= 1.0
            let one_q = _1 - q;
            -one_q * one_q * na::convert::<_, Real>(6.0)
        };

        normalizer * rhs / h

        /*
        let q = r / h;
        #[cfg(feature = "dim2")]
            let normalizer = na::convert::<_, Real>(10.0 / 7.0) / (Real::pi() * h * h);
        #[cfg(feature = "dim3")]
            let normalizer = na::one::<Real>() / (Real::pi() * h * h * h);

        let _2: Real = na::convert::<_, Real>(2.0);
        let _3: Real = na::convert::<_, Real>(3.0);
        let rhs = if q <= na::one::<Real>() {
            -_3 * q * (na::one::<Real>() - q * na::convert::<_, Real>(3.0 / 4.0))
        } else if q <= _2 {
            -(_2 - q).powi(2) * na::convert::<_, Real>(3.0 / 4.0)
        } else {
            na::zero::<Real>()
        };

        normalizer * rhs
        */
    }
}
