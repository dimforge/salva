use crate::kernel::Kernel;
use na::RealField;

/// The cubic spline SPH kernel.
///
/// See https://pysph.readthedocs.io/en/latest/reference/kernels.html
pub struct CubicSplineKernel;

impl Kernel for CubicSplineKernel {
    fn scalar_apply<N: RealField>(r: N, h: N) -> N {
        assert!(r >= N::zero());

        let q = r / h;
        #[cfg(feature = "dim2")]
        let normalizer = na::convert::<_, N>(40.0 / 7.0) / (N::pi() * h * h);
        #[cfg(feature = "dim3")]
        let normalizer = na::convert::<_, N>(8.0) / (N::pi() * h * h * h);

        let _2: N = na::convert(2.0);
        let rhs = if q <= na::convert(0.5) {
            N::one() + (q * q * q - q * q) * na::convert(6.0)
        } else if q <= N::one() {
            (N::one() - q).powi(3) * _2
        } else {
            N::zero()
        };

        normalizer * rhs

        /*
        let q = r / h;
        #[cfg(feature = "dim2")]
            let normalizer = na::convert::<_, N>(10.0 / 7.0) / (N::pi() * h * h);
        #[cfg(feature = "dim3")]
            let normalizer = N::one() / (N::pi() * h * h * h);

        let _2: N = na::convert(2.0);
        let _3: N = na::convert(3.0);
        let rhs = if q <= N::one() {
            N::one() - _3 / _2 * q * q * (N::one() - q / _2)
        } else if q <= _2 {
            (_2 - q).powi(3) / na::convert(4.0)
        } else {
            N::zero()
        };

        normalizer * rhs
        */
    }

    fn scalar_apply_diff<N: RealField>(r: N, h: N) -> N {
        assert!(r >= N::zero());

        let q = r / h;
        #[cfg(feature = "dim2")]
        let normalizer = na::convert::<_, N>(40.0 / 7.0) / (N::pi() * h * h);
        #[cfg(feature = "dim3")]
        let normalizer = na::convert::<_, N>(8.0) / (N::pi() * h * h * h);

        let _2: N = na::convert(2.0);
        let _3: N = na::convert(3.0);
        let rhs = if q <= na::convert(0.5) {
            (q * q * _3 - q * _2) * na::convert(6.0)
        } else if q <= N::one() {
            -(N::one() - q).powi(2) * na::convert(6.0)
        } else {
            N::zero()
        };

        normalizer * rhs

        /*
        let q = r / h;
        #[cfg(feature = "dim2")]
            let normalizer = na::convert::<_, N>(10.0 / 7.0) / (N::pi() * h * h);
        #[cfg(feature = "dim3")]
            let normalizer = N::one() / (N::pi() * h * h * h);

        let _2: N = na::convert(2.0);
        let _3: N = na::convert(3.0);
        let rhs = if q <= N::one() {
            -_3 * q * (N::one() - q * na::convert(3.0 / 4.0))
        } else if q <= _2 {
            -(_2 - q).powi(2) * na::convert(3.0 / 4.0)
        } else {
            N::zero()
        };

        normalizer * rhs
        */
    }
}
