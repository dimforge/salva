use crate::kernel::Kernel;
use na::RealField;

/// The Spiky smoothing kernel.
///
/// Refer to "Particle-Based Fluid Simulation for Interactive Applications", Müller et al.pub struct SpikyKernel;
pub struct SpikyKernel;

impl Kernel for SpikyKernel {
    fn scalar_apply<N: RealField>(r: N, h: N) -> N {
        assert!(r >= N::zero());

        #[cfg(feature = "dim2")]
        let normalizer = na::convert::<_, N>(10.0) / (N::pi() * h.powi(5));
        #[cfg(feature = "dim3")]
        let normalizer = na::convert::<_, N>(15.0) / (N::pi() * h.powi(6));

        if r <= h {
            normalizer * (h - r).powi(3)
        } else {
            N::zero()
        }
    }

    fn scalar_apply_diff<N: RealField>(r: N, h: N) -> N {
        assert!(r >= N::zero());

        #[cfg(feature = "dim2")]
        let normalizer = na::convert::<_, N>(10.0) / (N::pi() * h.powi(5));
        #[cfg(feature = "dim3")]
        let normalizer = na::convert::<_, N>(15.0) / (N::pi() * h.powi(6));

        if r <= h {
            -normalizer * (h - r).powi(2) * na::convert(3.0)
        } else {
            N::zero()
        }
    }
}
