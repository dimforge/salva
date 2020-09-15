use crate::kernel::Kernel;
use crate::math::Real;
use na::RealField;

/// The Poly6 smoothing kernel.
///
/// Refer to "Particle-Based Fluid Simulation for Interactive Applications", Müller et al.
#[derive(Copy, Clone, Debug)]
pub struct Poly6Kernel;

impl Kernel for Poly6Kernel {
    fn scalar_apply(r: Real, h: Real) -> N {
        assert!(r >= N::zero());

        #[cfg(feature = "dim2")]
        let normalizer = na::convert::<_, N>(4.0) / (N::pi() * h.powi(8));
        #[cfg(feature = "dim3")]
        let normalizer = na::convert::<_, N>(315.0 / 64.0) / (N::pi() * h.powi(9));

        if r <= h {
            normalizer * (h * h - r * r).powi(3)
        } else {
            N::zero()
        }
    }

    fn scalar_apply_diff(r: Real, h: Real) -> N {
        assert!(r >= N::zero());

        #[cfg(feature = "dim2")]
        let normalizer = na::convert::<_, N>(4.0) / (N::pi() * h.powi(8));
        #[cfg(feature = "dim3")]
        let normalizer = na::convert::<_, N>(315.0 / 64.0) / (N::pi() * h.powi(9));

        if r <= h {
            normalizer * (h * h - r * r).powi(2) * r * na::convert(-6.0)
        } else {
            N::zero()
        }
    }
}
