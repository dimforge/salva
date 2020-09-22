use crate::kernel::Kernel;
use crate::math::Real;
use na::RealField;

/// The Poly6 smoothing kernel.
///
/// Refer to "Particle-Based Fluid Simulation for Interactive Applications", Müller et al.
#[derive(Copy, Clone, Debug)]
pub struct Poly6Kernel;

impl Kernel for Poly6Kernel {
    fn scalar_apply(r: Real, h: Real) -> Real {
        assert!(r >= na::zero::<Real>());

        #[cfg(feature = "dim2")]
        let normalizer = na::convert::<_, Real>(4.0) / (Real::pi() * h.powi(8));
        #[cfg(feature = "dim3")]
        let normalizer = na::convert::<_, Real>(315.0 / 64.0) / (Real::pi() * h.powi(9));

        if r <= h {
            normalizer * (h * h - r * r).powi(3)
        } else {
            na::zero::<Real>()
        }
    }

    fn scalar_apply_diff(r: Real, h: Real) -> Real {
        assert!(r >= na::zero::<Real>());

        #[cfg(feature = "dim2")]
        let normalizer = na::convert::<_, Real>(4.0) / (Real::pi() * h.powi(8));
        #[cfg(feature = "dim3")]
        let normalizer = na::convert::<_, Real>(315.0 / 64.0) / (Real::pi() * h.powi(9));

        if r <= h {
            normalizer * (h * h - r * r).powi(2) * r * na::convert::<_, Real>(-6.0)
        } else {
            na::zero::<Real>()
        }
    }
}
