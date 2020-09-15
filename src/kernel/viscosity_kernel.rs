use crate::kernel::Kernel;
use crate::math::Real;
use na::RealField;

/// The Viscosity smoothing kernel.
///
/// Refer to "Particle-Based Fluid Simulation for Interactive Applications", Müller et al.pub struct ViscosityKernel;
#[derive(Copy, Clone, Debug)]
pub struct ViscosityKernel;

impl Kernel for ViscosityKernel {
    fn scalar_apply(r: Real, h: Real) -> N {
        assert!(r >= N::zero());

        let _2: Real = na::convert(2.0);
        let _3: Real = na::convert(3.0);

        #[cfg(feature = "dim2")]
        let normalizer = na::convert::<_, N>(10.0) / (_3 * N::pi() * h.powi(2));
        #[cfg(feature = "dim3")]
        let normalizer = na::convert::<_, N>(15.0) / (_2 * N::pi() * h.powi(3));

        if r > N::zero() && r <= h {
            let rr_hh = r * r / (h * h);
            normalizer * (rr_hh * (N::one() - r / (_2 * h)) + h / (_2 * r) - N::one())
        } else {
            N::zero()
        }
    }

    fn scalar_apply_diff(r: Real, h: Real) -> N {
        assert!(r >= N::zero());

        let _2: Real = na::convert(2.0);
        let _3: Real = na::convert(3.0);

        #[cfg(feature = "dim2")]
        let normalizer = na::convert::<_, N>(10.0) / (_3 * N::pi() * h.powi(2));
        #[cfg(feature = "dim3")]
        let normalizer = na::convert::<_, N>(15.0) / (_2 * N::pi() * h.powi(3));

        if r > N::zero() && r <= h {
            let rr = r * r;
            let hh = h * h;
            let hhh = hh * h;
            normalizer * (-_3 * rr / (_2 * hhh) + _2 * r / hh - h / (_2 * rr))
        } else {
            N::zero()
        }
    }
}
