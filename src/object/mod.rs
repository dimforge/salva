//! Fluid and boundary objects that can be simulated.

pub use self::boundary::{Boundary, BoundaryHandle};
pub use self::fluid::{Fluid, FluidHandle};

mod boundary;
mod fluid;
