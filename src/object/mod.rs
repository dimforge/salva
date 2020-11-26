//! Fluid and boundary objects that can be simulated.

pub use self::boundary::{Boundary, BoundaryHandle, BoundarySet};
pub use self::contiguous_arena::{ContiguousArena, ContiguousArenaIndex};
pub use self::fluid::{Fluid, FluidHandle, FluidSet};

mod boundary;
mod contiguous_arena;
mod fluid;

/// The identifier of a single particle.
pub enum ParticleId {
    FluidParticle(FluidHandle, usize),
    BoundaryParticle(BoundaryHandle, usize),
}
