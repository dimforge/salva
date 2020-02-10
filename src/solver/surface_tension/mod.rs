#[cfg(feature = "dim3")]
pub use self::akinci2013_surface_tension::Akinci2013SurfaceTension;
pub use self::he2014_surface_tension::He2014SurfaceTension;

#[cfg(feature = "dim3")]
mod akinci2013_surface_tension;
mod he2014_surface_tension;
