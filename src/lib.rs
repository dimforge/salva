/*!
**Salva** is a 2 and 3-dimensional particle-based fluid simulation engine for games and animations.
It uses [nalgebra](https://nalgebra.org) for vector/matrix math and can optionally interface with
[rapier](https://rapier.rs) for two-way coupling with rigid bodies, multibodies, and deformable bodies.
2D and 3D implementations both share (mostly) the same code!


Examples are available in the `examples2d` and `examples3d` directories.  Because those demos are based on
WASM and WebGl 1.0 they should work on most modern browsers. Feel free to ask for help
and discuss features on the official [discord](https://discord.gg/vt9DJSW).

## Why the name Salva?

The name of this library is inspired from the famous surrealist artist `Salvador Dalì`. The logo of `Salva`
 is inspired from its renown painting [The Persistence of Memory](https://en.wikipedia.org/wiki/The_Persistence_of_Memory).

## Features
- **Pressure resolution:** DFSPH and IISPH.
- **Viscosity:** DFSPH viscosity, Artificial viscosity, and XSPH viscosity.
- **Surface tension:** WCSPH surface tension, and methods from He et al. 2014 and Akinci et al. 2013
- **Elasticity:** method from Becker et al. 2009
- **Multiphase fluids**: mix several fluids with different characteristics (densities, viscosities, etc.)
- Optional **two-way coupling** with bodies from **rapier**.
- **WASM** support
*/
#![deny(non_camel_case_types)]
#![deny(unused_parens)]
#![deny(non_upper_case_globals)]
#![deny(unused_qualifications)]
#![warn(missing_docs)] // FIXME: deny this
#![deny(unused_results)]
#![allow(type_alias_bounds)]
#![warn(non_camel_case_types)]
#![allow(missing_copy_implementations)]
#![doc(html_root_url = "https://salva.rs/rustdoc/")]
#![doc(html_logo_url = "https://salva.rs/img/logo_salva_rustdoc.svg")]

extern crate nalgebra as na;
extern crate num_traits as num;
#[cfg(all(feature = "dim2", feature = "parry"))]
pub extern crate parry2d as parry;
#[cfg(all(feature = "dim3", feature = "parry"))]
pub extern crate parry3d as parry;
#[cfg(all(feature = "dim2", feature = "rapier"))]
pub extern crate rapier2d as rapier;
#[cfg(all(feature = "dim3", feature = "rapier"))]
pub extern crate rapier3d as rapier;
#[cfg(all(feature = "dim2", feature = "rapier-testbed"))]
extern crate rapier_testbed2d as rapier_testbed;
#[cfg(all(feature = "dim3", feature = "rapier-testbed"))]
extern crate rapier_testbed3d as rapier_testbed;

macro_rules! par_iter {
    ($t: expr) => {{
        #[cfg(not(feature = "parallel"))]
        let it = $t.iter();

        #[cfg(feature = "parallel")]
        let it = $t.par_iter();
        it
    }};
}

macro_rules! par_iter_mut {
    ($t: expr) => {{
        #[cfg(not(feature = "parallel"))]
        let it = $t.iter_mut();

        #[cfg(feature = "parallel")]
        let it = $t.par_iter_mut();
        it
    }};
}

macro_rules! par_reduce_sum {
    ($identity: expr, $t: expr) => {{
        #[cfg(not(feature = "parallel"))]
        let res = $t.fold($identity, |a, b| a + b);
        #[cfg(feature = "parallel")]
        let res = $t.reduce(|| $identity, |a, b| a + b);
        res
    }};
}

pub mod counters;
pub mod coupling;
pub mod geometry;
pub mod helper;
pub mod integrations;
pub mod kernel;
mod liquid_world;
pub mod object;
#[cfg(feature = "sampling")]
pub mod sampling;
pub mod solver;
mod timestep_manager;
pub(crate) mod z_order;

pub use crate::liquid_world::LiquidWorld;
pub use crate::timestep_manager::TimestepManager;

/// Compilation flags dependent aliases for mathematical types.
#[cfg(feature = "dim3")]
pub mod math {
    use na::{
        Isometry3, Matrix3, Matrix6, Matrix6xX, MatrixView6xX, MatrixViewMut6xX, Point3, Rotation3,
        Translation3, UnitQuaternion, Vector3, Vector6, U3, U6,
    };

    /// The maximum number of possible rotations and translations of a rigid body.
    pub const SPATIAL_DIM: usize = 6;
    /// The maximum number of possible rotations of a rigid body.
    pub const ANGULAR_DIM: usize = 3;
    /// The maximum number of possible translations of a rigid body.
    pub const DIM: usize = 3;

    /// The scalar type.
    pub type Real = f32;

    /// The dimension of the ambient space.
    pub type Dim = U3;

    /// The dimension of a spatial vector.
    pub type SpatialDim = U6;

    /// The dimension of the rotations.
    pub type AngularDim = U3;

    /// The point type.
    pub type Point<Real> = Point3<Real>;

    /// The angular vector type.
    pub type AngularVector<Real> = Vector3<Real>;

    /// The vector type.
    pub type Vector<Real> = Vector3<Real>;

    /// The vector type with dimension `SpatialDim × 1`.
    pub type SpatialVector<Real> = Vector6<Real>;

    /// The orientation type.
    pub type Orientation<Real> = Vector3<Real>;

    /// The transformation matrix type.
    pub type Isometry<Real> = Isometry3<Real>;

    /// The rotation type.
    pub type Rotation<Real> = UnitQuaternion<Real>;

    /// The rotation matrix type.
    pub type RotationMatrix<Real> = Rotation3<Real>;

    /// The translation type.
    pub type Translation<Real> = Translation3<Real>;

    /// The inertia tensor type.
    pub type AngularInertia<Real> = Matrix3<Real>;

    /// The inertia matrix type.
    pub type InertiaMatrix<Real> = Matrix6<Real>;

    /// Square matrix with dimension `Dim × Dim`.
    pub type Matrix<Real> = Matrix3<Real>;

    /// Square matrix with dimension `SpatialDim × SpatialDim`.
    pub type SpatialMatrix<Real> = Matrix6<Real>;

    /// The type of a constraint jacobian in twist coordinates.
    pub type Jacobian<Real> = Matrix6xX<Real>;

    /// The type of a slice of the constraint jacobian in twist coordinates.
    pub type JacobianSlice<'a, Real> = MatrixView6xX<'a, Real>;

    /// The type of a mutable slice of the constraint jacobian in twist coordinates.
    pub type JacobianSliceMut<'a, Real> = MatrixViewMut6xX<'a, Real>;

    /// The cross-product matrix for the given vector.
    pub fn gcross_matrix(v: &Vector<Real>) -> Matrix<Real> {
        v.cross_matrix()
    }
}

/// Compilation flags dependent aliases for mathematical types.
#[cfg(feature = "dim2")]
pub mod math {
    use na::{
        Isometry2, Matrix1, Matrix2, Matrix3, Matrix6xX, MatrixView3xX, MatrixViewMut3xX, Point2,
        Rotation2, RowVector2, Translation2, UnitComplex, Vector1, Vector2, Vector3, U1, U2, U3,
    };

    /// The maximum number of possible rotations and translations of a rigid body.
    pub const SPATIAL_DIM: usize = 3;
    /// The maximum number of possible rotations of a rigid body.
    pub const ANGULAR_DIM: usize = 1;
    /// The maximum number of possible translations of a rigid body.
    pub const DIM: usize = 2;

    /// The scalar type.
    pub type Real = f32;

    /// The dimension of the ambient space.
    pub type Dim = U2;

    /// The dimension of the rotation.
    pub type AngularDim = U1;

    /// The dimension of a spatial vector.
    pub type SpatialDim = U3;

    /// The point type.
    pub type Point<Real> = Point2<Real>;

    /// The vector type with dimension `SpatialDim × 1`.
    pub type SpatialVector<Real> = Vector3<Real>;

    /// The angular vector type.
    pub type AngularVector<Real> = Vector1<Real>;

    /// The vector type.
    pub type Vector<Real> = Vector2<Real>;

    /// The orientation type.
    pub type Orientation<Real> = Vector1<Real>;

    /// The transformation matrix type.
    pub type Isometry<Real> = Isometry2<Real>;

    /// The rotation type.
    pub type Rotation<Real> = UnitComplex<Real>;

    /// The rotation matrix type.
    pub type RotationMatrix<Real> = Rotation2<Real>;

    /// The translation type.
    pub type Translation<Real> = Translation2<Real>;

    /// The inertia tensor type.
    pub type AngularInertia<Real> = Matrix1<Real>;

    /// The inertia matrix type.
    pub type InertiaMatrix<Real> = Matrix3<Real>;

    /// Square matrix with dimension `Dim × Dim`.
    pub type Matrix<Real> = Matrix2<Real>;

    /// Square matrix with dimension `SpatialDim × SpatialDim`.
    pub type SpatialMatrix<Real> = Matrix3<Real>;

    /// The type of a constraint jacobian in twist coordinates.
    pub type Jacobian<Real> = Matrix6xX<Real>;

    /// The type of a slice of the constraint jacobian in twist coordinates.
    pub type JacobianSlice<'a, Real> = MatrixView3xX<'a, Real>;

    /// The type of a mutable slice of the constraint jacobian in twist coordinates.
    pub type JacobianSliceMut<'a, Real> = MatrixViewMut3xX<'a, Real>;

    /// The cross-product matrix for the given vector, generalized in 2D.
    pub fn gcross_matrix(v: &Vector<Real>) -> RowVector2<Real> {
        RowVector2::new(-v.y, v.x)
    }
}
