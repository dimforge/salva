/*!
**Salva** is a 2 and 3-dimensional particle-based fluid simulation engine for games and animations.
It uses [nalgebra](https://nalgebra.org) for vector/matrix math and can optionally interface with
[nphysics](https://nphysics.org) for two-way coupling with rigid bodies, multibodies, and deformable bodies.
2D and 3D implementations both share (mostly) the same code!


Examples are available in the `examples2d` and `examples3d` directories.  Because those demos are based on
WASM and WebGl 1.0 they should work on most modern browsers. Feel free to ask for help
and discuss features on the official [user forum](https://discourse.nphysics.org).

## Why the name Salva?

The name of this library is inspired from the famous surrealist artist `Salvador Dalì`. The logo of `Salva`
 is inspired from its renown painting [The Persistence of Memory](https://en.wikipedia.org/wiki/The_Persistence_of_Memory).

## Features
- PBF pressure resolution.
- XSPH viscosity.
- Multiphase fluids.
- Optional two-way coupling with bodies from **nphysics**.
- WASM support
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
#[cfg(all(feature = "dim2", feature = "nphysics"))]
extern crate ncollide2d as ncollide;
#[cfg(all(feature = "dim3", feature = "nphysics"))]
extern crate ncollide3d as ncollide;
#[cfg(all(feature = "dim2", feature = "nphysics"))]
extern crate nphysics2d as nphysics;
#[cfg(all(feature = "dim3", feature = "nphysics"))]
extern crate nphysics3d as nphysics;

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

#[cfg(feature = "nphysics")]
pub mod coupling;
pub mod geometry;
pub mod kernel;
mod liquid_world;
pub mod object;
#[cfg(feature = "sampling")]
pub mod sampling;
pub mod solver;
mod timestep_manager;

pub use crate::liquid_world::LiquidWorld;
pub use crate::timestep_manager::TimestepManager;

/// Compilation flags dependent aliases for mathematical types.
#[cfg(feature = "dim3")]
pub mod math {
    use na::{
        Dynamic, Isometry3, Matrix3, Matrix6, MatrixMN, MatrixSlice6xX, MatrixSliceMut6xX, Point3,
        RealField, Rotation3, Translation3, UnitQuaternion, Vector3, Vector6, U3, U6,
    };

    /// The maximum number of possible rotations and translations of a rigid body.
    pub const SPATIAL_DIM: usize = 6;
    /// The maximum number of possible rotations of a rigid body.
    pub const ANGULAR_DIM: usize = 3;
    /// The maximum number of possible translations of a rigid body.
    pub const DIM: usize = 3;

    /// The dimension of the ambient space.
    pub type Dim = U3;

    /// The dimension of a spatial vector.
    pub type SpatialDim = U6;

    /// The dimension of the rotations.
    pub type AngularDim = U3;

    /// The point type.
    pub type Point<N> = Point3<N>;

    /// The angular vector type.
    pub type AngularVector<N> = Vector3<N>;

    /// The vector type.
    pub type Vector<N> = Vector3<N>;

    /// The vector type with dimension `SpatialDim × 1`.
    pub type SpatialVector<N> = Vector6<N>;

    /// The orientation type.
    pub type Orientation<N> = Vector3<N>;

    /// The transformation matrix type.
    pub type Isometry<N> = Isometry3<N>;

    /// The rotation type.
    pub type Rotation<N> = UnitQuaternion<N>;

    /// The rotation matrix type.
    pub type RotationMatrix<N> = Rotation3<N>;

    /// The translation type.
    pub type Translation<N> = Translation3<N>;

    /// The inertia tensor type.
    pub type AngularInertia<N> = Matrix3<N>;

    /// The inertia matrix type.
    pub type InertiaMatrix<N> = Matrix6<N>;

    /// Square matrix with dimension `Dim × Dim`.
    pub type Matrix<N> = Matrix3<N>;

    /// Square matrix with dimension `SpatialDim × SpatialDim`.
    pub type SpatialMatrix<N> = Matrix6<N>;

    /// The type of a constraint jacobian in twist coordinates.
    pub type Jacobian<N> = MatrixMN<N, U6, Dynamic>;

    /// The type of a slice of the constraint jacobian in twist coordinates.
    pub type JacobianSlice<'a, N> = MatrixSlice6xX<'a, N>;

    /// The type of a mutable slice of the constraint jacobian in twist coordinates.
    pub type JacobianSliceMut<'a, N> = MatrixSliceMut6xX<'a, N>;

    /// The cross-product matrix for the given vector.
    pub fn gcross_matrix<N: RealField>(v: &Vector<N>) -> Matrix<N> {
        v.cross_matrix()
    }
}

/// Compilation flags dependent aliases for mathematical types.
#[cfg(feature = "dim2")]
pub mod math {
    use na::{
        Dynamic, Isometry2, Matrix1, Matrix2, Matrix3, MatrixMN, MatrixSlice3xX, MatrixSliceMut3xX,
        Point2, RealField, Rotation2, RowVector2, Translation2, UnitComplex, Vector1, Vector2,
        Vector3, U1, U2, U3,
    };

    /// The maximum number of possible rotations and translations of a rigid body.
    pub const SPATIAL_DIM: usize = 3;
    /// The maximum number of possible rotations of a rigid body.
    pub const ANGULAR_DIM: usize = 1;
    /// The maximum number of possible translations of a rigid body.
    pub const DIM: usize = 2;

    /// The dimension of the ambient space.
    pub type Dim = U2;

    /// The dimension of the rotation.
    pub type AngularDim = U1;

    /// The dimension of a spatial vector.
    pub type SpatialDim = U3;

    /// The point type.
    pub type Point<N> = Point2<N>;

    /// The vector type with dimension `SpatialDim × 1`.
    pub type SpatialVector<N> = Vector3<N>;

    /// The angular vector type.
    pub type AngularVector<N> = Vector1<N>;

    /// The vector type.
    pub type Vector<N> = Vector2<N>;

    /// The orientation type.
    pub type Orientation<N> = Vector1<N>;

    /// The transformation matrix type.
    pub type Isometry<N> = Isometry2<N>;

    /// The rotation type.
    pub type Rotation<N> = UnitComplex<N>;

    /// The rotation matrix type.
    pub type RotationMatrix<N> = Rotation2<N>;

    /// The translation type.
    pub type Translation<N> = Translation2<N>;

    /// The inertia tensor type.
    pub type AngularInertia<N> = Matrix1<N>;

    /// The inertia matrix type.
    pub type InertiaMatrix<N> = Matrix3<N>;

    /// Square matrix with dimension `Dim × Dim`.
    pub type Matrix<N> = Matrix2<N>;

    /// Square matrix with dimension `SpatialDim × SpatialDim`.
    pub type SpatialMatrix<N> = Matrix3<N>;

    /// The type of a constraint jacobian in twist coordinates.
    pub type Jacobian<N> = MatrixMN<N, U3, Dynamic>;

    /// The type of a slice of the constraint jacobian in twist coordinates.
    pub type JacobianSlice<'a, N> = MatrixSlice3xX<'a, N>;

    /// The type of a mutable slice of the constraint jacobian in twist coordinates.
    pub type JacobianSliceMut<'a, N> = MatrixSliceMut3xX<'a, N>;

    /// The cross-product matrix for the given vector, generalized in 2D.
    pub fn gcross_matrix<N: RealField>(v: &Vector<N>) -> RowVector2<N> {
        RowVector2::new(-v.y, v.x)
    }
}
