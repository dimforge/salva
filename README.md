<p align="center">
  <img src="https://salva.rs/img/logo_salva_full.svg" alt="crates.io">
</p>
<p align="center">
    <a href="https://discord.gg/vt9DJSW">
        <img src="https://img.shields.io/discord/507548572338880513.svg?logo=discord&colorB=7289DA">
    </a>
    <a href="https://crates.io/crates/salva2d">
         <img src="https://img.shields.io/crates/v/salva2d.svg?style=flat-square&label=crates.io%20(salva2d)" alt="crates.io (salva2d)">
    </a>
    <a href="https://crates.io/crates/salva3d">
         <img src="https://img.shields.io/crates/v/salva3d.svg?style=flat-square&label=crates.io%20(salva3d)" alt="crates.io (salva3d)">
    </a>
    <a href="https://travis-ci.org/rustsim/salva">
        <img src="https://travis-ci.org/rustsim/salva.svg?branch=master" alt="Build status">
    </a>
</p>
<p align = "center">
    <strong>
        <a href="https://salva.rs">Users guide</a> | <a href="https://salva.rs/rustdoc/salva2d/index.html">2D Documentation</a> | <a href="https://salva.rs/rustdoc/salva3d/index.html">3D Documentation</a> | <a href="https://discourse.nphysics.org">Forum</a>
    </strong>
</p>

-----

<p align = "center">
  <i>Click one of those buttons if you wish to donate to support the development of</i> <b>salva</b>:
</p>

<p align = "center">
<a href="https://www.patreon.com/bePatron?u=7111380" ><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron!" /></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://liberapay.com/sebcrozet/donate"><img alt="Donate using Liberapay" src="https://liberapay.com/assets/widgets/donate.svg"></a>
</p>

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
