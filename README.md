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
    <a href="https://travis-ci.org/dimforge/salva">
        <img src="https://travis-ci.org/dimforge/salva.svg?branch=master" alt="Build status">
    </a>
</p>
<p align = "center">
    <strong>
        <a href="https://salva.rs">Users guide</a> | <a href="https://docs.rs/salva2d/latest/salva2d">2D Documentation</a> | <a href="https://docs.rs/salva3d/latest/salva3d">3D Documentation</a> | <a href="https://discord.gg/vt9DJSW">Discord</a>
    </strong>
</p>

-----

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
