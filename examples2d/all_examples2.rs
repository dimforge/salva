#![allow(dead_code)]

extern crate nalgebra as na;

use inflector::Inflector;

use rapier_testbed2d::{Testbed, TestbedApp};

mod basic2;
mod custom_forces2;
mod elasticity2;
mod surface_tension2;

fn demo_name_from_command_line() -> Option<String> {
    let mut args = std::env::args();

    while let Some(arg) = args.next() {
        if &arg[..] == "--example" {
            return args.next();
        }
    }

    None
}

#[cfg(any(target_arch = "wasm32", target_arch = "asmjs"))]
fn demo_name_from_url() -> Option<String> {
    let window = stdweb::web::window();
    let hash = window.location()?.search().ok()?;
    if !hash.is_empty() {
        Some(hash[1..].to_string())
    } else {
        None
    }
}

#[cfg(not(any(target_arch = "wasm32", target_arch = "asmjs")))]
fn demo_name_from_url() -> Option<String> {
    None
}

fn main() {
    let demo = demo_name_from_command_line()
        .or_else(|| demo_name_from_url())
        .unwrap_or(String::new())
        .to_camel_case();

    let mut builders: Vec<(_, fn(&mut Testbed))> = vec![
        ("Basic", basic2::init_world),
        ("Custom forces", custom_forces2::init_world),
        ("Elasticity", elasticity2::init_world),
        ("Surface tension", surface_tension2::init_world),
    ];
    builders.sort_by_key(|builder| builder.0);

    let i = builders
        .iter()
        .position(|builder| builder.0.to_camel_case().as_str() == demo.as_str())
        .unwrap_or(0);
    let testbed = TestbedApp::from_builders(i, builders);

    testbed.run()
}
