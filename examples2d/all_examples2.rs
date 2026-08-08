#![allow(dead_code)]
#![allow(clippy::type_complexity)]

use rapier_testbed2d::{ExampleEntry, TestbedViewer};
use std::future::Future;
use std::pin::Pin;

mod basic2;
mod custom_forces2;
mod elasticity2;
mod layers2;
mod surface_tension2;

/// A registered example: a fn pointer running the example's owned loop.
/// (A non-capturing closure coerces to this higher-ranked fn pointer.)
type ExampleFn =
    for<'a> fn(&'a mut TestbedViewer) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + 'a>>;

/// `(group, name, run-fn)` -> `(ExampleEntry, ExampleFn)`.
macro_rules! examples {
    ($($group:expr, $name:expr, $run:path);* $(;)?) => {
        vec![ $( (ExampleEntry::new($group, $name), (|v| Box::pin($run(v))) as ExampleFn) ),* ]
    };
}

#[kiss3d::main]
pub async fn main() {
    const FLUIDS: &str = "Fluids";

    let examples: Vec<(ExampleEntry, ExampleFn)> = examples![
        FLUIDS, "Basic", basic2::run;
        FLUIDS, "Custom forces", custom_forces2::run;
        FLUIDS, "Elasticity", elasticity2::run;
        FLUIDS, "Layers", layers2::run;
        FLUIDS, "Surface tension", surface_tension2::run;
    ];

    let (entries, run_fns): (Vec<_>, Vec<ExampleFn>) = examples.into_iter().unzip();
    let mut viewer = TestbedViewer::new(entries).await;

    // The example owns its physics state and render loop; this outer loop just
    // (re)dispatches the example the UI has selected.
    loop {
        viewer.clear_scene();
        let idx = viewer.selected();
        if let Err(e) = run_fns[idx](&mut viewer).await {
            eprintln!("example #{idx} failed: {e:?}");
        }
        if viewer.quitting() {
            break;
        }
    }
}
