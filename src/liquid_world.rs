use crate::boundary::Boundary;
use crate::fluid::Fluid;
use crate::math::Vector;
use crate::solver::PBFSolver;
use na::RealField;

pub struct LiquidWorld<N: RealField> {
    fluids: Vec<Fluid<N>>,
    boundaries: Vec<Boundary<N>>,
    solver: PBFSolver<N>,
}

impl<N: RealField> LiquidWorld<N> {
    pub fn new(h: N) -> Self {
        Self {
            fluids: Vec::new(),
            boundaries: Vec::new(),
            solver: PBFSolver::new(h),
        }
    }
    pub fn step(&mut self, dt: N, gravity: &Vector<N>) {
        let mut fluid_contacts = Vec::new();
        let mut boundary_contacts = Vec::new();

        self.solver.step(
            dt,
            gravity,
            &mut fluid_contacts,
            &mut boundary_contacts,
            &mut self.fluids,
            &self.boundaries,
        );
    }
}
