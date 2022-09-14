use std::collections::HashMap;
use nalgebra::{DMatrix, RowDVector};
pub use nmi::*;
use crate::callback::EvalData;
use crate::params::thin::ThinParams;
use crate::state::GlobalState;


mod nmi;
mod ic;


pub trait Metric<P: ThinParams> {
    fn compute(
        &mut self,
        i: usize,
        data: &EvalData,
        params: &P,
        metrics: &mut HashMap<String, f64>,
    );
}