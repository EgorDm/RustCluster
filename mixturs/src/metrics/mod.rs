use std::collections::HashMap;
pub use nmi::*;
pub use ic::*;
use crate::callback::EvalData;
use crate::params::thin::ThinParams;


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