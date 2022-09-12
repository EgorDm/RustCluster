use std::collections::HashMap;
use nalgebra::{DMatrix, RowDVector};
pub use nmi::*;
use crate::params::thin::ThinParams;
use crate::state::GlobalState;


mod nmi;
mod ic;


pub trait Metric<P: ThinParams> {
    fn compute(
        &mut self,
        data: &EvaluationData,
        params: &P,
        metrics: &mut HashMap<String, f64>,
    );
}

pub struct ComposedMetric<P: ThinParams> {
    metrics: Vec<Box<dyn Metric<P>>>,
}

impl<P: ThinParams> ComposedMetric<P> {
    pub fn new(metrics: Vec<Box<dyn Metric<P>>>) -> Self {
        Self { metrics }
    }

    pub fn add(&mut self, metric: impl Metric<P> + 'static) {
        self.metrics.push(Box::new(metric));
    }
}

impl<P: ThinParams> Metric<P> for ComposedMetric<P> {
    fn compute(&mut self, data: &EvaluationData, params: &P, metrics: &mut HashMap<String, f64>) {
        for metric in &mut self.metrics {
            metric.compute(data, params, metrics);
        }
    }
}

pub struct EvaluationData {
    pub points: DMatrix<f64>,
    pub labels: Option<RowDVector<usize>>,
}