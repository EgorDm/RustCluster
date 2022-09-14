use std::collections::HashMap;
use std::time::Instant;
use itertools::Itertools;
use nalgebra::{DMatrix, RowDVector};
use rand::prelude::*;
use crate::metrics::{Metric};
use crate::params::thin::ThinParams;
use crate::utils::reservoir_sampling;

pub trait Callback<P: ThinParams> {
    fn before_step(&mut self, _i: usize) {}

    fn during_step(&mut self, _i: usize, _params: &P) {}

    fn after_step(&mut self, _i: usize) {}
}

pub struct EvalData {
    pub points: DMatrix<f64>,
    pub labels: Option<RowDVector<usize>>,
}

pub struct MonitoringCallback<P: ThinParams> {
    data: EvalData,
    metrics: Vec<Box<dyn Metric<P>>>,
    callbacks: Vec<Box<dyn Callback<P>>>,
    measures: HashMap<String, f64>,
    step_started: Instant,
    verbose: bool,
}

impl<P: ThinParams> MonitoringCallback<P> {
    pub fn from_data(
        points: &DMatrix<f64>,
        labels: Option<&RowDVector<usize>>,
        max_points: usize,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut indices = vec![0; max_points];
        let n_points = reservoir_sampling(&mut rng, 0..points.ncols(), &mut indices);
        let points = points.select_columns(&indices[..n_points]);
        let labels = labels.map(|labels| labels.select_columns(&indices[..n_points]));

        Self {
            data: EvalData {
                points,
                labels,
            },
            metrics: vec![],
            callbacks: vec![],
            measures: HashMap::new(),
            step_started: Instant::now(),
            verbose: false,
        }
    }


    pub fn add_metric(&mut self, metric: impl Metric<P> + 'static) {
        self.metrics.push(Box::new(metric));
    }

    pub fn add_callback(&mut self, callback: impl Callback<P> + 'static) {
        self.callbacks.push(Box::new(callback));
    }

    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }
}

impl<P: ThinParams> Callback<P> for MonitoringCallback<P> {
    fn before_step(&mut self, i: usize) {
        self.measures.clear();
        for callback in &mut self.callbacks {
            callback.before_step(i);
        }
        self.step_started = Instant::now();
    }

    fn during_step(&mut self, i: usize, params: &P) {
        self.measures.insert("k".to_string(), params.n_clusters() as f64);
        for metric in &mut self.metrics {
            metric.compute(i, &self.data, params, &mut self.measures);
        }
        for callback in &mut self.callbacks {
            callback.during_step(i, params);
        }
    }

    fn after_step(&mut self, i: usize) {
        for callback in &mut self.callbacks {
            callback.after_step(i);
        }
        if self.verbose {
            let elapsed = self.step_started.elapsed();
            let measures = self.measures.iter().map(|(k, v)| format!("{}={:.4}", k, v)).join(", ");
            println!("Run iteration {} in {:.2?}; {}", i, elapsed, measures);
        }
    }
}
