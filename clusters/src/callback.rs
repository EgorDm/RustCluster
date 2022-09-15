use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use itertools::Itertools;
use nalgebra::{DMatrix, Dynamic, Matrix, RowDVector, Storage};
use plotters::prelude::*;
use rand::prelude::*;
use crate::metrics::{Metric};
use crate::params::{MixtureParams, SuperMixtureParams};
use crate::params::thin::ThinParams;
use crate::plotting::{axes_range_from_points, Cluster2D, init_axes2d};
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

impl EvalData {
    pub fn from_sample(
        points: &DMatrix<f64>,
        labels: Option<&RowDVector<usize>>,
        max_points: usize,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut indices = vec![0; max_points];
        let n_points = reservoir_sampling(&mut rng, 0..points.ncols(), &mut indices);
        let points = points.select_columns(&indices[..n_points]);
        let labels = labels.map(|labels| labels.select_columns(&indices[..n_points]));

        Self { points, labels }
    }
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
    pub fn from_data(data: EvalData) -> Self {
        Self {
            data,
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


pub struct PlotCallback {
    freq: usize,
    data: EvalData,
    path: PathBuf,
}

impl PlotCallback {
    pub fn new(freq: usize, path: PathBuf, data: EvalData) -> Self {
        Self { freq, data, path }
    }
}

impl<P: ThinParams> Callback<P> for PlotCallback {
    fn during_step(&mut self, i: usize, params: &P) {
        if i % self.freq != 0 { return; }

        let path = self.path.join(format!("step_{:04}.png", i));
        if let Some(labels) = &self.data.labels {
            plot(&path, &self.data.points, labels.as_slice(), &SuperMixtureParams(params));
        }
    }
}


fn plot<S: Storage<f64, Dynamic, Dynamic>>(
    path: &PathBuf,
    points: &Matrix<f64, Dynamic, Dynamic, S>,
    labels: &[usize],
    params: &impl MixtureParams
) {
    let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
    let (range_x, range_y) = axes_range_from_points(points);
    let mut plot_ctx = init_axes2d((range_x, range_y), &root);

    plot_ctx.draw_series(
        points
            .column_iter()
            .zip(labels.iter())
            .map(|(row, label)|
                Circle::new((row[0], row[1]), 2, Palette99::pick(*label).mix(0.9).filled())),
    ).unwrap();

    for k in 0..params.n_clusters() {
        let cluster = params.dist(k);

        plot_ctx.draw_series(
            Cluster2D::from_mat(
                cluster.mu(), cluster.cov(),
                50,
                Palette99::pick(k).filled(),
                Palette99::pick(k).stroke_width(2),
            )
        ).unwrap();
    }

    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", path.to_str().unwrap());
}

