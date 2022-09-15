use std::fs::File;
use nalgebra::{DMatrix, DVector, Dynamic, Matrix, RowDVector, Storage};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use plotters::prelude::*;
use clusters::callback::{EvalData, MonitoringCallback};
use clusters::model::Model;
use clusters::params::options::{FitOptions, ModelOptions};
use clusters::stats::NIW;

fn main() {
    let mut f = File::open("examples/data/x.bin").unwrap();
    let x: DMatrix<f64> = deserialize_from(&mut f).unwrap();
    let mut f = File::open("examples/data/y.bin").unwrap();
    let y: RowDVector<usize> = deserialize_from(&mut f).unwrap();

    let dim = x.nrows();
    let mut model_options = ModelOptions::<NIW>::default(dim);
    model_options.alpha = 100.0;
    model_options.outlier = None;
    let mut fit_options = FitOptions::default();
    fit_options.init_clusters = 10;
    // fit_options.iters = 20;
    // fit_options.iters = 40;
    fit_options.workers = 10;

    let mut model = Model::from_options(model_options);
    let callback = MonitoringCallback::from_data(
        EvalData::from_sample(&x, Some(&y), 1000)
    );
    // callback.add_metric(NMI);
    // callback.set_verbose(true);

    model.fit(
        x.clone_owned(),
        &fit_options,
        Some(callback),
    );
}