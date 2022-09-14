use std::f64::consts::PI;
use std::io::Read;
use std::time::Instant;
use nalgebra::{DMatrix, DVector, Dynamic, Matrix, Storage};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use plotters::prelude::*;
use rand::prelude::{SmallRng, StdRng};
use rand::{Rng, SeedableRng};
use clusters::callback::MonitoringCallback;
use clusters::state::{GlobalWorker, GlobalState, LocalState, LocalWorker};
use clusters::metrics::{NMI, normalized_mutual_info_score};
use clusters::callback::EvalData;
use clusters::model::Model;
use clusters::params::clusters::SuperClusterParams;
use clusters::params::options::{FitOptions, ModelOptions};
use clusters::plotting::{axes_range_from_points, Cluster2D, init_axes2d};
use clusters::stats::{FromData, NIW, NIWStats, SufficientStats};

fn plot<S: Storage<f64, Dynamic, Dynamic>>(
    path: &str,
    points: &Matrix<f64, Dynamic, Dynamic, S>, labels: &[usize], clusters: &[SuperClusterParams<NIW>]
) {
    let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
    let (mut range_x, mut range_y) = axes_range_from_points(points);
    // let (mut range_x, mut range_y) = (-50.0..50.0, -50.0..50.0);
    let mut plot_ctx = init_axes2d((range_x, range_y), &root);

    plot_ctx.draw_series(
        points
            .column_iter()
            .zip(labels.iter())
            .map(|(row, label)|
                Circle::new((row[0], row[1]), 2, Palette99::pick(*label).mix(0.9).filled())),
    ).unwrap();

    for (k, cluster) in clusters.iter().enumerate() {
        plot_ctx.draw_series(
            Cluster2D::from_mat(
                cluster.prim.dist.mu(), cluster.prim.dist.cov(),
                50,
                Palette99::pick(k).filled(),
                Palette99::pick(k).stroke_width(2),
            )
        ).unwrap();
    }

    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", path);
}


fn main() {
    println!("Hello, world!");

    let x_data: Array2<f64> = read_npy("examples/data/x.npy").unwrap();
    let x = DMatrix::from_row_slice(x_data.nrows(), x_data.ncols(), &x_data.as_slice().unwrap()).transpose();
    let y_data: Array1<i64> = read_npy("examples/data/y.npy").unwrap();
    let y = DVector::from_row_slice(&y_data.as_slice().unwrap());
    let y = y.map(|x| x as usize).into_owned().transpose();

    let dim = x.nrows();

    let mut model_options = ModelOptions::<NIW>::default(dim);
    model_options.alpha = 100.0;
    model_options.outlier = None;
    let mut fit_options = FitOptions::default();
    fit_options.init_clusters = 1;
    // fit_options.init_clusters = 10;
    // fit_options.iters = 20;
    // fit_options.iters = 40;

    let mut model = Model::from_options(model_options);
    let mut callback = MonitoringCallback::from_data(
        &x, Some(&y), 1000,
    );
    callback.add_metric(NMI);
    callback.set_verbose(true);

    model.fit(
        x.clone_owned(),
        &fit_options,
        Some(callback),
    );
}