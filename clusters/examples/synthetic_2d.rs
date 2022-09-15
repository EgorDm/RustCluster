use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use clusters::{AIC, FitOptions, Model, ModelOptions, MonitoringCallback, NIW, NMI};
use clusters::callback::{EvalData};
use clusters::plotting::{PlotCallback};

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
        EvalData::from_sample(&x, Some(&y), 1000)
    );
    callback.add_metric(NMI);
    callback.add_metric(AIC);
    callback.add_callback(PlotCallback::new(
        3,
        "examples/data/plot/synthetic_2d".into(),
        EvalData::from_sample(&x, Some(&y), 1000)
    ));
    callback.set_verbose(true);

    model.fit(
        x.clone_owned(),
        &fit_options,
        Some(callback),
    );
}