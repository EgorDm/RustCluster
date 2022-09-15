use std::fs::File;
use nalgebra::{DMatrix, RowDVector};
use clusters::{AIC, FitOptions, Model, ModelOptions, MonitoringCallback, NIW, NMI};
use clusters::callback::{EvalData};
use clusters::plotting::{PlotCallback};
use bincode::{deserialize_from};


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