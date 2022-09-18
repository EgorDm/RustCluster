use image::Rgb;
use itertools::izip;
use nalgebra::DMatrix;
use palette::{FromColor, Lab, Pixel, Srgb};
use mixturs::{AIC, FitOptions, Model, ModelOptions, MonitoringCallback, NIW};
use mixturs::callback::EvalData;
use mixturs::params::ThinParams;
use mixturs::stats::{Covariance, NIWParams};

fn main() {
    let mut input_image = image::open("examples/data/lenna.png").unwrap().into_rgb32f();
    let (width, height) = input_image.dimensions();
    let input_buffer = Srgb::from_raw_slice(input_image.as_raw());

    let mut data = DMatrix::<f64>::zeros(5, (width * height) as usize);
    for (i, src, mut dst) in izip!(0.., input_buffer, data.column_iter_mut()) {
        dst[0] = src.red as f64;
        dst[1] = src.green as f64;
        dst[2] = src.blue as f64;
        dst[3] = (i % width) as f64;
        dst[4] = (i / width) as f64;
    }

    let rgb_prior_multiplier = 30.0;
    let xy_prior_multiplier = 30.0;

    let data_mean = data.column_mean();
    let mut data_cov = data.column_cov();
    data_cov.slice_range_mut(3..5, 0..3).apply(|v| *v = 0.0);
    data_cov.slice_range_mut(0..3, 3..5).apply(|v| *v = 0.0);

    data_cov.slice_range_mut(0..3, 0..3).apply(|v| *v *= rgb_prior_multiplier);
    data_cov.slice_range_mut(3..5, 3..5).apply(|v| *v *= xy_prior_multiplier);

    let prior = NIWParams::new(1.0, data_mean, 8.0, data_cov);

    let dim = data.nrows();
    let mut model_options = ModelOptions::<NIW>::default(dim);
    model_options.alpha = 100.0;
    model_options.outlier = None;
    model_options.data_dist = prior;

    let mut fit_options = FitOptions::default();
    fit_options.init_clusters = 1;
    fit_options.workers = -1;
    fit_options.iters = 1000;

    let mut model = Model::from_options(model_options);
    let mut callback = MonitoringCallback::from_data(
        EvalData::from_sample(&data, None, 1000)
    );
    // callback.add_metric(AIC);
    callback.set_verbose(true);

    model.fit(
        data.clone_owned(),
        &fit_options,
        Some(callback),
    );

    let (_, labels) = model.predict(data);

    let params = model.params();
    let mut colors = Vec::new();
    for i in 0..params.n_clusters() {
        let mu = params.cluster_dist(i).mu();
        colors.push(Rgb::<u8>([
            (mu[0] * 255.0).round().min(255.0).max(0.0) as u8,
            (mu[1] * 255.0).round().min(255.0).max(0.0) as u8,
            (mu[2] * 255.0).round().min(255.0).max(0.0) as u8,
        ]));
    }

    let mut output_image = image::RgbImage::new(width, height);
    for (i, pixel) in output_image.pixels_mut().enumerate() {
        pixel.clone_from(&colors[labels[i]]);
    }

    output_image.save("examples/data/lenna_seg.png").unwrap();

    let test = 0;
}