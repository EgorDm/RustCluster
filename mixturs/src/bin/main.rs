use clap::Parser;
use image::codecs::jpeg::JpegEncoder;
use image::{ColorType, ImageEncoder, Rgb};
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use itertools::izip;
use nalgebra::DMatrix;
use palette::{Srgb, Pixel};
use mixturs::{AIC, BIC, FitOptions, Model, ModelOptions, MonitoringCallback, NIW, NMI};
use mixturs::callback::EvalData;
use mixturs::params::ThinParams;
use mixturs::stats::{Covariance, NIWParams};

#[derive(Debug, Parser)]
#[clap(author, version, about, long_about = None)]
pub struct Opt {
    /// Input file.
    #[clap(short, long, parse(from_os_str))]
    pub input: std::path::PathBuf,

    /// Output file, defaults to PNG image output.
    #[clap(short, long, parse(from_os_str))]
    pub output: Option<std::path::PathBuf>,

    /// Burnout period for the supercluster log likelihood history (in iterations)
    #[clap(short, short_alias = 'b', default_value_t = 20)]
    pub burnout_period: usize,

    /// Concentration parameter for the Dirichlet process
    #[clap(short, short_alias = 'a', default_value_t = 10.0)]
    pub alpha: f64,

    /// Whether to use outlier removal
    #[clap(long)]
    pub outlier_removal: bool,

    /// Weight of the outlier prior
    #[clap(long, default_value_t = 0.05)]
    pub outlier_weight: f64,

    /// Seed for the random number generator
    #[clap(long, default_value_t = 42)]
    pub seed: u64,

    /// Weight of the outlier prior
    #[clap(long, default_value_t = 30.0)]
    pub rgb_prior_multiplier: f64,

    /// Weight of the outlier prior
    #[clap(long, default_value_t = 1.0)]
    pub xy_prior_multiplier: f64,

    /// Number of initial clusters
    #[clap(short, short_alias = 'k', default_value_t = 1)]
    pub k: usize,

    /// Maximum number of iterations
    #[clap(long, default_value_t = 100)]
    pub iters: usize,

    /// Number of workers (threads) for parallelization (-1 = number of CPUs)
    #[clap(short, short_alias = 't', default_value_t = -1)]
    pub workers: i32,

    /// Print the iteration statistics.
    #[clap(short, long, default_value_t = true)]
    pub verbose: bool,

    /// Save as a JPG or PNG file.
    #[clap(long, default_value = "png")]
    pub format: String,

    /// Add nmi measure to monitoring callback.
    #[clap(long)]
    pub nmi: bool,

    /// Add aic measure to monitoring callback.
    #[clap(long)]
    pub aic: bool,

    /// Add bic measure to monitoring callback.
    #[clap(long)]
    pub bic: bool,
}


fn main() {
    if let Err(e) = try_main() {
        eprintln!("mixturs: {}", e);
        std::process::exit(1);
    }
}

fn try_main() -> Result<(), Box<dyn std::error::Error>> {
    let opt: Opt = Opt::parse();

    let output_image = if let Some(output) = opt.output {
        output
    } else {
        generate_filename(&opt)?.into()
    };


    let input_image = image::open(opt.input)?.into_rgb8();
    let (width, height) = input_image.dimensions();
    let input_buffer = Srgb::from_raw_slice(input_image.as_raw());

    let mut data = DMatrix::<f64>::zeros(5, (width * height) as usize);
    for (i, src, mut dst) in izip!(0.., input_buffer, data.column_iter_mut()) {
        dst[0] = src.red as f64 / 255.0;
        dst[1] = src.green as f64 / 255.0;
        dst[2] = src.blue as f64 / 255.0;
        dst[3] = (i % width) as f64;
        dst[4] = (i / width) as f64;
    }

    let data_mean = data.column_mean();
    let mut data_cov = data.column_cov();
    data_cov.slice_range_mut(3..5, 0..3).apply(|v| *v = 0.0);
    data_cov.slice_range_mut(0..3, 3..5).apply(|v| *v = 0.0);

    data_cov.slice_range_mut(0..3, 0..3).apply(|v| *v *= opt.rgb_prior_multiplier);
    data_cov.slice_range_mut(3..5, 3..5).apply(|v| *v *= opt.xy_prior_multiplier);

    let data_prior = NIWParams::new(1.0, data_mean, 8.0, data_cov);

    let dim = data.nrows();
    let mut model_options = ModelOptions::<NIW>::default(dim);
    model_options.alpha = opt.alpha;
    model_options.data_dist = data_prior;
    model_options.burnout_period = opt.burnout_period;
    if opt.outlier_removal {
        model_options.outlier.as_mut().unwrap().weight = opt.outlier_weight;
    } else {
        model_options.outlier = None;
    };

    let mut fit_options = FitOptions::default();
    fit_options.init_clusters = opt.k;
    fit_options.seed = opt.seed;
    fit_options.workers = opt.workers;
    fit_options.iters = opt.iters;

    let mut model = Model::from_options(model_options);
    let mut callback = MonitoringCallback::from_data(
        EvalData::from_sample(&data, None, 1000)
    );
    if opt.verbose {
        if opt.nmi {
            callback.add_metric(NMI);
        }
        if opt.aic {
            callback.add_metric(AIC);
        }
        if opt.bic {
            callback.add_metric(BIC);
        }
    }
    callback.set_verbose(opt.verbose);

    model.fit(
        data.clone_owned(),
        &fit_options,
        Some(callback),
    );

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

    let (_, labels) = model.predict(data);
    let mut output_buffer = image::RgbImage::new(width, height);
    for (pixel, &label) in izip!(output_buffer.pixels_mut(), labels.into_iter()) {
        pixel.clone_from(&colors[label]);
    }

    save_image(output_image.as_ref(), &output_buffer, width, height)?;
    Ok(())
}

pub fn generate_filename(opt: &Opt) -> Result<String, Box<dyn std::error::Error>> {
    let mut filename = opt
        .input
        .file_stem()
        .ok_or("No file stem")?
        .to_str()
        .ok_or("Could not convert file stem to string")?
        .to_string();

    let format =
        if opt.format.eq_ignore_ascii_case("jpg") || opt.format.eq_ignore_ascii_case("jpeg") {
            "jpg"
        } else {
            opt.format.as_str()
        };

    use std::fmt::Write;
    write!(
        &mut filename,
        "-{k}-{a:02}-seg",
        k = opt.k,
        a = opt.alpha,
    )?;

    write!(&mut filename, ".{format}")?;
    Ok(filename)
}

// Saves image buffer to file.
pub fn save_image(
    output: &std::path::Path,
    imgbuf: &[u8],
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let w = std::io::BufWriter::new(std::fs::File::create(output)?);

    // Save as jpg if it matches the extension
    if let Some(ext) = output.extension() {
        if ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("jpeg") {
            let mut encoder = JpegEncoder::new_with_quality(w, 90);

            if let Err(err) = encoder.encode(imgbuf, width, height, ColorType::Rgb8) {
                eprintln!("simple_clustering: {}", err);
                std::fs::remove_file(output)?;
            }

            return Ok(());
        }
    }

    // Sub filter seemed to result in better filesize compared to Adaptive
    let encoder = PngEncoder::new_with_quality(w, CompressionType::Best, FilterType::Sub);

    // Clean up if file is created but there's a problem writing to it
    if let Err(err) = encoder.write_image(imgbuf, width, height, ColorType::Rgb8) {
        eprintln!("simple_clustering: {}", err);
        std::fs::remove_file(output)?;
    }

    Ok(())
}