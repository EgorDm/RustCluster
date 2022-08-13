use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use plotters::prelude::*;
use dpmmsc::plotting::{Cluster2D, init_axes2d};
use dpmmsc::priors::{GaussianPrior, NIW, NIWParams};
use statrs::distribution::{InverseWishart, MultivariateNormal};


fn main() {
    // plot_points_dist();
    plot_niw_dist();
}


fn plot_points_dist() {
    const PATH: &str = "examples/data/plot/plot_sampling_points.png";

    let mu = DVector::from_row_slice(&[0.0, 0.0]);
    let cov = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);

    let dist = MultivariateNormal::new(mu.clone(), cov.clone()).unwrap();
    let mut rng = StdRng::seed_from_u64(0);



    let root = BitMapBackend::new(PATH, (1024, 768)).into_drawing_area();
    let mut plot_ctx = init_axes2d((-4.0..4.0, -4.0..4.0), &root);

    plot_ctx.draw_series(
        Cluster2D::from_mat(
            &mu, &cov,
            100,
            Palette99::pick(1).filled(),
            Palette99::pick(1).stroke_width(2),
        )
    ).unwrap();

    plot_ctx.draw_series(
        dist.sample_iter(&mut rng)
            .take(1000)
            .map(|x| Circle::new((x[0], x[1]), 2, Palette99::pick(3).mix(0.8).filled()))
    ).unwrap();

    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", PATH);
}

fn plot_niw_dist() {
    const PATH: &str = "examples/data/plot/plot_sampling_niw.png";

    let mu = DVector::from_row_slice(&[0.0, 0.0]);
    let cov = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);

    let dist_params = NIWParams::new(1.0, mu.clone(), 5.0, cov.clone());
    // let mut rng = StdRng::seed_from_u64(0);
    let mut rng = StdRng::seed_from_u64(1);
    let mut rng = StdRng::seed_from_u64(2);

    let root = BitMapBackend::new(PATH, (1024, 768)).into_drawing_area();
    let mut plot_ctx = init_axes2d((-10.0..10.0, -10.0..10.0), &root);

    plot_ctx.draw_series(
        Cluster2D::from_mat(
            &mu, &cov,
            20,
            Palette99::pick(1).filled(),
            Palette99::pick(1).stroke_width(4),
        )
    ).unwrap();

    for i in 0..5 {
        let dist = NIW::sample(&dist_params, &mut rng);

        plot_ctx.draw_series(
            Cluster2D::from_mat(
                dist.mu(), dist.cov(),
                50,
                Palette99::pick(i+3).filled(),
                Palette99::pick(i+3).stroke_width(2),
            )
        ).unwrap();
    }

    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", PATH);
}
