mod parallellism;

use nalgebra::{DMatrix, DVector};
use ndarray::AssignElem;
use num_traits::real::Real;
use plotters::prelude::*;
use rand::prelude::{Distribution, StdRng};
use rand::SeedableRng;
use mixturs::plotting::{Cluster2D, init_axes2d};
use mixturs::stats::{ConjugatePrior, NormalConjugatePrior, NIW, NIWParams, NIWStats, SufficientStats, FromData};
use statrs::distribution::{InverseWishart, MultivariateNormal};


fn main() {
    plot_points_dist();
    plot_niw_dist();
    plot_niw_chain();
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
    let mut rng = StdRng::seed_from_u64(42);
    // let mut rng = StdRng::seed_from_u64(1);
    // let mut rng = StdRng::seed_from_u64(2);

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
                Palette99::pick(i + 3).filled(),
                Palette99::pick(i + 3).stroke_width(2),
            )
        ).unwrap();
    }

    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", PATH);
}

fn plot_niw_chain() {
    const PATH: &str = "examples/data/plot/plot_sampling_niw_chain.png";

    let mu = DVector::from_row_slice(&[0.0, 0.0]);
    let cov = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);

    let mut dist_params = NIWParams::new(1.0, mu.clone(), 10.0, cov.clone());
    let mut rng = StdRng::seed_from_u64(42);

    let root = BitMapBackend::new(PATH, (1024, 768)).into_drawing_area();
    let mut plot_ctx = init_axes2d((-5.0..5.0, -5.0..5.0), &root);

    plot_ctx.draw_series(
        Cluster2D::from_mat(
            &mu, &cov,
            20,
            Palette99::pick(1).filled(),
            Palette99::pick(1).stroke_width(4),
        )
    ).unwrap();

    for i in 0..5 {
        let mut dist = NIW::sample(&dist_params, &mut rng);
        let mut points = DMatrix::<f64>::zeros(2, 500);
        for mut point in points.column_iter_mut() {
            point.copy_from(&dist.sample(&mut rng));
        }

        let stats = NIWStats::from_data(&points);
        dist_params = NIW::posterior(&dist_params, &stats);

        plot_ctx.draw_series(
            points.column_iter()
                .map(|x| Circle::new((x[0], x[1]), 2, Palette99::pick(i + 3).mix(0.8).filled()))
        ).unwrap();

        plot_ctx.draw_series(
            Cluster2D::from_mat(
                dist.mu(), dist.cov(),
                50,
                Palette99::pick(i + 3).filled(),
                Palette99::pick(i + 3).stroke_width(2),
            )
        ).unwrap()
            .label(i.to_string())
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], Palette99::pick(i + 3).filled()));
    }

    plot_ctx
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()
        .unwrap();


    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", PATH);
}