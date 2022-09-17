use std::fs::File;
use nalgebra::{DMatrix, DVector, RowDVector};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use plotters::coord::Shift;
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use mixturs::plotting::{Cluster2D, Ellipse, init_axes2d, axes_range_from_points};
use mixturs::stats::Covariance;

const PATH: &str = "examples/data/plot/plot_data.png";

fn main() {
    let mut f = File::open("examples/data/x.bin").unwrap();
    let x: DMatrix<f64> = deserialize_from(&mut f).unwrap();
    let mut f = File::open("examples/data/y.bin").unwrap();
    let y: RowDVector<usize> = deserialize_from(&mut f).unwrap();

    let (mut range_x, mut range_y) = axes_range_from_points(&x);
    let root: DrawingArea<BitMapBackend, Shift> = BitMapBackend::new(PATH, (1024, 768)).into_drawing_area();
    let mut plot_ctx: ChartContext<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>> = init_axes2d((range_x, range_y), &root);

    // Scatter plot points
    plot_ctx.draw_series(
        x
            .column_iter()
            .zip(y.iter())
            .map(|(row, label)|
                Circle::new((row[0], row[1]), 2, Palette99::pick(*label).mix(0.4).filled())
            ),
    ).unwrap();

    for k in 0..7 {
        let idx = y.iter().enumerate().filter_map(|(i, &y)| if y == k { Some(i) } else { None }).collect::<Vec<usize>>();
        let points = x.select_columns(&idx);

        let mu = points.column_mean().into_owned();
        let cov = points.col_cov();

        plot_ctx.draw_series(
            Cluster2D::from_mat(
                &mu, &cov,
                100,
                Palette99::pick(k + 1).filled(),
                Palette99::pick(k + 1).stroke_width(2),
            )
        ).unwrap();
    }

    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", PATH);
}