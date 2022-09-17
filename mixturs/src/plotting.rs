use std::f64::consts::PI;
use std::marker::PhantomData;
use std::ops::{Index, Range};
use std::path::PathBuf;
use nalgebra::{Dynamic, Matrix, Storage};
use plotters::coord::Shift;
use plotters::coord::types::RangedCoordf64;
use plotters::element::{Drawable, PointCollection};
use plotters::prelude::*;
use plotters_backend::{BackendCoord, DrawingErrorKind};
use crate::callback::{EvalData, Callback};
use crate::params::{ThinParams, SuperMixtureParams, MixtureParams};

pub type PointF = (f64, f64);

/// A drawable ellipse object
pub struct Ellipse {
    center: PointF,
    size: [[f64; 2]; 2],
    style: ShapeStyle,
    precision: usize,
}

impl Ellipse {
    pub fn new<S: Into<ShapeStyle>>(center: PointF, size: [[f64; 2]; 2], style: S, precision: usize) -> Self {
        Ellipse { center, size, style: style.into(), precision }
    }
}

impl<'a> PointCollection<'a, PointF> for &'a Ellipse {
    type Point = PointF;
    type IntoIter = Box<dyn Iterator<Item=PointF> + 'a>;

    fn point_iter(self) -> Self::IntoIter {
        Box::new((0..self.precision).map(|i| {
            let t = i as f64 / (self.precision as f64) * PI * 2.0;
            let (cx, cy) = (t.cos(), t.sin());

            let x = self.size[0][0] * cx + self.size[0][1] * cy + self.center.0;
            let y = self.size[1][0] * cx + self.size[1][1] * cy + self.center.1;
            (x, y)
        }))
    }
}

impl<DB: DrawingBackend> Drawable<DB> for Ellipse {
    fn draw<I: Iterator<Item=BackendCoord>>(
        &self,
        points: I,
        backend: &mut DB,
        _: (u32, u32),
    ) -> Result<(), DrawingErrorKind<DB::ErrorType>> {
        backend.draw_path(
            points,
            &self.style,
        )
    }
}


/// A drawable cluster object
pub struct Cluster2D<DB: DrawingBackend> {
    mu: PointF,
    cov: [[f64; 2]; 2],
    accuracy: usize,
    center_style: ShapeStyle,
    contour_style: ShapeStyle,
    _phantom: PhantomData<DB>,
}

impl<DB: DrawingBackend> Cluster2D<DB> {
    pub fn new(mu: PointF, cov: [[f64; 2]; 2], accuracy: usize, center_style: ShapeStyle, contour_style: ShapeStyle) -> Self {
        Cluster2D { mu, cov, accuracy, center_style, contour_style, _phantom: PhantomData }
    }

    pub fn from_mat(
        mu: &impl Index<usize, Output=f64>,
        cov: &impl Index<(usize, usize), Output=f64>,
        accuracy: usize, center_style: ShapeStyle, contour_style: ShapeStyle,
    ) -> Self {
        Cluster2D {
            mu: (mu[0], mu[1]),
            cov: [
                [cov[(0, 0)], cov[(0, 1)]],
                [cov[(1, 0)], cov[(1, 1)]],
            ],
            accuracy,
            center_style,
            contour_style,
            _phantom: PhantomData,
        }
    }
}

impl<DB: DrawingBackend> IntoIterator for Cluster2D<DB> {
    type Item = DynElement<'static, DB, PointF>;
    type IntoIter = std::array::IntoIter<Self::Item, 2>;

    fn into_iter(self) -> Self::IntoIter {
        [
            Circle::new(self.mu, 8, self.center_style).into_dyn(),
            Ellipse::new(self.mu, self.cov, self.contour_style, self.accuracy).into_dyn(),
        ].into_iter()
    }
}


/// Computes the plot range for a given set of data
///
/// # Arguments
///
/// * `points`: The data points (n_dims, n_points)
///
/// # Returns:
///
/// A tuple containing:
/// * X range
/// * Y range
pub fn axes_range_from_points<S: Storage<f64, Dynamic, Dynamic>>(
    points: &Matrix<f64, Dynamic, Dynamic, S>
) -> (Range<f64>, Range<f64>) {
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    for col in points.column_iter() {
        min_x = min_x.min(col[0]);
        max_x = max_x.max(col[0]);
        min_y = min_y.min(col[1]);
        max_y = max_y.max(col[1]);
    }
    (min_x..max_x, min_y..max_y)
}

/// Configures 2D plotting area and axes for plotting clusters
///
/// # Arguments
///
/// * `range`: The X and Y ranges
/// * `root`: The root drawing area
pub fn init_axes2d<
    'a, DB: 'a + DrawingBackend,
>(
    range: (Range<f64>, Range<f64>),
    root: &'a DrawingArea<DB, Shift>,
) -> ChartContext<'a, DB, Cartesian2d<RangedCoordf64, RangedCoordf64>> {
    root.fill(&WHITE).unwrap();
    let mut plot_ctx = ChartBuilder::on(root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(range.0, range.1).unwrap();
    plot_ctx
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw().unwrap();

    plot_ctx
}


/// Callback for plotting the clustering state every `freq` iterations
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
    params: &impl MixtureParams,
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
