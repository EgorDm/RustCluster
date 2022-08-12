use std::f64::consts::PI;
use std::io::Read;
use std::time::Instant;
use nalgebra::{DMatrix, DVector, Dynamic, Matrix, Storage};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use plotters::prelude::*;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use dpmmsc::clusters::SuperClusterParams;
use dpmmsc::global::{GlobalActions, GlobalState};
use dpmmsc::local::{LocalActions, LocalState};
use dpmmsc::metrics::normalized_mutual_info_score;
use dpmmsc::options::{FitOptions, ModelOptions};
use dpmmsc::priors::{NIW, NIWParams, NIWStats, SufficientStats};

fn plot<S: Storage<f64, Dynamic, Dynamic>>(
    path: &str,
    points: &Matrix<f64, Dynamic, Dynamic, S>, labels: &[usize], clusters: &[SuperClusterParams<NIW>]
) {
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    for row in points.row_iter() {
        min_x = min_x.min(row[0]);
        max_x = max_x.max(row[0]);
        min_y = min_y.min(row[1]);
        max_y = max_y.max(row[1]);
    }

    let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let areas = root.split_by_breakpoints([944], [80]);
    let mut scatter_ctx = ChartBuilder::on(&areas[2])
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_x..max_x, min_y..max_y).unwrap();
    scatter_ctx
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw().unwrap();
    scatter_ctx.draw_series(
        points
            .row_iter()
            .zip(labels.iter())
            .map(|(row, label)|
                Circle::new((row[0], row[1]), 2, Palette99::pick(*label).mix(0.9).filled(),
                )),
    ).unwrap();

    scatter_ctx.draw_series(
        clusters.iter()
            .enumerate()
            .map(|(k, cluster)| {
                let color = Palette99::pick(k);
                let circle = Circle::new((cluster.prim.dist.mu()[0], cluster.prim.dist.mu()[1]), 8, color.filled().stroke_width(2));
                circle
            }),
    ).unwrap();

    for (k, cluster) in clusters.iter().enumerate() {
        scatter_ctx.draw_series(LineSeries::new(
            (0..100).map(|i| {
                let t = i as f64 / 100.0 * PI * 2.0;
                let (cx, cy) = (t.cos(), t.sin());

                let x = cluster.prim.dist.cov()[(0, 0)] * cx + cluster.prim.dist.cov()[(0, 1)] * cy + cluster.prim.dist.mu()[0];
                let y = cluster.prim.dist.cov()[(1, 0)] * cx + cluster.prim.dist.cov()[(1, 1)] * cy + cluster.prim.dist.mu()[1];

                (x, y)
            }),
            Palette99::pick(k)
        )).unwrap();
    }



    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", path);
}


fn main() {
    println!("Hello, world!");

    let x_data: Array2<f64> = read_npy("examples/data/x.npy").unwrap();
    let x = DMatrix::from_row_slice(x_data.nrows(), x_data.ncols(), &x_data.as_slice().unwrap());
    let y_data: Array1<i64> = read_npy("examples/data/y.npy").unwrap();
    let y = DVector::from_row_slice(&y_data.as_slice().unwrap());
    let y = y.map(|x| x as usize).into_owned();

    let mut rng = StdRng::seed_from_u64(42);
    let plot_idx: Vec<usize> = (0..1000).map(|_| rng.gen_range(0..x.nrows())).collect();
    let plot_x = x.select_rows(&plot_idx);
    let plot_y = y.select_rows(&plot_idx);

    let dim = x.ncols();

    let mut model_options = ModelOptions::<NIW>::default(dim);
    model_options.alpha = 100.0;
    model_options.outlier = None;
    let mut fit_options = FitOptions::default();
    fit_options.init_clusters = 2;

    let mut rng = StdRng::seed_from_u64(fit_options.seed);
    let mut local_state = LocalState::<NIW>::from_init(x, fit_options.init_clusters, &model_options, &mut rng);
    let data_stats = NIWStats::from_data(&local_state.data);
    let mut global_state = GlobalState::<NIW>::from_init(&data_stats, fit_options.init_clusters, &model_options, &mut rng);

    // update_suff_stats_posterior!
    let stats = LocalState::<NIW>::collect_stats(&local_state, global_state.n_clusters());
    for (k, (prim, aux)) in stats.into_iter().enumerate() {
        global_state.clusters[k].prim.stats = prim;
        for (ki, aux) in aux.into_iter().enumerate() {
            global_state.clusters[k].aux[ki].stats = aux;
        }
    }
    GlobalState::update_sample_clusters(&mut global_state, &model_options, &mut rng);

    for i in 0..fit_options.iters {
        let is_final = i >= fit_options.iters - fit_options.argmax_sample_stop;
        let no_more_splits = i >= fit_options.iters - fit_options.iter_split_stop || global_state.n_clusters() >= fit_options.max_clusters;

        // Run step
        let now = Instant::now();
        {
            GlobalState::update_sample_clusters(&mut global_state, &model_options, &mut rng);
            LocalState::update_sample_labels(&global_state, &mut local_state, is_final, &mut rng);
            LocalState::update_sample_labels_aux(&global_state, &mut local_state, &mut rng);
            // update_suff_stats_posterior!
            let stats = LocalState::<NIW>::collect_stats(&local_state, global_state.n_clusters());
            for (k, (prim, aux)) in stats.into_iter().enumerate() {
                global_state.clusters[k].prim.stats = prim;
                for (ki, aux) in aux.into_iter().enumerate() {
                    global_state.clusters[k].aux[ki].stats = aux;
                }
            }
            let bad_clusters = GlobalState::collect_bad_clusters(&mut global_state);
            LocalState::update_reset_clusters(&mut local_state, &bad_clusters, &mut rng);

            /*if !no_more_splits {
                let split_idx = GlobalActions::check_and_split(&mut global_state, &model_options, &mut rng);
                LocalActions::apply_split(&mut local_state, &split_idx, &mut rng);
                if split_idx.len() > 0 {
                    let stats = LocalState::<NIW>::collect_stats(&local_state, global_state.n_clusters());
                    for (k, (prim, aux)) in stats.into_iter().enumerate() {
                        global_state.clusters[k].prim.stats = prim;
                        for (ki, aux) in aux.into_iter().enumerate() {
                            global_state.clusters[k].aux[ki].stats = aux;
                        }
                    }
                }
                let merge_idx = GlobalActions::check_and_merge(&mut global_state, &model_options, &mut rng);
                LocalActions::apply_merge(&mut local_state, &merge_idx);
            }*/

            let removed_idx = GlobalState::update_remove_empty_clusters(&mut global_state, &model_options);
            LocalState::update_remove_clusters(&mut local_state, &removed_idx);
        }
        let elapsed = now.elapsed();

        let nmi = normalized_mutual_info_score(y.as_slice(), local_state.labels.as_slice());
        println!("Run iteration {} in {:.2?}; k={}, nmi={}", i, elapsed, global_state.n_clusters(), nmi);

        plot(&format!("examples/data/plot/step_{:04}.png", i), &plot_x, plot_y.as_slice(), &global_state.clusters);


        // calculate NMI
    }

    // dbg!(local_state);
    // dbg!(global_state);


    let u = 0;
}