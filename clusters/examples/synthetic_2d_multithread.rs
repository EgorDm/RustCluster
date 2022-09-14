use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::f64::consts::PI;
use std::io::Read;
use std::rc::Rc;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Instant;
use crossbeam_channel::unbounded;
use nalgebra::{DMatrix, DVector, Dynamic, Matrix, Storage};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use plotters::prelude::*;
use rand::prelude::{SmallRng, StdRng};
use rand::{Rng, SeedableRng};
use clusters::params::options::{FitOptions, ModelOptions};
use clusters::plotting::{axes_range_from_points, Cluster2D, init_axes2d};
use clusters::stats::{NIW, NIWStats, SufficientStats};
use rayon::prelude::*;
use clusters::callback::MonitoringCallback;
use clusters::metrics::NMI;
use clusters::model::Model;
use clusters::params::clusters::SuperClusterParams;
use clusters::state::{GlobalState, LocalWorker, ShardedState};
use clusters::state::GlobalWorker;


fn plot<S: Storage<f64, Dynamic, Dynamic>>(
    path: &str,
    points: &Matrix<f64, Dynamic, Dynamic, S>, labels: &[usize], clusters: &[SuperClusterParams<NIW>],
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

    // let mut rng = SmallRng::seed_from_u64(42);
    let mut rng = StdRng::seed_from_u64(42);
    let plot_idx: Vec<usize> = (0..1000).map(|_| rng.gen_range(0..x.ncols())).collect();
    let plot_x = x.select_columns(&plot_idx);

    let dim = x.nrows();

    let mut model_options = ModelOptions::<NIW>::default(dim);
    model_options.alpha = 100.0;
    model_options.outlier = None;
    let mut fit_options = FitOptions::default();
    fit_options.init_clusters = 10;
    // fit_options.iters = 20;
    // fit_options.iters = 40;
    fit_options.workers = 10;

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

    // let mut rng = StdRng::seed_from_u64(fit_options.seed);
    // let mut worker = ShardedState::from_data(x.clone(), 10);
    // worker.init(fit_options.init_clusters, &mut rng);
    //
    // let data_stats = worker.collect_data_stats();
    // let mut global_state = GlobalState::<NIW>::from_init(&data_stats, fit_options.init_clusters, &model_options, &mut rng);
    //
    // // update_suff_stats_posterior!
    // let stats = worker.collect_cluster_stats(global_state.n_clusters());
    // global_state.update_clusters_post(stats);
    // global_state.update_sample_clusters(&model_options, &mut rng);
    //
    //
    // // let plot_y = local_state.labels.select_rows(&plot_idx);
    // // plot(&format!("examples/data/plot/synthetic_2d/init.png"), &plot_x, plot_y.as_slice(), &global_state.clusters);
    //
    // // // Testing !!!
    // // LocalState::update_sample_labels(&global_state, &mut local_state, true, &mut rng);
    // // LocalState::update_sample_labels_aux(&global_state, &mut local_state, &mut rng);
    // //
    // // let plot_y = y.select_columns(&plot_idx);
    // // plot(&format!("examples/data/plot/synthetic_2d/test.png"), &plot_x, plot_y.as_slice(), &global_state.clusters);
    // // // End Testing
    //
    // for i in 0..fit_options.iters {
    //     let is_final = i >= fit_options.iters - fit_options.argmax_sample_stop;
    //     let no_more_splits = i >= fit_options.iters - fit_options.iter_split_stop || global_state.n_clusters() >= fit_options.max_clusters;
    //
    //     // Run step
    //     let now = Instant::now();
    //     {
    //         global_state.update_sample_clusters(&model_options, &mut rng);
    //         worker.apply_label_sampling(&global_state, is_final, &mut rng);
    //
    //         // update_suff_stats_posterior!
    //         let stats = worker.collect_cluster_stats(global_state.n_clusters());
    //         global_state.update_clusters_post(stats);
    //         // Remove reset bad clusters (concentrated subclusters)
    //         let bad_clusters = GlobalState::collect_bad_clusters(&mut global_state);
    //         worker.apply_cluster_reset(&bad_clusters, &mut rng);
    //
    //         if !no_more_splits {
    //             let split_idx = global_state.check_and_split(&model_options, &mut rng);
    //             worker.apply_split(&split_idx, &mut rng);
    //
    //             if split_idx.len() > 0 {
    //                 let stats = worker.collect_cluster_stats(global_state.n_clusters());
    //                 global_state.update_clusters_post( stats);
    //             }
    //             let merge_idx = global_state.check_and_merge(&model_options, &mut rng);
    //             worker.apply_merge(&merge_idx);
    //         }
    //
    //         let removed_idx = global_state.collect_remove_clusters(&model_options);
    //         worker.apply_cluster_remove(&removed_idx);
    //     }
    //     let elapsed = now.elapsed();
    //
    //     let mut nmi: f64 = worker.shards.par_iter().enumerate().map(|(i, local_state)| {
    //         normalized_mutual_info_score(
    //             y.columns_range(i * 1000..(i + 1) * 1000).clone_owned().as_slice(),
    //             local_state.labels.as_slice(),
    //         )
    //     }).sum();
    //     nmi /= worker.n_shards() as f64;
    //
    //     println!("Run iteration {} in {:.2?}; k={}, nmi={}", i, elapsed, global_state.n_clusters(), nmi);
    //
    //
    //     // let plot_y = y.select_columns(&plot_idx);
    //     // plot(&format!("examples/data/plot/synthetic_2d/step_{:04}.png", i), &plot_x, plot_y.as_slice(), &global_state.clusters);
    //     // calculate NMI
    // }
    //
    // // dbg!(local_state);
    // // dbg!(global_state);


    let u = 0;
}