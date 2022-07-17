use std::io::Read;
use std::time::Instant;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use rand::prelude::StdRng;
use rand::SeedableRng;
use dpmmsc::global::{GlobalActions, GlobalState};
use dpmmsc::local::{LocalActions, LocalState};
use dpmmsc::metrics::normalized_mutual_info_score;
use dpmmsc::options::{FitOptions, ModelOptions};
use dpmmsc::priors::{NIW, NIWParams, NIWStats, SufficientStats};


fn main() {
    println!("Hello, world!");

    let x_data: Array2<f64> = read_npy("examples/data/x.npy").unwrap();
    let x = DMatrix::from_row_slice(x_data.nrows(), x_data.ncols(), &x_data.as_slice().unwrap());
    let y_data: Array1<i64> = read_npy("examples/data/y.npy").unwrap();
    let y = DVector::from_row_slice(&y_data.as_slice().unwrap());
    let y = y.map(|x| x as usize).into_owned();

    let dim = x.ncols();

    let mut model_options = ModelOptions::<NIW>::default(dim);
    model_options.alpha = 100.0;
    let fit_options = FitOptions::default();
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

            if !no_more_splits {
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
            }

            let removed_idx = GlobalState::update_remove_empty_clusters(&mut global_state, &model_options);
            LocalState::update_remove_clusters(&mut local_state, &removed_idx);
        }
        let elapsed = now.elapsed();

        let nmi = normalized_mutual_info_score(y.as_slice(), local_state.labels.as_slice());
        println!("Run iteration {} in {:.2?}; k={}, nmi={}", i, elapsed, global_state.n_clusters(), nmi);

        // calculate NMI
    }

    // dbg!(local_state);
    // dbg!(global_state);


    let u = 0;
}