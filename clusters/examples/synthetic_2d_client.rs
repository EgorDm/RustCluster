use remoc::prelude::*;
use std::{net::Ipv4Addr, time::Duration};
use std::time::Instant;
use rand::prelude::*;
use tokio::net::TcpStream;
use clusters::executor::rtc::{TCP_PORT, CounterClient, Counter};
use clusters::global::{GlobalActions, GlobalState};
use clusters::options::{FitOptions, ModelOptions};
use clusters::stats::NIW;

#[tokio::main]
async fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    let dim = 2;
    let mut model_options = ModelOptions::<NIW>::default(dim);
    model_options.alpha = 100.0;
    model_options.outlier = None;
    let mut fit_options = FitOptions::default();
    fit_options.init_clusters = 10;

    let mut rng = StdRng::seed_from_u64(fit_options.seed + 1000);

    // ------------------------------------------------------------------------
    // Establish TCP connection to server.
    let socket = TcpStream::connect((Ipv4Addr::LOCALHOST, TCP_PORT)).await.unwrap();
    let (socket_rx, socket_tx) = socket.into_split();

    // Establish a Remoc connection with default configuration over the TCP connection and
    // consume (i.e. receive) the counter client from the server.
    let mut client: CounterClient =
        remoc::Connect::io(remoc::Cfg::default(), socket_rx, socket_tx).consume().await.unwrap();

    // ------------------------------------------------------------------------

    let data_stats = client.collect_data_stats().await.unwrap();

    let mut global_state = GlobalState::<NIW>::from_init(&data_stats, fit_options.init_clusters, &model_options, &mut rng);
    let stats = client.collect_stats(global_state.n_clusters()).await.unwrap();
    GlobalState::update_clusters_post(&mut global_state, stats);
    GlobalState::update_sample_clusters(&mut global_state, &model_options, &mut rng);

    for i in 0..fit_options.iters {
        let is_final = i >= fit_options.iters - fit_options.argmax_sample_stop;
        let no_more_splits = i >= fit_options.iters - fit_options.iter_split_stop || global_state.n_clusters() >= fit_options.max_clusters;

        // Run step
        let now = Instant::now();
        {
            GlobalState::update_sample_clusters(&mut global_state, &model_options, &mut rng);
            client.update_sample_labels(global_state.clone(), is_final).await.unwrap();

            // update_suff_stats_posterior!
            let stats = client.collect_stats(global_state.n_clusters()).await.unwrap();
            GlobalState::update_clusters_post(&mut global_state, stats);

            // Remove reset bad clusters (concentrated subclusters)
            let bad_clusters = GlobalState::collect_bad_clusters(&mut global_state);
            client.update_reset_clusters(bad_clusters).await.unwrap();

            if !no_more_splits {
                let split_idx = GlobalActions::check_and_split(&mut global_state, &model_options, &mut rng);
                let has_split = !split_idx.is_empty();
                client.apply_split(split_idx).await.unwrap();

                if has_split {
                    let stats = client.collect_stats(global_state.n_clusters()).await.unwrap();
                    GlobalState::update_clusters_post(&mut global_state, stats);
                }
                let merge_idx = GlobalActions::check_and_merge(&mut global_state, &model_options, &mut rng);
                client.apply_merge(merge_idx).await.unwrap();
            }

            let removed_idx = GlobalState::update_remove_empty_clusters(&mut global_state, &model_options);
            client.update_remove_clusters(removed_idx).await.unwrap();
        }
        let elapsed = now.elapsed();

        // let mut nmi: f64 = local_states.par_iter().enumerate().map(|(i, local_state)| {
        //     normalized_mutual_info_score(
        //         y.columns_range(i * 1000..(i + 1) * 1000).clone_owned().as_slice(),
        //         local_state.labels.as_slice(),
        //     )
        // }).sum();
        // nmi /= local_states.len() as f64;

        println!("Run iteration {} in {:.2?}; k={}", i, elapsed, global_state.n_clusters());


        // let plot_y = y.select_columns(&plot_idx);
        // plot(&format!("examples/data/plot/synthetic_2d/step_{:04}.png", i), &plot_x, plot_y.as_slice(), &global_state.clusters);
        // calculate NMI
    }
}