use std::collections::HashMap;
use std::thread::available_parallelism;
use std::time::{Duration, Instant};
use itertools::Itertools;
use nalgebra::{DMatrix, RowDVector};
use rand::prelude::*;
use crate::callback::{Callback, EvalData};
use crate::metrics::Metric;
use crate::params::options::{FitOptions, ModelOptions};
use crate::params::thin::{MixtureParams, SuperMixtureParams, ThinParams};
use crate::state::{GlobalState, GlobalWorker, LocalState, LocalWorker, ShardedState};
use crate::stats::NormalConjugatePrior;
use crate::utils::{reservoir_sampling, reservoir_sampling_weighted};

pub struct Model<
    P: NormalConjugatePrior,
> {
    global: Option<GlobalState<P>>,
    model_options: ModelOptions<P>,
}

impl<P: NormalConjugatePrior> Model<P> {
    pub fn from_options(model_options: ModelOptions<P>) -> Self {
        Self {
            global: None,
            model_options,
        }
    }

    pub fn n_clusters(&self) -> usize {
        if let Some(global) = &self.global {
            GlobalWorker::n_clusters(global)
        } else {
            0
        }
    }

    pub fn is_fitted(&self) -> bool {
        self.global.is_some()
    }

    pub fn fit(
        &mut self,
        data: DMatrix<f64>,
        fit_options: &FitOptions,
        mut callback: Option<impl Callback<GlobalState<P>>>,
    ) {
        let mut rng = SmallRng::seed_from_u64(fit_options.seed);
        match fit_options.workers {
            0 | 1 => {
                let mut local = LocalState::from_data(data);
                local.init(fit_options.init_clusters, &mut rng);

                self.fit_worker(&mut local, fit_options, callback);
            },
            workers => {
                let workers = if workers < 0 { available_parallelism().unwrap().get() as i32 } else { workers };
                let mut local = ShardedState::from_data(data, workers as usize);
                local.init(fit_options.init_clusters, &mut rng);

                self.fit_worker(&mut local, fit_options, callback);
            }
        }
    }

    pub fn fit_worker<L: LocalWorker<P>>(
        &mut self,
        local: &mut L,
        fit_options: &FitOptions,
        mut callback: Option<impl Callback<GlobalState<P>>>,
    ) {
        let mut rng = SmallRng::seed_from_u64(fit_options.seed);

        // (Re)initialize global state
        if fit_options.reuse {
            if self.global.is_none() {
                panic!("Cannot reuse global state if it has not been initialized yet");
            }
        } else {
            let data_stats = local.collect_data_stats();
            self.global = Some(
                GlobalState::from_init(&data_stats, fit_options.init_clusters, &self.model_options, &mut rng)
            );
        }
        let global = self.global.as_mut().unwrap();

        // Initialize clusters from local states / data
        let stats = local.collect_cluster_stats(GlobalWorker::n_clusters(global));
        global.update_clusters_post(stats);
        global.update_sample_clusters(&self.model_options, &mut rng);

        for i in 0..fit_options.iters {
            let is_cooldown = i >= fit_options.iters - fit_options.argmax_sample_stop;
            let no_more_actions = i >= fit_options.iters - fit_options.iter_split_stop;
            let no_more_splits = GlobalWorker::n_clusters(global) >= fit_options.max_clusters;

            // Before step callback
            if let Some(callback) = &mut callback {
                callback.before_step(i);
            }

            // Expectation step
            global.update_sample_clusters(&self.model_options, &mut rng);
            local.apply_label_sampling(global, is_cooldown, &mut rng);

            // Maximization step
            let stats = local.collect_cluster_stats(GlobalWorker::n_clusters(global));
            global.update_clusters_post(stats);

            // Reset bad clusters (with concentrated subclusters)
            let bad_clusters = global.collect_bad_clusters();
            local.apply_cluster_reset(&bad_clusters, &mut rng);

            // Compute metrics before any action is applied
            if let Some(callback) = &mut callback {
                callback.during_step(i, global);
            }

            // Proposal step
            if !no_more_actions {
                // Propose split actions
                if !no_more_splits {
                    let split_idx = global.check_and_split(&self.model_options, &mut rng);
                    local.apply_split(&split_idx, &mut rng);

                    if split_idx.len() > 0 {
                        let stats = local.collect_cluster_stats(GlobalWorker::n_clusters(global));
                        global.update_clusters_post( stats);
                    }
                }

                // Propose merge actions
                let merge_idx = global.check_and_merge(&self.model_options, &mut rng);
                local.apply_merge(&merge_idx);
            }

            // Remove empty clusters
            let removed_idx = global.collect_remove_clusters(&self.model_options);
            local.apply_cluster_remove(&removed_idx);

            // After step callback
            if let Some(callback) = &mut callback {
                callback.after_step(i);
            }
        }
    }

    pub fn predict<L: LocalWorker<P>>(
        &mut self,
        data: DMatrix<f64>,
    ) -> (DMatrix<f64>, RowDVector<usize>) {
        if self.global.is_none() {
            panic!("Cannot predict if model has not been fitted yet");
        }

        let global = self.global.as_ref().unwrap();
        SuperMixtureParams(global).predict(data)
    }
}


