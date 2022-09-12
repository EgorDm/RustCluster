use std::collections::HashMap;
use std::thread::available_parallelism;
use std::time::{Duration, Instant};
use itertools::Itertools;
use nalgebra::{DMatrix, RowDVector};
use rand::prelude::*;
use crate::metrics::{EvaluationData, Metric};
use crate::params::options::{FitOptions, ModelOptions};
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
            global.n_clusters()
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
        mut eval_data: Option<EvaluationData>,
        metric: Option<impl Metric<GlobalState<P>>>,
    ) {
        let mut rng = SmallRng::seed_from_u64(fit_options.seed);

        // Initialize the evaluation data.
        if metric.is_some() && eval_data.is_none() {
            let mut indices = vec![0; fit_options.max_eval_points];
            let n_points = reservoir_sampling(&mut rng, 0..data.ncols(), &mut indices);
            let points = data.select_columns(&indices[..n_points]);
            eval_data = Some(EvaluationData {
                points,
                labels: None,
            });
        }

        match fit_options.workers {
            0 | 1 => {
                let mut local = LocalState::from_data(data);
                local.init(fit_options.init_clusters, &mut rng);

                self.fit_worker(&mut local, fit_options, eval_data, metric);
            },
            workers => {
                let workers = if workers < 0 { available_parallelism().unwrap().get() as i32 } else { workers };
                let mut local = ShardedState::from_data(data, workers as usize);
                local.init(fit_options.init_clusters, &mut rng);

                self.fit_worker(&mut local, fit_options, eval_data, metric);
            }
        }
    }

    pub fn fit_worker<L: LocalWorker<P>>(
        &mut self,
        local: &mut L,
        fit_options: &FitOptions,
        eval_data: Option<EvaluationData>,
        mut metric: Option<impl Metric<GlobalState<P>>>,
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
        let stats = local.collect_cluster_stats(global.n_clusters());
        global.update_clusters_post(stats);
        global.update_sample_clusters(&self.model_options, &mut rng);

        let mut metrics = HashMap::new();

        for i in 0..fit_options.iters {
            let is_cooldown = i >= fit_options.iters - fit_options.argmax_sample_stop;
            let no_more_actions = i >= fit_options.iters - fit_options.iter_split_stop;
            let no_more_splits = global.n_clusters() >= fit_options.max_clusters;

            let now = Instant::now();
            metrics.clear();

            // Expectation step
            global.update_sample_clusters(&self.model_options, &mut rng);
            local.apply_label_sampling(global, is_cooldown, &mut rng);

            // Maximization step
            let stats = local.collect_cluster_stats(global.n_clusters());
            global.update_clusters_post(stats);

            // Reset bad clusters (with concentrated subclusters)
            let bad_clusters = global.collect_bad_clusters();
            local.apply_cluster_reset(&bad_clusters, &mut rng);

            // Compute metrics before any action is applied
            if fit_options.verbose {
                if let (Some(eval_data), Some(metric)) = (&eval_data, &mut metric) {
                    metric.compute(eval_data, global, &mut metrics);
                }
            }

            // Proposal step
            if !no_more_actions {
                // Propose split actions
                if !no_more_splits {
                    let split_idx = global.check_and_split(&self.model_options, &mut rng);
                    local.apply_split(&split_idx, &mut rng);

                    if split_idx.len() > 0 {
                        let stats = local.collect_cluster_stats(global.n_clusters());
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

            let elapsed = now.elapsed();
            if fit_options.verbose {
                let metrics = metrics.iter().map(|(k, v)| format!("{}={:.4}", k, v)).join(", ");
                println!("Run iteration {} in {:.2?}; k={}, {}", i, elapsed, global.n_clusters(), metrics);
            }
        }
    }

    pub fn predict<L: LocalWorker<P>>(
        &mut self,
        data: DMatrix<f64>,
        workers: i32,
    ) -> RowDVector<usize> {
        if self.global.is_none() {
            panic!("Cannot predict if model has not been fitted yet");
        }

        todo!()

       /* let mut rng = SmallRng::seed_from_u64(42);
        match workers {
            0 | 1 => {
                let mut local = LocalState::<P>::from_data(data);
                local.apply_sample_labels_prim(self.global.as_ref().unwrap(), true, &mut rng);
                local.labels
            },
            workers => {
                let workers = if workers < 0 { available_parallelism().unwrap().get() as i32 } else { workers };
                let mut local = ShardedState::<P>::from_data(data, workers as usize);
                local.apply_sample_labels_prim::<HardAssignSampling>(self.global.as_ref().unwrap(), &mut rng);

                let mut labels = RowDVector::zeros(local.n_points());
                let mut cursor = 0;
                for shard in local.shards {
                    labels.columns_range_mut(cursor..cursor+shard.n_points()).copy_from(&shard.labels);
                }
                labels
            }
        }*/
    }
}