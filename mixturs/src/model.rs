use std::thread::available_parallelism;
use nalgebra::{DMatrix, RowDVector};
use rand::prelude::*;
use crate::callback::{Callback};
use crate::params::options::{FitOptions, ModelOptions};
use crate::params::thin::{MixtureParams, SuperMixtureParams};
use crate::state::{GlobalState, GlobalWorker, LocalState, LocalWorker, ShardedState};
use crate::stats::NormalConjugatePrior;

// pub struct Model<
//     P: NormalConjugatePrior,
// > {
//     global: Option<GlobalState<P>>,
//     model_options: ModelOptions<P>,
// }

/// Dirichlet Process Mixture Model (DPMM) Split/Merge model introduced in
/// [1] and [2].
///
/// [1] J. Chang and J. W. Fisher III, “Parallel Sampling of DP Mixture Models using Sub-Cluster Splits,” in Advances in Neural Information Processing Systems, 2013.
/// [2] O. Dinari, A. Yu, O. Freifeld, and J. Fisher, “Distributed MCMC Inference in Dirichlet Process Mixture Models Using Julia,” in 2019 19th IEEE/ACM International Symposium on Cluster, Cloud and Grid Computing (CCGRID).
///
/// # Example:
/// ```
/// use nalgebra::{DMatrix, RowDVector};
/// use mixturs::{FitOptions, Model, ModelOptions, MonitoringCallback, NIW};use mixturs::callback::EvalData;
///
/// let dim = 2;
/// let x = DMatrix::new_random(dim, 100);
///
/// let model_options = ModelOptions::<NIW>::default(dim);
/// let mut model = Model::from_options(model_options);
///
/// let fit_options = FitOptions::default();
/// let callback = MonitoringCallback::from_data(
///         EvalData::from_sample(&x, None, 1000)
/// );
///
/// model.fit(
///     x.clone_owned(),
///     &fit_options,
///     Some(callback)
/// );
/// ```
pub struct Model<
    P: NormalConjugatePrior,
> {
    global: Option<GlobalState<P>>,
    model_options: ModelOptions<P>,
}

impl<P: NormalConjugatePrior> Model<P> {
    /// Create a new model from a set of model options.
    pub fn from_options(model_options: ModelOptions<P>) -> Self {
        Self {
            global: None,
            model_options,
        }
    }

    /// Count the number of clusters in the model.
    pub fn n_clusters(&self) -> usize {
        if let Some(global) = &self.global {
            GlobalWorker::n_clusters(global)
        } else {
            0
        }
    }

    /// Check whether the model is already fitted.
    pub fn is_fitted(&self) -> bool {
        self.global.is_some()
    }

    /// Fit the model to the data.
    ///
    /// # Arguments
    ///
    /// * `data`: The data to fit the model to.
    /// * `fit_options`: Options for the fitting procedure.
    /// * `callback`: Callback function to monitor the fitting procedure.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{DMatrix, RowDVector};
    /// use mixturs::{FitOptions, Model, ModelOptions, MonitoringCallback, NIW};
    /// use mixturs::callback::EvalData;
    ///
    /// let dim = 2;
    /// let x = DMatrix::new_random(dim, 100);
    ///
    /// let model_options = ModelOptions::<NIW>::default(dim);
    /// let mut model = Model::from_options(model_options);
    ///
    /// let fit_options = FitOptions::default();
    /// let callback = MonitoringCallback::from_data(
    ///        EvalData::from_sample(&x, None, 1000)
    /// );
    ///
    /// model.fit(
    ///    x.clone_owned(),
    ///   &fit_options,
    ///  Some(callback)
    /// );
    /// ```
    pub fn fit(
        &mut self,
        data: DMatrix<f64>,
        fit_options: &FitOptions,
        callback: Option<impl Callback<GlobalState<P>>>,
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

    /// Fit the model using the data workers.
    ///
    /// # Arguments
    ///
    /// * `local`: The data workers.
    /// * `fit_options`: Options for the fitting procedure.
    /// * `callback`: Callback function to monitor the fitting procedure.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{DMatrix, RowDVector};
    /// use mixturs::{FitOptions, Model, ModelOptions, MonitoringCallback, NIW};
    /// use mixturs::callback::EvalData;
    /// use mixturs::state::ShardedState;
    ///
    /// let dim = 2;
    /// let x = DMatrix::new_random(dim, 100);
    ///
    /// let model_options = ModelOptions::<NIW>::default(dim);
    /// let mut model = Model::from_options(model_options);
    ///
    /// let fit_options = FitOptions::default();
    /// let callback = MonitoringCallback::from_data(
    ///        EvalData::from_sample(&x, None, 1000)
    /// );
    /// let mut local = ShardedState::from_data(x, 4);
    ///
    /// model.fit_worker(
    ///     &mut local,
    ///     &fit_options,
    ///     Some(callback)
    /// );
    /// ```
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

                    if !split_idx.is_empty() {
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


    /// Predict the cluster labels for the data and their confidence.
    ///
    /// # Arguments
    ///
    /// * `data`: The data to predict the labels for. (n_features, n_samples)
    ///
    /// # Returns
    ///
    /// * `confidence`: The confidence/probability of the predicted labels. (n_samples,)
    /// * `labels`: The predicted labels for the data. (n_samples,)
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{DMatrix, RowDVector};
    /// use mixturs::{FitOptions, Model, ModelOptions, MonitoringCallback, NIW};
    /// use mixturs::state::GlobalState;
    ///
    /// let dim = 2;
    /// let x = DMatrix::new_random(dim, 100);
    ///
    /// let model_options = ModelOptions::<NIW>::default(dim);
    /// let mut model = Model::from_options(model_options);
    ///
    /// let fit_options = FitOptions::default();
    /// model.fit(x.clone_owned(), &fit_options, None::<MonitoringCallback<GlobalState<NIW>>>);
    ///
    /// let (confidence, labels) = model.predict(x);
    /// ```
    pub fn predict(
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


