use nalgebra::{DVector};
use rand::distributions::{Distribution};
use rand::Rng;
use statrs::distribution::{Dirichlet};
use crate::clusters::{ClusterParams, SuperClusterParams};
use crate::local::LocalStats;
use crate::options::{ModelOptions, OutlierRemoval};
use crate::stats::dp::stick_breaking_sample;
use crate::stats::GaussianPrior;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlobalState<P: GaussianPrior> {
    pub clusters: Vec<SuperClusterParams<P>>,
    pub weights: Vec<f64>,
}

impl<P: GaussianPrior> GlobalState<P> {
    pub fn from_init<R: Rng>(
        data_stats: &P::SuffStats,
        n_clusters: usize,
        options: &ModelOptions<P>,
        rng: &mut R,
    ) -> Self {
        let mut clusters = Vec::new();
        let mut points_count = Vec::new();

        for k in 0..n_clusters + options.outlier.is_some() as usize {
            let (prior, stats) = match (k, &options.outlier) {
                (0, Some(OutlierRemoval { dist, .. })) => (dist, data_stats.clone()), // TODO: use data stats
                _ => (&options.data_dist, P::SuffStats::default())
            };

            let dist = P::sample(prior, rng);
            let prim = ClusterParams::new(
                prior.clone(),
                prior.clone(),
                stats,
                dist,
            );
            let cluster = SuperClusterParams::from_split_params(prim, options.alpha, options.burnout_period, rng);
            points_count.push((cluster.n_points() as f64).max(1.0));
            clusters.push(cluster);
        }

        let weights = if let Some(OutlierRemoval { weight, .. }) = &options.outlier {
            stick_breaking_sample(&points_count[1..], *weight, rng)
        } else {
            stick_breaking_sample(&points_count[..], 0.0, rng)
        };

        Self {
            clusters,
            weights,
        }
    }

    pub fn n_clusters(&self) -> usize {
        self.clusters.len()
    }

    pub fn n_points(&self) -> usize {
        self.clusters.iter().map(|c| c.n_points()).sum()
    }

    pub fn update_sample_clusters<R: Rng>(
        global: &mut GlobalState<P>,
        options: &ModelOptions<P>,
        rng: &mut R,
    ) {
        let mut points_count = Vec::new();
        for cluster in global.clusters.iter_mut() {
            let (prim, aux, weights) = cluster.sample(options.alpha, rng);

            cluster.prim.dist = prim;
            for (k, dist) in aux.into_iter().enumerate() {
                cluster.aux[k].dist = dist;
            }

            cluster.weights = weights;

            cluster.ll_history.push(
                cluster.aux.iter().map(|c| c.marginal_log_likelihood()).sum::<f64>()
            );
            cluster.splittable = cluster.ll_history.converged(options.burnout_period);
            points_count.push(cluster.n_points() as f64);
        }

        global.weights = if let Some(OutlierRemoval { weight, .. }) = &options.outlier {
            stick_breaking_sample(&points_count[1..], *weight, rng)
        } else {
            stick_breaking_sample(&points_count[..], 0.0, rng)
        };
    }

    pub fn update_clusters_post(
        global: &mut GlobalState<P>,
        stats: LocalStats<P>
    ) {
        for (k, stats) in stats.0.into_iter().enumerate() {
            global.clusters[k].update_post(stats)
        }
    }

    pub fn collect_bad_clusters(
        global: &mut GlobalState<P>,
    ) -> Vec<usize> {
        let mut bad_clusters = Vec::new();
        for (k, cluster) in global.clusters.iter_mut().enumerate() {
            if cluster.aux.iter().any(|c| c.n_points() == 0) {
                cluster.ll_history.clear();
                cluster.splittable = false;
                bad_clusters.push(k);
            }
        }

        bad_clusters
    }

    pub fn update_remove_empty_clusters(
        global: &mut GlobalState<P>,
        options: &ModelOptions<P>,
    ) -> Vec<usize> {
        let mut new_clusters = Vec::new();
        let mut removed_cluster_idx = Vec::new();

        for (k, cluster) in global.clusters.iter().enumerate() {
            if cluster.n_points() > 0
                || (options.outlier.is_some() && k == 0)
                || (options.outlier.is_some() && k == 1 && global.n_clusters() == 2)
            {
                new_clusters.push(cluster.clone());
            } else {
                removed_cluster_idx.push(k);
            }
        }
        global.clusters = new_clusters;
        removed_cluster_idx
    }
}