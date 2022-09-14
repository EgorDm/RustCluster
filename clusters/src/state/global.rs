use rand::Rng;
use statrs::distribution::MultivariateNormal;
use serde::{Serialize, Deserialize};
use crate::params::clusters::{ClusterParams, SuperClusterParams, SuperClusterStats};
use crate::params::options::{ModelOptions, OutlierRemoval};
use crate::params::thin::ThinParams;
use crate::stats::{NormalConjugatePrior, SplitMerge, stick_breaking_sample};
use crate::state::GlobalWorker;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct GlobalState<P: NormalConjugatePrior> {
    pub clusters: Vec<SuperClusterParams<P>>,
    pub weights: Vec<f64>,
}

impl<P: NormalConjugatePrior> GlobalState<P> {
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
}

impl<P: NormalConjugatePrior> GlobalWorker<P> for GlobalState<P> {
    fn n_clusters(&self) -> usize {
        self.clusters.len()
    }

    fn n_points(&self) -> usize {
        self.clusters.iter().map(|c| c.n_points()).sum()
    }

    fn update_clusters_post(&mut self, stats: Vec<SuperClusterStats<P>>) {
        for (k, stats) in stats.into_iter().enumerate() {
            self.clusters[k].update_post(stats)
        }
    }

    fn update_sample_clusters<R: Rng>(&mut self, options: &ModelOptions<P>, rng: &mut R) {
        let mut points_count = Vec::new();
        for cluster in self.clusters.iter_mut() {
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

        self.weights = if let Some(OutlierRemoval { weight, .. }) = &options.outlier {
            stick_breaking_sample(&points_count[1..], *weight, rng)
        } else {
            stick_breaking_sample(&points_count[..], 0.0, rng)
        };
    }

    fn collect_bad_clusters(&mut self) -> Vec<usize> {
        let mut bad_clusters = Vec::new();
        for (k, cluster) in self.clusters.iter_mut().enumerate() {
            if cluster.aux.iter().any(|c| c.n_points() == 0) {
                cluster.ll_history.clear();
                cluster.splittable = false;
                bad_clusters.push(k);
            }
        }
        bad_clusters
    }

    fn collect_remove_clusters(&mut self, options: &ModelOptions<P>) -> Vec<usize> {
        let mut new_clusters = Vec::new();
        let mut removed_cluster_idx = Vec::new();

        for (k, cluster) in self.clusters.iter().enumerate() {
            if cluster.n_points() > 0
                || (options.outlier.is_some() && k == 0)
                || (options.outlier.is_some() && k == 1 && GlobalWorker::n_clusters(self) == 2)
            {
                new_clusters.push(cluster.clone());
            } else {
                removed_cluster_idx.push(k);
            }
        }
        self.clusters = new_clusters;
        removed_cluster_idx
    }

    fn check_and_split<R: Rng>(&mut self, options: &ModelOptions<P>, rng: &mut R) -> Vec<(usize, usize)> {
        let mut decisions = vec![false; GlobalWorker::n_clusters(self)];
        for (k, cluster) in self.clusters.iter().enumerate() {
            if k == 0 && options.outlier.is_some() {
                continue;
            }

            if cluster.splittable && cluster.n_points() > 1 {
                decisions[k] = SplitMerge::should_split(cluster, options.alpha, rng);
            }
        }

        let mut split_idx = Vec::new();
        for (k, _) in decisions.into_iter().enumerate().filter(|(_, split)| *split) {
            let new_idx = GlobalWorker::n_clusters(self);

            // Split cluster parameters
            self.clusters.push(self.clusters[k].clone());
            let cluster = &self.clusters[k];

            let cluster_l = SuperClusterParams::from_split_params(cluster.aux[0].clone(), options.alpha, options.burnout_period, rng);
            let cluster_r = SuperClusterParams::from_split_params(cluster.aux[1].clone(), options.alpha, options.burnout_period, rng);

            self.clusters[k] = cluster_l;
            self.clusters[new_idx] = cluster_r;

            split_idx.push((k, new_idx));
        }

        split_idx
    }

    fn check_and_merge<R: Rng>(&mut self, options: &ModelOptions<P>, rng: &mut R) -> Vec<(usize, usize)> {
        let mut decisions = Vec::new();
        for ki in 0..GlobalWorker::n_clusters(self) {
            if ki == 0 && options.outlier.is_some() {
                continue;
            }

            for kj in ki + 1..GlobalWorker::n_clusters(self) {
                let (cluster_i, cluster_j) = (&self.clusters[ki], &self.clusters[kj]);

                if !cluster_i.splittable || !cluster_j.splittable
                    || cluster_i.n_points() == 0 || cluster_j.n_points() == 0 {
                    continue;
                }

                if !SplitMerge::should_merge(&cluster_i.prim, &cluster_j.prim, options.alpha, rng) {
                    continue;
                }

                let cluster = SuperClusterParams::from_merge_params(
                    cluster_i.prim.clone(), cluster_j.prim.clone(),
                    options.alpha, options.burnout_period, rng,
                );
                self.clusters[ki] = cluster;
                self.clusters[kj].prim.stats = P::SuffStats::default();
                self.clusters[kj].splittable = false;
                decisions.push((ki, kj));
            }
        }

        decisions
    }
}

impl<P: NormalConjugatePrior> ThinParams for GlobalState<P> {
    fn n_clusters(&self) -> usize {
        self.clusters.len()
    }

    fn cluster_dist(&self, cluster_id: usize) -> &MultivariateNormal {
        &self.clusters[cluster_id].prim.dist
    }

    fn cluster_weights(&self) -> &[f64] {
        self.weights.as_slice()
    }

    fn cluster_aux_dist(&self, cluster_id: usize, aux_id: usize) -> &MultivariateNormal {
        &self.clusters[cluster_id].aux[aux_id].dist
    }

    fn cluster_aux_weights(&self, cluster_id: usize) -> &[f64; 2] {
        &self.clusters[cluster_id].weights
    }
}
