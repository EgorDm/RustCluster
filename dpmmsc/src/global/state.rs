use nalgebra::{DVector};
use rand::distributions::{Distribution};
use rand::Rng;
use statrs::distribution::{Dirichlet};
use crate::clusters::SuperClusterParams;
use crate::priors::{GaussianPrior};

#[derive(Debug, Clone, PartialEq)]
pub struct ModelOptions<P: GaussianPrior> {
    pub prior_dist: P::HyperParams,
    pub alpha: f64,
    pub dim: usize,
    pub burnout_period: usize,
    pub outlier_mod: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlobalState<P: GaussianPrior> {
    pub clusters: Vec<SuperClusterParams<P>>,
    pub weights: Vec<f64>,
}

impl<P: GaussianPrior> GlobalState<P> {
    pub fn n_clusters(&self) -> usize {
        self.clusters.len()
    }

    pub fn n_points(&self) -> usize {
        self.clusters.iter().map(|c| c.n_points()).sum()
    }

    pub fn sample<R: Rng>(&self, rng: &mut R) -> DVector<f64> {
        let dir = Dirichlet::new(DVector::from_element(self.n_clusters(), self.n_points() as f64)).unwrap();
        dir.sample(rng)
    }

    pub fn update_sample_clusters<R: Rng>(
        global: &mut GlobalState<P>,
        options: &ModelOptions<P>,
        rng: &mut R,
    ) {
        for cluster in global.clusters.iter_mut() {
            let (prim, aux, weights) = cluster.sample(options.alpha, rng);

            cluster.prim.dist = prim;
            for (k, dist) in aux.into_iter().enumerate() {
                cluster.aux[k].dist = dist;
            }
            cluster.weights = weights;
            cluster.update_history(
                cluster.aux.iter().map(|c| c.marginal_log_likelihood()).sum::<f64>()
            );
            cluster.splittable = cluster.converged(options.burnout_period);
        }

        let n_clusters = global.n_clusters();
        let weights = global.sample(rng) * (1.0 - options.outlier_mod);
        global.weights[0..n_clusters].copy_from_slice(weights.as_slice());
        global.weights[n_clusters] = options.outlier_mod;
    }

    pub fn collect_bad_clusters(
        global: &mut GlobalState<P>,
    ) -> Vec<usize> {
        let mut bad_clusters = Vec::new();
        for (k, cluster) in global.clusters.iter_mut().enumerate() {
            if cluster.aux.iter().any(|c| c.n_points() == 0) {
                cluster.ll_history = vec![f64::NEG_INFINITY; cluster.ll_history.len()];
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
                || (options.outlier_mod > 0.0 && k == 1)
                || (options.outlier_mod > 0.0 && k == 2 && global.n_clusters() == 2) {
                new_clusters.push(cluster.clone());
            } else {
                removed_cluster_idx.push(k);
            }
        }
        global.clusters = new_clusters;
        removed_cluster_idx
    }
}