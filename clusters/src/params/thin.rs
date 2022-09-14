use itertools::repeat_n;
use nalgebra::{DMatrix, RowDVector};
use rand::Rng;
use serde::de::DeserializeOwned;
use serde::{Serialize, Deserialize};
use statrs::distribution::MultivariateNormal;
use crate::stats::ContinuousBatchwise;
use crate::utils::{col_normalize_log_weights, replacement_sampling_weighted};

pub trait ThinParams: Clone + Send + Sync + Serialize + DeserializeOwned {
    fn n_clusters(&self) -> usize;

    fn cluster_dist(&self, cluster_id: usize) -> &MultivariateNormal;

    fn cluster_weights(&self) -> &[f64];

    fn cluster_aux_dist(&self, cluster_id: usize, aux_id: usize) -> &MultivariateNormal;

    fn cluster_aux_weights(&self, cluster_id: usize) -> &[f64; 2];

    fn n_params(&self) -> usize {
        let dim = self.cluster_dist(0).mu().len();
        self.n_clusters() * (dim * dim + dim + 1)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OwnedThinParams {
    pub clusters: Vec<MultivariateNormal>,
    pub cluster_weights: Vec<f64>,
    pub clusters_aux: Vec<[MultivariateNormal; 2]>,
    pub cluster_weights_aux: Vec<[f64; 2]>,
}

impl ThinParams for OwnedThinParams {
    fn n_clusters(&self) -> usize {
        self.clusters.len()
    }

    fn cluster_dist(&self, cluster_id: usize) -> &MultivariateNormal {
        &self.clusters[cluster_id]
    }

    fn cluster_weights(&self) -> &[f64] {
        &self.cluster_weights
    }

    fn cluster_aux_dist(&self, cluster_id: usize, aux_id: usize) -> &MultivariateNormal {
        &self.clusters_aux[cluster_id][aux_id]
    }

    fn cluster_aux_weights(&self, cluster_id: usize) -> &[f64; 2] {
        &self.cluster_weights_aux[cluster_id]
    }
}

pub struct SuperMixtureParams<'a, D: ThinParams>(pub(crate) &'a D);

impl<'a, D: ThinParams> MixtureParams for SuperMixtureParams<'a, D> {
    fn n_clusters(&self) -> usize {
        self.0.n_clusters()
    }

    fn dist(&self, cluster_id: usize) -> &MultivariateNormal {
        self.0.cluster_dist(cluster_id)
    }

    fn weights(&self) -> &[f64] {
        self.0.cluster_weights()
    }
}

pub struct AuxMixtureParams<'a, D: ThinParams>(pub (crate) &'a D, pub(crate) usize);

impl<'a, D: ThinParams> MixtureParams for AuxMixtureParams<'a, D> {
    fn n_clusters(&self) -> usize {
        2
    }

    fn dist(&self, cluster_id: usize) -> &MultivariateNormal {
        self.0.cluster_aux_dist(self.1, cluster_id)
    }

    fn weights(&self) -> &[f64] {
        self.0.cluster_aux_weights(self.1)
    }
}

pub trait MixtureParams {
    fn n_clusters(&self) -> usize;

    fn dist(&self, cluster_id: usize) -> &MultivariateNormal;

    fn weights(&self) -> &[f64];

    fn log_likelihood(&self, data: DMatrix<f64>) -> DMatrix<f64> {
        let mut ll = DMatrix::zeros(self.n_clusters(), data.ncols());
        // Add cluster log probabilities
        for (cluster_id, data) in repeat_n(data, self.n_clusters()).enumerate() {
            ll.row_mut(cluster_id).copy_from_slice(
                self.dist(cluster_id)
                    .batchwise_ln_pdf(data)
                    .as_slice()
            );
        }

        // Add mixture weights
        let weights = self.weights();
        for (prim, mut row) in ll.row_iter_mut().enumerate() {
            let ln_weight = weights[prim].ln();
            row.apply(|x| *x += ln_weight);
        }

        ll
    }

    fn predict(&self, data: DMatrix<f64>) -> (DMatrix<f64>, RowDVector<usize>) {
        let mut labels = RowDVector::zeros(data.ncols());
        let log_likelihood = self.log_likelihood(data);
        hard_assignment(&log_likelihood, labels.as_mut_slice());
        let probs = col_normalize_log_weights(log_likelihood);

        (probs, labels)
    }
}


pub fn hard_assignment(
    log_likelihood: &DMatrix<f64>,
    labels: &mut [usize],
) {
    for (i, row) in log_likelihood.column_iter().enumerate() {
        labels[i] = row.argmax().0;
    }
}

pub fn soft_assignment(
    log_likelihood: DMatrix<f64>,
    labels: &mut [usize],
    rng: &mut impl Rng,
) {
    let probs = col_normalize_log_weights(log_likelihood);
    for (i, col) in probs.column_iter().enumerate() {
        replacement_sampling_weighted(rng, col.into_iter().cloned(), &mut labels[i..=i]);
    }
}