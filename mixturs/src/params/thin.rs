use itertools::repeat_n;
use nalgebra::{DMatrix, RowDVector};
use rand::Rng;
use statrs::distribution::MultivariateNormal;
use crate::stats::ContinuousBatchwise;
use crate::utils::{col_normalize_log_weights, replacement_sampling_weighted};


pub trait ThinParams: Clone + Send + Sync {
    /// Number of clusters.
    fn n_clusters(&self) -> usize;

    /// Distribution of the primary cluster.
    fn cluster_dist(&self, cluster_id: usize) -> &MultivariateNormal;

    /// Weights of the primary clusters.
    fn cluster_weights(&self) -> &[f64];

    /// Distribution of the auxiliary clusters given the primary cluster.
    fn cluster_aux_dist(&self, cluster_id: usize, aux_id: usize) -> &MultivariateNormal;

    /// Weights of the auxiliary clusters given the primary cluster.
    fn cluster_aux_weights(&self, cluster_id: usize) -> &[f64; 2];

    /// Number of parameters in the model.
    fn n_params(&self) -> usize {
        let dim = self.cluster_dist(0).mu().len();
        self.n_clusters() * (dim * dim + dim + 1)
    }
}

#[derive(Debug, Clone, PartialEq)]
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

/// Selects super cluster params from thin params
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

/// Selects auxiliary cluster params from thin params for a given super cluster
pub struct AuxMixtureParams<'a, D: ThinParams>(pub(crate) &'a D, pub(crate) usize);

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
    /// Number of clusters.
    fn n_clusters(&self) -> usize;

    /// Distribution of the primary cluster.
    fn dist(&self, cluster_id: usize) -> &MultivariateNormal;

    /// Weights of the primary clusters.
    fn weights(&self) -> &[f64];

    /// Log-likelihood of the data points (columns) given the model.
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

    /// Predict the cluster labels for the data points (columns).
    fn predict(&self, data: DMatrix<f64>) -> (DMatrix<f64>, RowDVector<usize>) {
        let mut labels = RowDVector::zeros(data.ncols());
        let log_likelihood = self.log_likelihood(data);
        hard_assignment(&log_likelihood, labels.as_mut_slice());
        let probs = col_normalize_log_weights(log_likelihood);

        (probs, labels)
    }
}


/// Assigns each column of `log_likelihood` to the cluster with the highest log probability.
///
/// # Arguments
///
/// * `log_likelihood`: A matrix of log probabilities of shape (n_clusters, n_samples)
/// * `labels`: A mutable vector of length `n_samples` the cluster assignments will be written to.
///
/// # Examples
///
/// ```
/// use mixturs::params::thin::hard_assignment;
/// use nalgebra::{DMatrix, RowDVector};
///
/// let log_likelihood = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let mut labels = RowDVector::zeros(3);
/// hard_assignment(&log_likelihood, labels.as_mut_slice());
/// assert_eq!(labels, RowDVector::from_row_slice(&[1, 1, 1]));
/// ```
pub fn hard_assignment(
    log_likelihood: &DMatrix<f64>,
    labels: &mut [usize],
) {
    for (i, row) in log_likelihood.column_iter().enumerate() {
        labels[i] = row.argmax().0;
    }
}


/// Assigns each column of `log_likelihood` to a cluster according to the probability distribution
/// defined by the log probabilities.
///
/// # Arguments
///
/// * `log_likelihood`: A matrix of log probabilities of shape (n_clusters, n_samples)
/// * `labels`: A mutable vector of length `n_samples` the cluster assignments will be written to.
/// * `rng`: A random number generator.
///
/// # Examples
///
/// ```
/// use mixturs::params::thin::soft_assignment;
/// use nalgebra::{DMatrix, RowDVector};
///
/// let log_likelihood = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let mut labels = RowDVector::zeros(3);
/// let mut rng = rand::thread_rng();
/// soft_assignment(log_likelihood, labels.as_mut_slice(), &mut rng);
/// ```
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