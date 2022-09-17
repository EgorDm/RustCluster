use itertools::izip;
use std::marker::PhantomData;
use nalgebra::{DMatrix, RowDVector};
use rand::Rng;
use crate::stats::{FromData, NormalConjugatePrior};
use crate::utils::{col_scatter, group_sort};
use crate::utils::Iterutils;
use crate::params::clusters::{SuperClusterStats};
use crate::params::thin::{AuxMixtureParams, hard_assignment, MixtureParams, soft_assignment, SuperMixtureParams, ThinParams};
use crate::state::LocalWorker;


/// Local state performs all computations on the locally on the data.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalState<P: NormalConjugatePrior> {
    pub data: DMatrix<f64>,
    pub labels: RowDVector<usize>,
    pub labels_aux: RowDVector<usize>,
    _phantoms: PhantomData<fn() -> P>,
}

impl<P: NormalConjugatePrior> LocalState<P> {
    /// Create a new local state.
    ///
    /// # Arguments
    ///
    /// * `data`: Data points matrix (n_dims, n_samples).
    /// * `labels`: Primary cluster labels.
    /// * `labels_aux`: Auxiliary cluster labels.
    pub fn new(
        data: DMatrix<f64>,
        labels: RowDVector<usize>,
        labels_aux: RowDVector<usize>,
    ) -> Self {
        Self { data, labels, labels_aux, _phantoms: PhantomData }
    }

    /// Create a new local state from data
    /// all labels are set to 0.
    ///
    /// # Arguments
    ///
    /// * `data`: Data points matrix (n_dims, n_samples).
    pub fn from_data(data: DMatrix<f64>) -> Self {
        let labels = RowDVector::zeros(data.ncols());
        let labels_aux = RowDVector::zeros(data.ncols());
        Self::new(data, labels, labels_aux)
    }

    /// Number of points in the data.
    pub fn n_points(&self) -> usize {
        self.data.ncols()
    }

    /// Samples primary labels given cluster parameters.
    ///
    /// # Arguments
    ///
    /// * `params`: The cluster parameters
    /// * `hard_assign`: Whether to perform hard assignment or soft assignment (i.e. sampling strategy)
    /// * `rng`: The random number generator
    ///
    pub fn apply_sample_labels_prim(
        &mut self,
        params: &impl ThinParams,
        hard_assign: bool,
        rng: &mut impl Rng,
    ) {
        // Calculate log likelihood for each point
        let ll = SuperMixtureParams(params).log_likelihood(self.data.clone_owned());

        // Sample labels
        if hard_assign {
            hard_assignment(&ll, self.labels.as_mut_slice());
        } else {
            soft_assignment(ll, self.labels.as_mut_slice(), rng);
        }
    }

    /// Samples auxiliary labels given cluster parameters and their assignment to primary clusters.
    ///
    /// # Arguments
    ///
    /// * `params`: The cluster parameters
    /// * `hard_assign`: Whether to perform hard assignment or soft assignment (i.e. sampling strategy)
    /// * `rng`: The random number generator
    ///
    pub fn apply_sample_labels_aux(
        &mut self,
        params: &impl ThinParams,
        hard_assign: bool,
        rng: &mut impl Rng,
    ) {
        // Split data points into contiguous blocks (indexes only for now)
        let (indices, offsets) = self.sorted_indices(params.n_clusters());

        // Calculate log likelihood for each point given its cluster
        // done by grouping the data points in blocks with the same label
        let mut ll = DMatrix::zeros(2, self.n_points());
        for prim in 0..params.n_clusters() {
            let indices = &indices[offsets[prim * 2]..offsets[(prim + 1) * 2]];
            let block = self.data.select_columns(indices);

            let block_ll = AuxMixtureParams(params, prim).log_likelihood(block);
            col_scatter(&mut ll, indices, &block_ll);
        }

        // Sample labels
        if hard_assign {
            hard_assignment(&ll, self.labels_aux.as_mut_slice());
        } else {
            soft_assignment(ll, self.labels_aux.as_mut_slice(), rng);
        }
    }

    /// Group sorts the data points by primary and auxiliary labels in O(n)
    ///
    /// # Arguments
    ///
    /// * `n_clusters`: Number of primary clusters
    ///
    /// # Returns
    ///
    /// Tuple containing:
    /// * Sorted indices of to the data points
    /// * Offsets of each cluster block
    ///
    /// Cluster blocks are labelled as block_id = prim * 2 + aux
    fn sorted_indices(&self, n_clusters: usize) -> (Vec<usize>, Vec<usize>) {
        // Split data points into contiguous blocks
        let n_blocks = n_clusters * 2;
        let counts = izip!(&self.labels, &self.labels_aux)
            .map(|(&prim, &aux)| prim * 2 + aux)
            .bincounts(n_blocks);
        group_sort(
            &counts,
            izip!(&self.labels, &self.labels_aux),
            |(&prim, &aux)| prim * 2 + aux,
        )
    }
}

impl<P: NormalConjugatePrior> LocalWorker<P> for LocalState<P> {
    fn init<R: Rng + Clone + Send + Sync>(&mut self, n_clusters: usize, rng: &mut R) {
        self.labels.apply(|v| *v = rng.gen_range(0..n_clusters));
        self.labels_aux.apply(|v| *v = rng.gen_range(0..2));
    }

    fn n_points(&self) -> usize {
        self.data.ncols()
    }

    fn collect_data_stats(&self) -> P::SuffStats {
        P::SuffStats::from_data(&self.data)
    }

    fn collect_cluster_stats(&self, n_clusters: usize) -> Vec<SuperClusterStats<P>> {
        // Split data points into contiguous blocks (indexes only for now)
        let (indices, offsets) = self.sorted_indices(n_clusters);

        // Gather data from sorted indices
        let data = self.data.select_columns(&indices);

        let mut stats = Vec::with_capacity(n_clusters);
        for i in (0..n_clusters * 2).step_by(2) {
            stats.push(SuperClusterStats {
                prim: P::SuffStats::from_data(
                    &data.columns_range(offsets[i]..offsets[i + 2])
                ),
                aux: [
                    P::SuffStats::from_data(&data.columns_range(offsets[i]..offsets[i + 1])),
                    P::SuffStats::from_data(&data.columns_range(offsets[i + 1]..offsets[i + 2])),
                ],
            });
        }

        stats
    }

    fn apply_label_sampling<R: Rng + Clone + Send + Sync>(
        &mut self,
        params: &impl ThinParams,
        hard_assign: bool,
        rng: &mut R,
    ) {
        self.apply_sample_labels_prim(params, hard_assign, rng);
        self.apply_sample_labels_aux(params, false, rng);
    }

    fn apply_cluster_reset<R: Rng + Clone + Send + Sync>(
        &mut self,
        cluster_ids: &[usize],
        rng: &mut R,
    ) {
        for &k in cluster_ids {
            for i in 0..self.n_points() {
                if self.labels[i] == k {
                    self.labels_aux[i] = rng.gen_range(0..2);
                }
            }
        }
    }

    fn apply_cluster_remove(
        &mut self,
        cluster_ids: &[usize],
    ) {
        for (removed, &k) in cluster_ids.iter().enumerate() {
            for l in self.labels.iter_mut() {
                if *l > k - removed {
                    *l -= 1;
                }
            }
        }
    }

    fn apply_split<R: Rng + Clone + Send + Sync>(
        &mut self,
        split_decisions: &[(usize, usize)],
        rng: &mut R,
    ) {
        for &(kl, kr) in split_decisions {
            for (label, label_aux) in izip!(self.labels.iter_mut(), self.labels_aux.iter_mut()) {
                if *label == kl {
                    *label = if *label_aux == 0 { kl } else { kr };
                    *label_aux = rng.gen_range(0..2);
                }
            }
        }
    }

    fn apply_merge(
        &mut self,
        merge_decisions: &[(usize, usize)],
    ) {
        for &(kl, kr) in merge_decisions {
            for (label, label_aux) in izip!(self.labels.iter_mut(), self.labels_aux.iter_mut()) {
                if *label == kl {
                    *label_aux = 0;
                } else if *label == kr {
                    *label = kl;
                    *label_aux = 2;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector, RowDVector};
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use statrs::distribution::MultivariateNormal;
    use crate::params::clusters::SuperClusterStats;
    use crate::params::thin::{OwnedThinParams, ThinParams};
    use crate::state::{LocalState, LocalWorker};
    use crate::stats::{FromData, NIW, NIWStats};
    use crate::stats::tests::test_almost_mat;

    #[test]
    fn test_collect_stats() {
        let mut rng = StdRng::seed_from_u64(42);
        let data = DMatrix::from_fn(2, 120, |_, _| rng.gen_range(0.0..1.0));
        let labels = RowDVector::from_fn(120, |_, i| i / 30);
        let labels_aux = RowDVector::from_fn(120, |_, i| i / 15 % 2);

        let local = super::LocalState::<NIW>::new(data.clone(), labels, labels_aux);
        let stats = local.collect_cluster_stats(4);
        for (i, SuperClusterStats { prim, aux }) in stats.into_iter().enumerate() {
            let prim_og = NIWStats::from_data(&data.columns_range(i * 30..(i + 1) * 30).into_owned());
            let aux_og = [
                NIWStats::from_data(&data.columns_range(i * 30..i * 30 + 15).into_owned()),
                NIWStats::from_data(&data.columns_range(i * 30 + 15..i * 30 + 30).into_owned()),
            ];

            assert_eq!(prim.n_points, prim_og.n_points);
            test_almost_mat(&prim.mean_sum, &prim_og.mean_sum, 1e-4);
            test_almost_mat(&prim.cov_sum, &prim_og.cov_sum, 1e-4);

            for a in 0..2 {
                test_almost_mat(&aux[a].mean_sum, &aux_og[a].mean_sum, 1e-4);
                test_almost_mat(&aux[a].cov_sum, &aux_og[a].cov_sum, 1e-4);
                assert_eq!(aux[a].n_points, aux_og[a].n_points);
            }
        }
    }

    #[test]
    fn test_sorted_indices() {
        let mut rng = StdRng::seed_from_u64(42);
        let data = DMatrix::from_fn(2, 120, |_, _| rng.gen_range(0.0..1.0));
        let labels = RowDVector::from_fn(120, |_, i| (i % 8) / 2);
        let labels_aux = RowDVector::from_fn(120, |_, i| i % 2);

        let local = LocalState::<NIW>::new(data.clone(), labels.clone_owned(), labels_aux.clone_owned());

        let (indices, offsets) = local.sorted_indices(4);
        assert_eq!(offsets, vec![0, 15, 30, 45, 60, 75, 90, 105, 120]);
        for (i, &index) in indices.iter().enumerate() {
            assert_eq!(labels[index], i / 30);
            assert_eq!(labels_aux[index], (i / 15) % 2);
        }
    }

    #[test]
    fn test_sample_labels() {
        let mut rng = StdRng::seed_from_u64(42);
        let params = OwnedThinParams {
            clusters: vec![
                MultivariateNormal::new(
                    DVector::from_vec(vec![0.0, 0.0]).data.into(),
                    DMatrix::from_diagonal_element(2, 2, 1.0).data.into(),
                ).unwrap(),
                MultivariateNormal::new(
                    DVector::from_vec(vec![1.0, 1.0]).data.into(),
                    DMatrix::from_diagonal_element(2, 2, 1.0).data.into(),
                ).unwrap(),
            ],
            cluster_weights: vec![0.5, 0.5],
            clusters_aux: vec![],
            cluster_weights_aux: vec![],
        };

        let data = DMatrix::from_fn(2, 120, |_, _| rng.gen_range(0.0..1.0));
        let labels = RowDVector::zeros(120);
        let labels_aux = RowDVector::zeros(120);

        let mut local = LocalState::<NIW>::new(data.clone(), labels, labels_aux);
        local.apply_sample_labels_prim(&params, true, &mut rng);

        for (i, point) in data.column_iter().enumerate() {
            assert_eq!(
                local.labels[i],
                if (point - DVector::from_vec(vec![0.0, 0.0])).norm() < (point - DVector::from_vec(vec![1.0, 1.0])).norm() {
                    0
                } else {
                    1
                }
            );
        }
    }

    #[test]
    fn test_sample_labels_aux() {
        let mut rng = StdRng::seed_from_u64(42);
        let params = OwnedThinParams {
            clusters: vec![
                MultivariateNormal::new(
                    DVector::from_vec(vec![0.0, 0.0]).data.into(),
                    DMatrix::from_diagonal_element(2, 2, 1.0).data.into(),
                ).unwrap(),
                MultivariateNormal::new(
                    DVector::from_vec(vec![1.0, 1.0]).data.into(),
                    DMatrix::from_diagonal_element(2, 2, 1.0).data.into(),
                ).unwrap(),
            ],
            cluster_weights: vec![0.5, 0.5],
            clusters_aux: vec![
                [
                    MultivariateNormal::new(
                        DVector::from_vec(vec![0.0, 0.0]).data.into(),
                        DMatrix::from_diagonal_element(2, 2, 1.0).data.into(),
                    ).unwrap(),
                    MultivariateNormal::new(
                        DVector::from_vec(vec![1.0, 1.0]).data.into(),
                        DMatrix::from_diagonal_element(2, 2, 1.0).data.into(),
                    ).unwrap(),
                ],
                [
                    MultivariateNormal::new(
                        DVector::from_vec(vec![0.0, 4.0]).data.into(),
                        DMatrix::from_diagonal_element(2, 2, 1.0).data.into(),
                    ).unwrap(),
                    MultivariateNormal::new(
                        DVector::from_vec(vec![4.0, 0.0]).data.into(),
                        DMatrix::from_diagonal_element(2, 2, 1.0).data.into(),
                    ).unwrap(),
                ],
            ],
            cluster_weights_aux: vec![
                [0.5, 0.5],
                [0.5, 0.5],
            ],
        };

        let data = DMatrix::from_fn(2, 120, |_, _| rng.gen_range(0.0..1.0));
        let labels = RowDVector::from_fn(120, |_, _| rng.gen_range(0..2));
        let labels_aux = RowDVector::zeros(120);

        let mut local = LocalState::<NIW>::new(data.clone(), labels.clone_owned(), labels_aux);
        local.apply_sample_labels_aux(&params, true, &mut rng);

        for (i, point) in data.column_iter().enumerate() {
            let label = labels[i];

            assert_eq!(
                local.labels_aux[i],
                if (point - params.cluster_aux_dist(label, 0).mu()).norm()
                    < (point - params.cluster_aux_dist(label, 1).mu()).norm() {
                    0
                } else {
                    1
                }
            );
        }
    }
}