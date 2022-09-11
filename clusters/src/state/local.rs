use std::iter::Sum;
use itertools::{Itertools, izip};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Range};
use std::vec::IntoIter;
use nalgebra::{Dim, DMatrix, DVector, Dynamic, Matrix, RowDVector, RowVector, Storage};
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use remoc::RemoteSend;
use statrs::distribution::{Continuous};
use crate::clusters::{SuperClusterStats, ThinParams, ThinStats};
use crate::options::ModelOptions;
use crate::stats::{ContinuousBatchwise, FromData, NormalConjugatePrior, NIW, SufficientStats};
use crate::utils::{col_normalize_log_weights, col_scatter, group_sort, replacement_sampling_weighted, row_normalize_log_weights};
use crate::utils::Iterutils;
use serde::{Serialize, Deserialize};
use crate::state::LocalWorker;


#[derive(Debug, Clone, PartialEq)]
pub struct LocalState<P: NormalConjugatePrior> {
    pub data: DMatrix<f64>,
    pub labels: RowDVector<usize>,
    pub labels_aux: RowDVector<usize>,
    _phantoms: PhantomData<fn() -> P>,
}

impl<P: NormalConjugatePrior> LocalState<P> {
    pub fn new(
        data: DMatrix<f64>,
        labels: RowDVector<usize>,
        labels_aux: RowDVector<usize>,
    ) -> Self {
        Self { data, labels, labels_aux, _phantoms: PhantomData }
    }

    pub fn from_data(data: DMatrix<f64>) -> Self {
        let labels = RowDVector::zeros(data.ncols());
        let labels_aux = RowDVector::zeros(data.ncols());
        Self::new(data, labels, labels_aux)
    }

    pub fn n_points(&self) -> usize {
        self.data.ncols()
    }

    pub fn apply_sample_labels_prim(
        &mut self,
        params: &impl ThinParams,
        hard_assignment: bool,
        rng: &mut impl Rng,
    ) {
        // Calculate log likelihood for each point
        let ln_weights = params.cluster_weights().iter().map(|w| w.ln()).collect::<Vec<_>>();
        let mut ll = DMatrix::zeros(params.n_clusters(), self.n_points());
        for k in 0..params.n_clusters() {
            ll.row_mut(k).copy_from_slice(
                params.cluster_dist(k).batchwise_ln_pdf(self.data.clone_owned()).as_slice()
            );
        }
        for (k, mut row) in ll.row_iter_mut().enumerate() {
            row.apply(|x| *x += ln_weights[k]);
        }

        // Sample labels
        if hard_assignment {
            for (i, row) in ll.column_iter().enumerate() {
                self.labels[i] = row.argmax().0;
            }
        } else {
            col_normalize_log_weights(&mut ll);
            sample_weighted(&ll, &mut self.labels, rng);
        }
    }

    pub fn apply_sample_labels_aux(
        &mut self,
        params: &impl ThinParams,
        rng: &mut impl Rng,
    ) {
        // Split data points into contiguous blocks (indexes only for now)
        let (indices, offsets) = self.sorted_indices(params.n_clusters());

        // Sample pdf for each block and scatter them back in order to local.data
        let mut ll = DMatrix::zeros(2, self.n_points());
        for block_id in 0..offsets.len() - 1 {
            let indices = &indices[offsets[block_id]..offsets[block_id + 1]];
            let block = self.data.select_columns(indices);

            let (prim, aux) = (block_id / 2, block_id % 2);
            let probs = params.cluster_aux_dist(prim, aux).batchwise_ln_pdf(block).transpose();
            col_scatter(&mut ll.row_mut(aux), indices, &probs);
        }

        // Sample labels
        col_normalize_log_weights(&mut ll);
        sample_weighted(&ll, &mut self.labels_aux, rng);
    }

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

    fn collect_data_stats(&self) -> P::SuffStats {
        P::SuffStats::from_data(&self.data)
    }

    fn collect_cluster_stats(&self, n_clusters: usize) -> ThinStats<P> {
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

        ThinStats(stats)
    }

    fn apply_label_sampling<R: Rng + Clone + Send + Sync>(
        &mut self,
        params: &impl ThinParams,
        hard_assignment: bool,
        rng: &mut R,
    ) {
        self.apply_sample_labels_prim(params, hard_assignment, rng);
        self.apply_sample_labels_aux(params, rng);
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
        let mut removed = 0;
        for &k in cluster_ids {
            for l in self.labels.iter_mut() {
                if *l > k - removed {
                    *l -= 1;
                }
            }
            removed += 1;
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

pub fn sample_weighted<R: Rng>(weights: &DMatrix<f64>, labels: &mut RowDVector<usize>, rng: &mut R) {
    let labels = labels.as_mut_slice();
    for (i, col) in weights.column_iter().enumerate() {
        replacement_sampling_weighted(rng, col.into_iter().cloned(), &mut labels[i..=i]);
    }
}


#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector, RowDVector};
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use crate::clusters::SuperClusterStats;
    use crate::state::LocalWorker;
    use crate::stats::{FromData, NIW, NIWStats, SufficientStats};
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

    // #[test]
    // fn test_sorted_indices() {
    //     let data = DMatrix::from_fn(2, 120, |_, _| rng.gen_range(0.0..1.0));
    //     let labels = RowDVector::from_fn(120, |_, i| i / 30);
    //     let labels_aux = RowDVector::from_fn(120, |_, i| i / 15 % 2);
    //
    //     let local = super::LocalState::new(data.clone(), labels, labels_aux);
    //
    //     todo!()
    // }
    //
    // #[test]
    // fn test_sample_labels() {
    //     todo!()
    // }
    //
    // #[test]
    // fn test_sample_labels_aux() {
    //     todo!()
    // }
}