use std::iter::Sum;
use itertools::{Itertools, izip};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Range};
use std::vec::IntoIter;
use nalgebra::{Dim, DMatrix, DVector, Dynamic, Matrix, RowDVector, RowVector, Storage};
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use statrs::distribution::{Continuous};
use crate::clusters::SuperClusterStats;
use crate::global::GlobalState;
use crate::options::ModelOptions;
use crate::stats::{ContinuousBatchwise, FromData, GaussianPrior, SufficientStats};
use crate::utils::{col_normalize_log_weights, col_scatter, group_sort, replacement_sampling_weighted, row_normalize_log_weights};
use crate::utils::Iterutils;

#[derive(Debug, Clone)]
pub struct LocalStats<P: GaussianPrior>(pub Vec<SuperClusterStats<P>>);

impl<P: GaussianPrior> IntoIterator for LocalStats<P> {
    type Item = SuperClusterStats<P>;
    type IntoIter = IntoIter<SuperClusterStats<P>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, P: GaussianPrior> Add<&'a Self> for LocalStats<P> {
    type Output = Self;

    fn add(mut self, rhs: &'a Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<'a, P: GaussianPrior> AddAssign<&'a Self> for LocalStats<P> {
    fn add_assign(&mut self, rhs: &'a Self) {
        for (l, r) in self.0.iter_mut().zip(rhs.0.iter()) {
            *l += r;
        }
    }
}

impl<P: GaussianPrior> SufficientStats for LocalStats<P> {
    fn n_points(&self) -> usize {
        self.0.iter().map(|x| x.n_points()).sum()
    }
}

impl<P: GaussianPrior> Default for LocalStats<P> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<P: GaussianPrior> Sum for LocalStats<P> {
    fn sum<I: Iterator<Item=Self>>(mut iter: I) -> Self {
        let mut res = iter.next().unwrap_or_default();
        for x in iter {
            res += &x;
        }
        res
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalState<P: GaussianPrior> {
    pub data: DMatrix<f64>,
    pub labels: RowDVector<usize>,
    pub labels_aux: RowDVector<usize>,
    pub _phantoms: PhantomData<P>,
}

impl<P: GaussianPrior> LocalState<P> {
    pub fn new(
        data: DMatrix<f64>,
        labels: RowDVector<usize>,
        labels_aux: RowDVector<usize>,
    ) -> Self {
        Self { data, labels, labels_aux, _phantoms: PhantomData }
    }

    pub fn from_init<R: Rng>(
        data: DMatrix<f64>,
        n_clusters: usize,
        options: &ModelOptions<P>,
        rng: &mut R,
    ) -> Self {
        let n_points = data.ncols();
        let n_clusters = n_clusters + options.outlier.is_some() as usize;
        let labels = RowDVector::from_fn(n_points, |i, _| rng.gen_range(0..n_clusters));
        let labels_aux = RowDVector::from_fn(n_points, |i, _| rng.gen_range(0..2));
        Self::new(data, labels, labels_aux)
    }

    pub fn n_points(&self) -> usize {
        self.data.ncols()
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

    pub fn update_sample_labels<R: Rng>(
        global: &GlobalState<P>,
        local: &mut LocalState<P>,
        is_final: bool,
        rng: &mut R,
    ) {
        // Calculate log likelihood for each point
        let ln_weights = global.weights.iter().map(|w| w.ln()).collect::<Vec<_>>();
        let mut ll = DMatrix::zeros(global.clusters.len(), local.n_points());
        for (k, cluster) in global.clusters.iter().enumerate() {
            ll.row_mut(k).copy_from_slice(
                cluster.prim.dist.batchwise_ln_pdf(local.data.clone_owned()).as_slice()
            );
        }
        for (k, mut row) in ll.row_iter_mut().enumerate() {
            row.apply(|x| *x += ln_weights[k]);
        }

        // Sample labels
        if is_final {
            for (i, row) in ll.column_iter().enumerate() {
                local.labels[i] = row.argmax().0;
            }
        } else {
            col_normalize_log_weights(&mut ll);
            sample_weighted(&ll, &mut local.labels, rng);
        }
    }

    pub fn update_sample_labels_aux<R: Rng>(
        global: &GlobalState<P>,
        local: &mut LocalState<P>,
        rng: &mut R,
    ) {
        // Split data points into contiguous blocks (indexes only for now)
        let (indices, offsets) = local.sorted_indices(global.clusters.len());

        // Sample pdf for each block and scatter them back in order to local.data
        let mut ll = DMatrix::zeros(2, local.n_points());
        for block_id in 0..offsets.len() - 1 {
            let indices = &indices[offsets[block_id]..offsets[block_id + 1]];
            let block = local.data.select_columns(indices);

            let (prim, aux) = (block_id / 2, block_id % 2);
            let probs = global.clusters[prim].aux[aux].dist.batchwise_ln_pdf(block).transpose();
            col_scatter(
                &mut ll.row_mut(aux),
                indices,
                &probs,
            );
        }

        // Sample labels
        col_normalize_log_weights(&mut ll);
        sample_weighted(&ll, &mut local.labels_aux, rng);
    }

    pub fn collect_stats(
        local: &LocalState<P>,
        clusters: Range<usize>,
    ) -> LocalStats<P> {
        // Split data points into contiguous blocks (indexes only for now)
        let (indices, offsets) = local.sorted_indices(clusters.len());

        // Gather data from sorted indices
        let data = local.data.select_columns(&indices);

        let mut stats = vec![];
        for i in clusters {
            let i = i * 2;
            let prim = P::SuffStats::from_data(
                &data.columns_range(offsets[i]..offsets[i + 2])
            );
            let aux = [
                P::SuffStats::from_data(
                    &data.columns_range(offsets[i]..offsets[i + 1])
                ),
                P::SuffStats::from_data(
                    &data.columns_range(offsets[i + 1]..offsets[i + 2])
                ),
            ];
            stats.push(SuperClusterStats::new(prim, aux));
        }

        LocalStats(stats)
    }

    pub fn collect_data_stats(
        local: &LocalState<P>,
    ) -> P::SuffStats {
        P::SuffStats::from_data(&local.data)
    }

    pub fn update_reset_clusters<R: Rng>(
        local: &mut LocalState<P>,
        cluster_idx: &[usize],
        rng: &mut R,
    ) {
        for k in cluster_idx.iter().cloned() {
            for i in 0..local.n_points() {
                if local.labels[i] == k {
                    local.labels_aux[i] = rng.gen_range(0..2);
                }
            }
        }
    }

    pub fn update_remove_clusters(
        local: &mut LocalState<P>,
        cluster_idx: &[usize],
    ) {
        let mut removed = 0;
        for k in cluster_idx.iter().cloned() {
            for l in local.labels.iter_mut() {
                if *l > k - removed {
                    *l -= 1;
                }
            }
            removed += 1;
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
    use crate::stats::{FromData, NIW, NIWStats, SufficientStats};
    use crate::stats::tests::test_almost_mat;

    #[test]
    fn test_collect_stats() {
        let mut rng = StdRng::seed_from_u64(42);
        let data = DMatrix::from_fn(2, 120, |_, _| rng.gen_range(0.0..1.0));
        let labels = RowDVector::from_fn(120, |_, i| i / 30);
        let labels_aux = RowDVector::from_fn(120, |_, i| i / 15 % 2);

        let local = super::LocalState::new(data.clone(), labels, labels_aux);
        let stats = super::LocalState::<NIW>::collect_stats(&local, 0..4);
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