use std::marker::PhantomData;
use nalgebra::{Dim, DMatrix, DVector, Dynamic, Matrix, RowDVector, RowVector, Storage};
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use statrs::distribution::{Continuous};
use crate::global::GlobalState;
use crate::options::ModelOptions;
use crate::stats::{GaussianPrior, SufficientStats};
use crate::utils::{col_normalize_log_weights, row_normalize_log_weights};

pub type LocalStats<P: GaussianPrior> = Vec<(P::SuffStats, [P::SuffStats; 2])>;

#[derive(Debug, Clone, PartialEq)]
pub struct LocalState<P: GaussianPrior> {
    pub data: DMatrix<f64>,
    pub labels: RowDVector<usize>,
    pub labels_aux: RowDVector<usize>,
    pub _phantoms: PhantomData<P>,
}

impl<P: GaussianPrior> LocalState<P> {
    pub fn from_init<R: Rng>(
        data: DMatrix<f64>,
        n_clusters: usize,
        options: &ModelOptions<P>,
        rng: &mut R
    ) -> Self {
        let n_points = data.ncols();
        let n_clusters = n_clusters + options.outlier.is_some() as usize;
        let labels = RowDVector::from_fn(n_points, |i, _| rng.gen_range(0..n_clusters));
        let labels_aux = RowDVector::from_fn(n_points, |i, _| rng.gen_range(0..2));
        Self {
            data,
            labels,
            labels_aux,
            _phantoms: PhantomData,
        }
    }

    pub fn n_points(&self) -> usize {
        self.data.ncols()
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
            for (i, point) in local.data.column_iter().enumerate() {
                ll[(k, i)] = cluster.prim.dist.ln_pdf(&point.clone_owned()) + ln_weights[k];
            }
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
        let mut ll = DMatrix::zeros(2, local.n_points());
        for (k, cluster) in global.clusters.iter().enumerate() {
            let ln_weights = cluster.weights.iter().map(|w| w.ln()).collect::<Vec<_>>();
            for (i, _) in local.labels.iter().enumerate().filter(|(_, &label)| label == k) {
                let point = local.data.column(i).clone_owned();
                for a in 0..2 {
                    ll[(a, i)] = cluster.aux[a].dist.ln_pdf(&point) + ln_weights[a];
                }
            }
        }

        // Sample labels
        col_normalize_log_weights(&mut ll);
        sample_weighted(&ll, &mut local.labels_aux, rng);
    }

    pub fn collect_stats(
        local: &LocalState<P>,
        n_clusters: usize,
    ) -> LocalStats<P> {
        (0..n_clusters)
            .map(|k| Self::collect_stats_cluster(local, k))
            .collect()
    }

    pub fn collect_stats_cluster(
        local: &LocalState<P>,
        cluster_id: usize,
    ) -> (P::SuffStats, [P::SuffStats; 2]) {
        let idx_l: Vec<_> = local.labels.iter().cloned()
            .zip(local.labels_aux.iter().cloned())
            .enumerate()
            .filter(|(_, (x, y))| *x == cluster_id && *y == 0)
            .map(|(i, _)| i)
            .collect();
        let idx_r = local.labels.iter().cloned()
            .zip(local.labels_aux.iter().cloned())
            .enumerate()
            .filter(|(_, (x, y))| *x == cluster_id && *y == 1)
            .map(|(i, _)| i);

        let idx: Vec<_> = idx_l.iter().cloned().chain(idx_r).collect();
        let points = local.data.select_columns(&idx);

        let prim = P::SuffStats::from_data(&points);
        let aux = [
            P::SuffStats::from_data(&points.columns_range(0..idx_l.len()).into_owned()),
            P::SuffStats::from_data(&points.columns_range(idx_l.len()..).into_owned()),
        ];

        (prim, aux)
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
    for (i, col) in weights.column_iter().enumerate() {
        // TODO: take the weighted reservoir sampler from tch-geometric
        let dist = WeightedIndex::new(&col).unwrap();
        labels[i] = dist.sample(rng);
    }
}