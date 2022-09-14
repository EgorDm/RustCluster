use nalgebra::DMatrix;
use rand::Rng;
use rayon::prelude::*;
use crate::params::clusters::SuperClusterStats;
use crate::params::thin::ThinParams;
use crate::state::{LocalState, LocalWorker};
use crate::stats::NormalConjugatePrior;

pub struct ShardedState<P: NormalConjugatePrior> {
    pub shards: Vec<LocalState<P>>,
}

impl<P: NormalConjugatePrior> ShardedState<P> {
    pub fn new(shards: Vec<LocalState<P>>) -> Self {
        Self { shards }
    }

    pub fn from_data(data: DMatrix<f64>, n_shards: usize) -> Self {
        let shard_size = (data.ncols() as f64 / n_shards as f64).ceil() as usize;

        let mut shards = vec![];
        for i in (0..data.ncols()).step_by(shard_size) {
            shards.push(LocalState::from_data(
                data.columns_range(i..std::cmp::min(i + shard_size, data.ncols())).clone_owned()
            ));
        }

        ShardedState::new(shards)
    }

    pub fn n_shards(&self) -> usize {
        self.shards.len()
    }
}

impl<P: NormalConjugatePrior> LocalWorker<P> for ShardedState<P> {
    fn init<R: Rng + Clone + Send + Sync>(&mut self, n_clusters: usize, rng: &mut R) {
        self.shards.iter_mut().for_each(|shard| {
            shard.init(n_clusters, rng);
        });
    }

    fn n_points(&self) -> usize {
        self.shards.iter().map(|shard| shard.n_points()).sum()
    }

    fn collect_data_stats(&self) -> P::SuffStats {
        self.shards.par_iter().map(LocalWorker::<P>::collect_data_stats).sum()
    }

    fn collect_cluster_stats(&self, n_clusters: usize) -> Vec<SuperClusterStats<P>> {
        let full: Vec<_> = self.shards.par_iter()
            .map(|shard| shard.collect_cluster_stats(n_clusters))
            .collect();
        let mut iter = full.into_iter();

        let first = iter.next().unwrap();
        iter.fold(first, |mut stats, next_stats| {
            for (i, stat) in stats.iter_mut().enumerate() {
                *stat += &next_stats[i];
            }
            stats
        })
    }

    fn apply_label_sampling<R: Rng + Clone + Send + Sync>(
        &mut self,
        params: &impl ThinParams,
        hard_assignment: bool,
        rng: &mut R,
    ) {
        self.shards.par_iter_mut().for_each_with(rng.clone(), |rng, shard| {
            shard.apply_label_sampling(params, hard_assignment, rng);
        });
    }

    fn apply_cluster_reset<R: Rng + Clone + Send + Sync>(
        &mut self,
        cluster_ids: &[usize],
        rng: &mut R,
    ) {
        self.shards.par_iter_mut().for_each_with(rng.clone(), |rng, shard| {
            shard.apply_cluster_reset(cluster_ids, rng);
        });
    }

    fn apply_cluster_remove(
        &mut self,
        cluster_ids: &[usize],
    ) {
        self.shards.par_iter_mut().for_each(|shard| {
            shard.apply_cluster_remove(cluster_ids);
        });
    }

    fn apply_split<R: Rng + Clone + Send + Sync>(
        &mut self,
        split_decisions: &[(usize, usize)],
        rng: &mut R,
    ) {
        self.shards.par_iter_mut().for_each_with(rng.clone(), |rng, shard| {
            shard.apply_split(split_decisions, rng);
        });
    }

    fn apply_merge(
        &mut self,
        merge_decisions: &[(usize, usize)],
    ) {
        self.shards.par_iter_mut().for_each(|shard| {
            shard.apply_merge(merge_decisions);
        });
    }
}