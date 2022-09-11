mod global;
mod local;
mod local_sharded;

pub use global::GlobalState;
pub use local::LocalState;
pub use local_sharded::ShardedState;

use rand::Rng;
use crate::clusters::{ThinParams, ThinStats};
use crate::options::ModelOptions;
use crate::stats::NormalConjugatePrior;

pub trait GlobalWorker<P: NormalConjugatePrior> {
    fn n_clusters(&self) -> usize;

    fn n_points(&self) -> usize;

    fn update_clusters_post(&mut self, stats: ThinStats<P>);

    fn update_sample_clusters<R: Rng>(&mut self, options: &ModelOptions<P>, rng: &mut R);

    fn collect_bad_clusters(&mut self) -> Vec<usize>;

    fn collect_remove_clusters(&mut self, options: &ModelOptions<P>) -> Vec<usize>;

    fn check_and_split<R: Rng>(&mut self, options: &ModelOptions<P>, rng: &mut R) -> Vec<(usize, usize)>;

    fn check_and_merge<R: Rng>(&mut self, options: &ModelOptions<P>, rng: &mut R) -> Vec<(usize, usize)>;
}

pub trait LocalWorker<P: NormalConjugatePrior> {
    fn init<R: Rng + Clone + Send + Sync>(
        &mut self,
        n_clusters: usize,
        rng: &mut R,
    );

    fn collect_data_stats(&self) -> P::SuffStats;

    fn collect_cluster_stats(&self, n_clusters: usize) -> ThinStats<P>;

    fn apply_label_sampling<R: Rng + Clone + Send + Sync>(
        &mut self,
        params: &impl ThinParams,
        hard_assignment: bool,
        rng: &mut R,
    );

    fn apply_cluster_reset<R: Rng + Clone + Send + Sync>(
        &mut self,
        cluster_ids: &[usize],
        rng: &mut R,
    );

    fn apply_cluster_remove(
        &mut self,
        cluster_ids: &[usize],
    );

    fn apply_split<R: Rng + Clone + Send + Sync>(
        &mut self,
        split_decisions: &[(usize, usize)],
        rng: &mut R,
    );

    fn apply_merge(
        &mut self,
        merge_decisions: &[(usize, usize)],
    );
}