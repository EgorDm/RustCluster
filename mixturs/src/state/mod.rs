mod global;
mod local;
mod local_sharded;

pub use global::GlobalState;
pub use local::{LocalState};
pub use local_sharded::ShardedState;

use rand::Rng;
use crate::params::clusters::SuperClusterStats;
use crate::params::options::ModelOptions;
use crate::params::thin::ThinParams;
use crate::stats::NormalConjugatePrior;

pub trait GlobalWorker<P: NormalConjugatePrior> {
    /// Returns the number of clusters in the global state.
    fn n_clusters(&self) -> usize;

    /// Returns the number of points the current stats are collected over
    fn n_points(&self) -> usize;

    /// Updates the global state with the given cluster stats
    ///
    /// # Arguments
    ///
    /// * `stats`: The cluster stats to update the global state with
    fn update_clusters_post(&mut self, stats: Vec<SuperClusterStats<P>>);

    /// Updates the global state by sampling new clusters from cluster parameters
    ///
    /// # Arguments
    ///
    /// * `options`: The model options
    /// * `rng`: The random number generator
    fn update_sample_clusters<R: Rng>(&mut self, options: &ModelOptions<P>, rng: &mut R);

    /// Collects the indices of clusters that are bad and should be removed
    /// A cluster is bad if all of its points are assigned to the same auxiliary cluster
    fn collect_bad_clusters(&mut self) -> Vec<usize>;

    /// Collects the indices of clusters that should be removed
    /// A cluster should be removed if it has no points
    fn collect_remove_clusters(&mut self, options: &ModelOptions<P>) -> Vec<usize>;

    /// Checks if any clusters should be split, performs the split and returns the indices of the split clusters
    ///
    /// # Arguments
    ///
    /// * `options`: The model options
    /// * `rng`: The random number generator
    ///
    /// # Returns
    ///
    /// A vector of tuples where first element is index of the split cluster and the second element is the index of the new cluster
    fn check_and_split<R: Rng>(&mut self, options: &ModelOptions<P>, rng: &mut R) -> Vec<(usize, usize)>;

    /// Checks if any clusters should be merged, performs the merge and returns the indices of the merged clusters
    ///
    /// # Arguments
    ///
    /// * `options`: The model options
    /// * `rng`: The random number generator
    ///
    /// # Returns
    ///
    /// A vector of tuples where first element is index of the merged cluster and the second element is the index of
    /// the cluster that was merged into the first cluster
    fn check_and_merge<R: Rng>(&mut self, options: &ModelOptions<P>, rng: &mut R) -> Vec<(usize, usize)>;
}

/// Local worker keeps track of data and is responsible to calculate the sufficient statistics over these points
/// given cluster parameters
pub trait LocalWorker<P: NormalConjugatePrior> {
    /// Initializes the local state by randomly assigning points to clusters
    ///
    /// # Arguments
    ///
    /// * `n_clusters`: The number of clusters to initialize the local state with
    /// * `rng`: The random number generator
    ///
    fn init<R: Rng + Clone + Send + Sync>(
        &mut self,
        n_clusters: usize,
        rng: &mut R,
    );

    /// Returns the number of points in the local state
    fn n_points(&self) -> usize;

    /// Collects the sufficient statistics over all of the data points
    fn collect_data_stats(&self) -> P::SuffStats;

    /// Collects the sufficient statistics over all of the data points for each cluster
    ///
    /// # Arguments
    ///
    /// * `n_clusters`: The number of clusters present in the model
    fn collect_cluster_stats(&self, n_clusters: usize) -> Vec<SuperClusterStats<P>>;

    /// Assigns points to clusters based on the cluster parameters and sampling strategy
    ///
    /// # Arguments
    ///
    /// * `params`: The cluster parameters
    /// * `hard_assignment`: Whether to perform hard assignment or soft assignment (i.e. sampling strategy)
    /// * `rng`: The random number generator
    fn apply_label_sampling<R: Rng + Clone + Send + Sync>(
        &mut self,
        params: &impl ThinParams,
        hard_assignment: bool,
        rng: &mut R,
    );

    /// Resets the auxiliary cluster assignments of the given clusters
    fn apply_cluster_reset<R: Rng + Clone + Send + Sync>(
        &mut self,
        cluster_ids: &[usize],
        rng: &mut R,
    );

    /// Removes the given clusters and updates label assignments accordingly
    ///
    /// # Arguments
    ///
    /// * `cluster_ids`: The indices of the clusters to remove
    fn apply_cluster_remove(
        &mut self,
        cluster_ids: &[usize],
    );

    /// Splits the given clusters and updates label assignments accordingly
    ///
    /// # Arguments
    ///
    /// * `split_decisions`: A vector of tuples where first element is index of the split cluster and the second element
    /// is the index of the new cluster
    /// * `rng`: The random number generator
    fn apply_split<R: Rng + Clone + Send + Sync>(
        &mut self,
        split_decisions: &[(usize, usize)],
        rng: &mut R,
    );

    /// Merges the given clusters and updates label assignments accordingly
    ///
    /// # Arguments
    ///
    /// * `merge_decisions`: A vector of tuples where first element is index of the merged cluster and the second element
    /// is the index of the cluster that was merged into the first cluster
    fn apply_merge(
        &mut self,
        merge_decisions: &[(usize, usize)],
    );
}