use crate::stats::{NormalConjugatePrior, PriorHyperParams};

/// Outlier removal options
#[derive(Debug, Clone, PartialEq)]
pub struct OutlierRemoval<P: NormalConjugatePrior> {
    /// Weight of the outlier prior
    pub weight: f64,
    /// Outlier prior
    pub dist: P::HyperParams,
}

/// Options for the DPMMSC model
#[derive(Debug, Clone, PartialEq)]
pub struct ModelOptions<P: NormalConjugatePrior> {
    /// Prior for the complete data distribution
    pub data_dist: P::HyperParams,
    /// Concentration parameter for the Dirichlet process
    pub alpha: f64,
    /// Dimensionality of the data
    pub dim: usize,
    /// Burnout period for the supercluster log likelihood history (in iterations)
    pub burnout_period: usize,
    /// Outlier removal options
    pub outlier: Option<OutlierRemoval<P>>,
    /// Whether to use hard assignment during expectation phase
    pub hard_assignment: bool,
}

impl<P: NormalConjugatePrior> ModelOptions<P> {
    pub fn default(dim: usize) -> Self {
        Self {
            data_dist: P::HyperParams::default(dim),
            alpha: 10.0,
            dim,
            burnout_period: 20,
            outlier: Some(OutlierRemoval {
                weight: 0.05,
                dist: P::HyperParams::default(dim),
            }),
            hard_assignment: false,
        }
    }
}

/// Options for the DPMMSC model fit method
#[derive(Debug, Clone)]
pub struct FitOptions {
    /// Seed for the random number generator
    pub seed: u64,
    /// Whether to reuse the previous model parameters
    pub reuse: bool,
    /// Number of initial clusters
    pub init_clusters: usize,
    /// Maximum number of clusters
    pub max_clusters: usize,
    /// Maximum number of iterations
    pub iters: usize,
    /// Number of iterations before max iteration to start using argmax label sampling strategy
    pub argmax_sample_stop: usize,
    /// Number of iterations before max iteration to stop split/merge proposals
    pub iter_split_stop: usize,
    /// Number of workers (threads) for parallelization (-1 = number of CPUs)
    pub workers: i32,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            seed: 42,
            reuse: false,
            init_clusters: 1,
            max_clusters: usize::MAX,
            iters: 100,
            argmax_sample_stop: 5,
            iter_split_stop: 5,
            workers: 1,
        }
    }
}
