use std::collections::VecDeque;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign};
use rand::{Rng, distributions::Distribution};
use statrs::distribution::{Dirichlet, MultivariateNormal};
use crate::stats::{NormalConjugatePrior, SufficientStats};

/// Parameters for a supercluster.
#[derive(Debug, Clone, PartialEq)]
pub struct SuperClusterParams<P: NormalConjugatePrior> {
    /// Parameters for the primary cluster.
    pub prim: ClusterParams<P>,
    /// Parameters for the two auxiliary clusters.
    pub aux: [ClusterParams<P>; 2],
    /// Weights of the two auxiliary clusters.
    pub weights: [f64; 2],
    /// Whether the supercluster is splittable.
    pub splittable: bool,
    /// History of the log likelihood of the supercluster to detect convergence.
    pub ll_history: LLHistory,
}

impl<P: NormalConjugatePrior> SuperClusterParams<P> {
    pub fn from_split_params<R: Rng>(
        prim: ClusterParams<P>,
        alpha: f64,
        burnout_period: usize,
        rng: &mut R,
    ) -> Self {
        let mut aux = [prim.clone(), prim.clone()];
        for aux_k in &mut aux {
            aux_k.dist = P::sample(&prim.post, rng);
        }

        let dir = Dirichlet::new(vec![alpha / 2.0, alpha / 2.0]).unwrap();
        let weights = dir.sample(rng).as_slice().try_into().unwrap();

        SuperClusterParams {
            prim,
            aux,
            weights,
            splittable: false,
            ll_history: LLHistory::new(burnout_period),
        }
    }

    pub fn from_merge_params<R: Rng>(
        prim_l: ClusterParams<P>,
        prim_r: ClusterParams<P>,
        alpha: f64,
        burnout_period: usize,
        rng: &mut R,
    ) -> Self {
        let stats = prim_l.stats.clone() + &prim_r.stats;
        let post = P::posterior(&prim_l.prior, &stats);
        let prim = ClusterParams::new(prim_l.prior.clone(), post, stats, prim_r.dist.clone());

        let dir = Dirichlet::new(vec![
            prim_l.stats.n_points() as f64 + alpha / 2.0, prim_r.stats.n_points() as f64 + alpha / 2.0,
        ]).unwrap();
        let weights = dir.sample(rng).as_slice().try_into().unwrap();

        SuperClusterParams {
            prim,
            aux: [prim_l, prim_r],
            weights,
            splittable: false,
            ll_history: LLHistory::new(burnout_period),
        }
    }

    /// Number of points in the supercluster.
    pub fn n_points(&self) -> usize {
        self.prim.n_points()
    }

    /// Sample a new supercluster distributions given current supercluster params.
    pub fn sample<R: Rng + ?Sized>(&self, alpha: f64, rng: &mut R) -> (MultivariateNormal, [MultivariateNormal; 2], [f64; 2]) {
        let prim = self.prim.sample(rng);
        let aux = [self.aux[0].sample(rng), self.aux[1].sample(rng)];

        let dir = Dirichlet::new(vec![
            self.aux[0].stats.n_points() as f64 + alpha / 2.0, self.aux[1].stats.n_points() as f64 + alpha / 2.0,
        ]).unwrap();
        let weights = dir.sample(rng).as_slice().try_into().unwrap();

        (prim, aux, weights)
    }

    /// Update the supercluster parameters given sufficient statistics gathered from data.
    pub fn update_post(&mut self, stats: SuperClusterStats<P>) {
        self.prim.update_post(stats.prim);
        for (i, stats_aux) in stats.aux.into_iter().enumerate() {
            self.aux[i].update_post(stats_aux);
        }
    }
}

/// Sufficient statistics for a supercluster.
#[derive(Debug, Clone)]
pub struct SuperClusterStats<P: NormalConjugatePrior> {
    /// Sufficient statistics for the primary cluster.
    pub prim: P::SuffStats,
    /// Sufficient statistics for the two auxiliary clusters.
    pub aux: [P::SuffStats; 2],
}

impl<P: NormalConjugatePrior> SuperClusterStats<P> {
    pub fn new(prim: P::SuffStats, aux: [P::SuffStats; 2]) -> Self {
        SuperClusterStats { prim, aux }
    }
}

impl<'a, P: NormalConjugatePrior> Add<&'a Self> for SuperClusterStats<P> {
    type Output = Self;

    fn add(mut self, rhs: &'a Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<'a, P: NormalConjugatePrior> AddAssign<&'a Self> for SuperClusterStats<P> {
    fn add_assign(&mut self, rhs: &'a Self) {
        self.prim += &rhs.prim;
        for (i, rhs_aux) in rhs.aux.iter().enumerate() {
            self.aux[i] += rhs_aux;
        }
    }
}

impl<P: NormalConjugatePrior> Sum for SuperClusterStats<P> {
    fn sum<I: Iterator<Item=Self>>(mut iter: I) -> Self {
        let res = iter.next().expect("Cannot sum over empty iterator");
        iter.fold(res, |mut acc, x| {
            acc += &x;
            acc
        })
    }
}

impl<P: NormalConjugatePrior> SufficientStats for SuperClusterStats<P> {
    fn n_points(&self) -> usize {
        self.prim.n_points()
    }
}

/// Parameters for a cluster.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterParams<P: NormalConjugatePrior> {
    /// Prior distribution params for the cluster.
    pub prior: P::HyperParams,
    /// Posterior distribution params for the cluster.
    pub post: P::HyperParams,
    /// Sufficient statistics for the cluster.
    pub stats: P::SuffStats,
    /// Normal Distribution for the cluster.
    pub dist: MultivariateNormal,
}

impl<P: NormalConjugatePrior> ClusterParams<P> {
    pub fn new(prior: P::HyperParams, post: P::HyperParams, stats: P::SuffStats, dist: MultivariateNormal) -> Self {
        Self { prior, post, stats, dist }
    }

    pub fn n_points(&self) -> usize {
        self.stats.n_points()
    }

    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> MultivariateNormal {
        P::sample(&self.post, rng)
    }

    pub fn update_post(&mut self, stats: P::SuffStats) {
        self.post = P::posterior(&self.prior, &stats);
        self.stats = stats;
    }

    pub fn marginal_log_likelihood(&self) -> f64 {
        P::marginal_log_likelihood(&self.prior, &self.post, &self.stats)
    }
}


/// Log likelihood history to track cluster convergence.
#[derive(Debug, Clone, PartialEq)]
pub struct LLHistory {
    pub ll_history: VecDeque<f64>,
    pub capacity: usize,
}

impl LLHistory {
    pub fn new(burnout_period: usize) -> Self {
        LLHistory {
            ll_history: VecDeque::with_capacity(burnout_period + 5),
            capacity: burnout_period + 5,
        }
    }

    pub fn clear(&mut self) {
        self.ll_history.clear();
    }

    pub fn push(&mut self, ll: f64) {
        if self.ll_history.len() == self.capacity {
            self.ll_history.pop_back();
        }
        self.ll_history.push_front(ll);
    }

    pub fn ll_weighted(&self, burnout_period: usize) -> f64 {
        if self.ll_history.len() <= burnout_period {
            f64::INFINITY
        } else {
            self.ll_history.iter().take(burnout_period).sum::<f64>() / (burnout_period as f64 - 0.1)
        }
    }

    pub fn converged(&self, burnout_period: usize) -> bool {
        let ll_weighted = self.ll_weighted(burnout_period);
        ll_weighted.is_finite() && ll_weighted - self.ll_history[burnout_period] < 1e-2
    }
}