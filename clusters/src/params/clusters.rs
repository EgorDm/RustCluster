use std::collections::VecDeque;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign};
use std::vec::IntoIter;
use itertools::repeat_n;
use nalgebra::{DMatrix, DVector, RowDVector};
use rand::distributions::Distribution;
use rand::Rng;
use statrs::distribution::{Dirichlet, MultivariateNormal};
use crate::stats::{ContinuousBatchwise, NormalConjugatePrior, SufficientStats};
use serde::{Serialize, Deserialize};
use serde::de::DeserializeOwned;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SuperClusterParams<P: NormalConjugatePrior> {
    pub prim: ClusterParams<P>,
    pub aux: [ClusterParams<P>; 2],
    pub weights: [f64; 2],
    pub splittable: bool,
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

        let dir = Dirichlet::new(DVector::from_vec(vec![alpha / 2.0, alpha / 2.0])).unwrap();
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

        let dir = Dirichlet::new(DVector::from_vec(vec![
            prim_l.stats.n_points() as f64 + alpha / 2.0, prim_r.stats.n_points() as f64 + alpha / 2.0,
        ])).unwrap();
        let weights = dir.sample(rng).as_slice().try_into().unwrap();

        SuperClusterParams {
            prim,
            aux: [prim_l, prim_r],
            weights,
            splittable: false,
            ll_history: LLHistory::new(burnout_period),
        }
    }

    pub fn n_points(&self) -> usize {
        self.prim.n_points()
    }

    pub fn sample<R: Rng + ?Sized>(&self, alpha: f64, rng: &mut R) -> (MultivariateNormal, [MultivariateNormal; 2], [f64; 2]) {
        let prim = self.prim.sample(rng);
        let aux = [self.aux[0].sample(rng), self.aux[1].sample(rng)];

        let dir = Dirichlet::new(DVector::from_vec(vec![
            self.aux[0].stats.n_points() as f64 + alpha / 2.0, self.aux[1].stats.n_points() as f64 + alpha / 2.0,
        ])).unwrap();
        let weights = dir.sample(rng).as_slice().try_into().unwrap();

        (prim, aux, weights)
    }

    pub fn update_post(&mut self, stats: SuperClusterStats<P>) {
        self.prim.update_post(stats.prim);
        for (i, stats_aux) in stats.aux.into_iter().enumerate() {
            self.aux[i].update_post(stats_aux);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperClusterStats<P: NormalConjugatePrior> {
    pub prim: P::SuffStats,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterParams<P: NormalConjugatePrior> {
    pub prior: P::HyperParams,
    pub post: P::HyperParams,
    pub stats: P::SuffStats,
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


#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
            return f64::INFINITY;
        } else {
            self.ll_history.iter().take(burnout_period).sum::<f64>() / (burnout_period as f64 - 0.1)
        }
    }

    pub fn converged(&self, burnout_period: usize) -> bool {
        let ll_weighted = self.ll_weighted(burnout_period);
        ll_weighted.is_finite() && ll_weighted - self.ll_history[burnout_period] < 1e-2
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinStats<P: NormalConjugatePrior>(pub Vec<SuperClusterStats<P>>);

impl<P: NormalConjugatePrior> IntoIterator for ThinStats<P> {
    type Item = SuperClusterStats<P>;
    type IntoIter = IntoIter<SuperClusterStats<P>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, P: NormalConjugatePrior> Add<&'a Self> for ThinStats<P> {
    type Output = Self;

    fn add(mut self, rhs: &'a Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<'a, P: NormalConjugatePrior> AddAssign<&'a Self> for ThinStats<P> {
    fn add_assign(&mut self, rhs: &'a Self) {
        for (l, r) in self.0.iter_mut().zip(rhs.0.iter()) {
            *l += r;
        }
    }
}

impl<P: NormalConjugatePrior> SufficientStats for ThinStats<P> {
    fn n_points(&self) -> usize {
        self.0.iter().map(|x| x.n_points()).sum()
    }
}

impl<P: NormalConjugatePrior> Default for ThinStats<P> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<P: NormalConjugatePrior> Sum for ThinStats<P> {
    fn sum<I: Iterator<Item=Self>>(mut iter: I) -> Self {
        let mut res = iter.next().unwrap_or_default();
        for x in iter {
            res += &x;
        }
        res
    }
}