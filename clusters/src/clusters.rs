use nalgebra::DVector;
use rand::distributions::Distribution;
use rand::Rng;
use statrs::distribution::{Dirichlet, MultivariateNormal};
use crate::stats::priors::{GaussianPrior, SufficientStats};

#[derive(Debug, Clone, PartialEq)]
pub struct SuperClusterParams<P: GaussianPrior> {
    pub prim: ClusterParams<P>,
    pub aux: [ClusterParams<P>; 2],
    pub weights: [f64; 2],
    pub splittable: bool,
    pub ll_history: Vec<f64>,
}

impl<P: GaussianPrior> SuperClusterParams<P> {
    pub fn from_split_params<R: Rng>(
        prim: ClusterParams<P>, alpha: f64, burnout_period: usize, rng: &mut R,
    ) -> Self {
        let mut aux = [prim.clone(), prim.clone()];
        for aux_k in &mut aux {
            aux_k.dist = P::sample(&prim.post, rng);
        }

        let dir = Dirichlet::new(DVector::from_row_slice(&[alpha / 2.0, alpha / 2.0])).unwrap();
        let weights = dir.sample(rng).as_slice().try_into().unwrap();

        SuperClusterParams {
            prim,
            aux,
            weights,
            splittable: false,
            ll_history: vec![f64::NEG_INFINITY; burnout_period + 5],
        }
    }

    pub fn from_merge_params<R: Rng>(
        prim_l: ClusterParams<P>,
        prim_r: ClusterParams<P>,
        alpha: f64, burnout_period: usize, rng: &mut R,
    ) -> Self {
        let stats = prim_l.stats.add(&prim_r.stats);
        let post = P::posterior(&prim_l.prior, &stats);
        let prim = ClusterParams::new(prim_l.prior.clone(), post, stats, prim_r.dist.clone());

        let dir = Dirichlet::new(DVector::from_row_slice(&[
            prim_l.stats.n_points() as f64 + alpha / 2.0, prim_r.stats.n_points() as f64 + alpha / 2.0
        ])).unwrap();
        let weights = dir.sample(rng).as_slice().try_into().unwrap();

        SuperClusterParams {
            prim,
            aux: [prim_l, prim_r],
            weights,
            splittable: false,
            ll_history: vec![f64::NEG_INFINITY; burnout_period + 5],
        }
    }

    pub fn n_points(&self) -> usize {
        self.prim.n_points()
    }

    pub fn sample<R: Rng + ?Sized>(&self, alpha: f64, rng: &mut R) -> (MultivariateNormal, [MultivariateNormal; 2], [f64; 2]) {
        let prim = self.prim.sample(rng);
        let aux = [self.aux[0].sample(rng), self.aux[1].sample(rng)];

        let dir = Dirichlet::new(DVector::from_row_slice(&[
            self.aux[0].stats.n_points() as f64 + alpha / 2.0, self.aux[1].stats.n_points() as f64 + alpha / 2.0
        ])).unwrap();
        let weights = dir.sample(rng).as_slice().try_into().unwrap();

        (prim, aux, weights)
    }

    pub fn update_post(
        &mut self,
        stats: P::SuffStats,
        stats_aux: [P::SuffStats; 2],
    ) {
        self.prim.update_post(stats);
        for (i, stats_aux) in stats_aux.into_iter().enumerate() {
            self.aux[i].update_post(stats_aux);
        }
    }

    pub fn update_history(&mut self, ll: f64) {
        self.ll_history[..].rotate_right(1);
        self.ll_history[0] = ll;
    }

    pub fn ll_weighted(&self, burnout_period: usize) -> f64 {
        assert!(self.ll_history.len() > burnout_period);
        self.ll_history.iter().take(burnout_period).sum::<f64>() / (burnout_period as f64 - 0.1)
    }

    pub fn converged(&self, burnout_period: usize) -> bool {
        let ll_weighted = self.ll_weighted(burnout_period);
        ll_weighted.is_finite() && ll_weighted - self.ll_history[burnout_period] < 1e-2
    }
}


#[derive(Debug, Clone, PartialEq)]
pub struct ClusterParams<P: GaussianPrior> {
    pub prior: P::HyperParams,
    pub post: P::HyperParams,
    pub stats: P::SuffStats,
    pub dist: MultivariateNormal,
}

impl<P: GaussianPrior> ClusterParams<P> {
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

