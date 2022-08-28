use std::fmt::Debug;
use std::ops::Add;
use nalgebra::{Dynamic, Matrix, Storage};
use rand::Rng;
use statrs::distribution::MultivariateNormal;

pub use niw::*;

mod niw;

pub trait ConjugatePrior: Clone {
    type HyperParams: PriorHyperParams + Debug + Clone + PartialEq;
    type SuffStats: SufficientStats + Debug + Clone + PartialEq;

    fn posterior(
        prior: &Self::HyperParams,
        stats: &Self::SuffStats,
    ) -> Self::HyperParams;

    fn marginal_log_likelihood(
        prior: &Self::HyperParams,
        post: &Self::HyperParams,
        stats: &Self::SuffStats,
    ) -> f64;

    fn posterior_predictive<S: Storage<f64, Dynamic, Dynamic>>(
        post: &Self::HyperParams,
        data: &Matrix<f64, Dynamic, Dynamic, S>,
    ) -> f64;
}

pub trait PriorHyperParams {
    fn default(dim: usize) -> Self;
}

pub trait SufficientStats: Sized + Default {
    fn from_data<S: Storage<f64, Dynamic, Dynamic>>(
        data: &Matrix<f64, Dynamic, Dynamic, S>,
    ) -> Self;

    fn n_points(&self) -> usize;

    fn add(&self, rhs: &Self) -> Self;
}

pub trait GaussianPrior: ConjugatePrior {
    fn sample<R: Rng + ?Sized>(prior: &Self::HyperParams, rng: &mut R) -> MultivariateNormal;
}