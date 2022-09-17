use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign};
use nalgebra::{Dynamic, Matrix, Storage};
use rand::Rng;
use statrs::distribution::MultivariateNormal;

pub use niw::*;

mod niw;

pub trait ConjugatePrior: Clone {
    type HyperParams: PriorHyperParams + Debug + Clone + PartialEq + Send + Sync + 'static;
    type SuffStats: SufficientStats + FromData + Default + Debug + PartialEq + Send + Sync + 'static;

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

pub trait FromData {
    fn from_data<S: Storage<f64, Dynamic, Dynamic>>(
        data: &Matrix<f64, Dynamic, Dynamic, S>,
    ) -> Self;
}

pub trait SufficientStats: Sized + Clone
+ for<'a> Add<&'a Self, Output=Self>
+ for<'a> AddAssign<&'a Self>
+ Sum
{
    fn n_points(&self) -> usize;
}

pub trait NormalConjugatePrior: ConjugatePrior {
    fn sample<R: Rng + ?Sized>(prior: &Self::HyperParams, rng: &mut R) -> MultivariateNormal;
}