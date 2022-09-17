use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign};
use nalgebra::{Dynamic, Matrix, Storage};
use rand::Rng;
use statrs::distribution::MultivariateNormal;

pub use niw::*;

mod niw;

pub trait ConjugatePrior: Clone {
    /// The hyperparameters of the prior distribution.
    type HyperParams: PriorHyperParams + Debug + Clone + PartialEq + Send + Sync + 'static;
    /// The sufficient statistics of the data needed to compute the posterior.
    type SuffStats: SufficientStats + FromData + Default + Debug + PartialEq + Send + Sync + 'static;


    /// Compute the posterior hyperparameters given the prior hyperparameters and the sufficient statistics.
    ///
    /// # Arguments
    ///
    /// * `prior`: the hyperparameters of the prior distribution
    /// * `stats`: the sufficient statistics of the data
    ///
    /// # Returns
    /// The hyperparameters for the posterior distribution
    fn posterior(
        prior: &Self::HyperParams,
        stats: &Self::SuffStats,
    ) -> Self::HyperParams;

    /// Compute the marginal log likelihood of the data given the prior and posterior hyperparameters.
    /// This is the log likelihood of the data given the prior hyperparameters, plus the log likelihood
    /// of the hyperparameters given the prior hyperparameters.
    ///
    /// # Arguments
    ///
    /// * `prior`: the hyperparameters of the prior distribution
    /// * `post`: the hyperparameters of the posterior distribution
    /// * `stats`: the sufficient statistics of the data
    ///
    /// # Returns
    /// The marginal log likelihood of the data given the prior and posterior distribution hyperparameters
    fn marginal_log_likelihood(
        prior: &Self::HyperParams,
        post: &Self::HyperParams,
        stats: &Self::SuffStats,
    ) -> f64;

    /// Compute the posterior predictive log likelihood of the data given the posterior hyperparameters.
    /// This is the log likelihood of the data given the posterior hyperparameters.
    ///
    /// # Arguments
    ///
    /// * `post`: the hyperparameters of the posterior distribution
    /// * `data`: the data
    ///
    /// # Returns
    /// The posterior predictive log likelihood of the data given the posterior distribution hyperparameters
    fn posterior_predictive<S: Storage<f64, Dynamic, Dynamic>>(
        post: &Self::HyperParams,
        data: &Matrix<f64, Dynamic, Dynamic, S>,
    ) -> f64;
}

pub trait PriorHyperParams {
    /// Returns empty distribution parameters.
    fn default(dim: usize) -> Self;
}

pub trait FromData {
    /// Create the distribution parameters from the data.
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
    /// Sample parameters of a normal distribution from the normal conjugate prior distribution.
    fn sample<R: Rng + ?Sized>(prior: &Self::HyperParams, rng: &mut R) -> MultivariateNormal;
}