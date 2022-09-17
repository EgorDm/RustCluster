use std::collections::HashMap;
use nalgebra::RowDVector;
use statrs::statistics::Statistics;
use crate::metrics::{EvalData, Metric};
use crate::params::thin::{hard_assignment, MixtureParams, SuperMixtureParams, ThinParams};

/// `aic` computes the Akaike Information Criterion (AIC) for a model
///
/// # Arguments:
///
/// * `dim`: the number of data points
/// * `n_params`: The number of parameters in the model.
/// * `avg_log_likelihood`: The average log likelihood of the model.
///
/// # Returns:
///
/// The Akaike Information Criterion (AIC)
///
/// # Example:
/// ```
/// use statrs::assert_almost_eq;
/// use mixturs::metrics::aic;
///
/// let aic = aic(100, 10, 0.0);
/// assert_almost_eq!(aic, 20.0, 1e-4);
/// ```
pub fn aic(
    dim: usize,
    n_params: usize,
    avg_log_likelihood: f64,
) -> f64 {
    -2.0 * avg_log_likelihood * dim as f64 + 2.0 * n_params as f64
}

/// It takes the number of parameters, the number of data points, and the average log likelihood, and returns the Bayesian
/// Information Criterion (BIC)
///
/// # Arguments:
///
/// * `dim`: the number of data points
/// * `n_params`: the number of parameters in the model
/// * `avg_log_likelihood`: The average log likelihood of the model.
///
/// # Returns:
///
/// The Bayesian Information Criterion (BIC)
///
/// # Example:
/// ```
/// use statrs::assert_almost_eq;
/// use mixturs::metrics::bic;
///
/// let bic = bic(100, 10, 0.0);
/// assert_almost_eq!(bic, 46.0517018, 1e-4);
/// ```
pub fn bic(
    dim: usize,
    n_params: usize,
    avg_log_likelihood: f64,
) -> f64 {
    -2.0 * avg_log_likelihood * dim as f64 + n_params as f64 * (dim as f64).ln()
}

/// Akaike Information Criterion measure
pub struct AIC;

impl<P: ThinParams> Metric<P> for AIC {
    fn compute(
        &mut self,
        _i: usize,
        data: &EvalData,
        params: &P,
        metrics: &mut HashMap<String, f64>
    ) {
        let log_likelihood = SuperMixtureParams(params).log_likelihood(data.points.clone_owned());
        let mut labels = RowDVector::zeros(data.points.ncols());
        hard_assignment(&log_likelihood, labels.as_mut_slice());

        let avg_log_likelihood = log_likelihood.column_iter().map(|col| col.max()).mean();
        let score = aic(data.points.nrows(), params.n_params(), avg_log_likelihood);

        metrics.insert("aic".to_string(), score);
    }
}

/// Bayesian Information Criterion measure
pub struct BIC;

impl<P: ThinParams> Metric<P> for BIC {
    fn compute(
        &mut self,
        _i: usize,
        data: &EvalData,
        params: &P,
        metrics: &mut HashMap<String, f64>
    ) {
        let log_likelihood = SuperMixtureParams(params).log_likelihood(data.points.clone_owned());
        let mut labels = RowDVector::zeros(data.points.ncols());
        hard_assignment(&log_likelihood, labels.as_mut_slice());

        let avg_log_likelihood = log_likelihood.column_iter().map(|col| col.max()).mean();
        let score = bic(data.points.nrows(), params.n_params(), avg_log_likelihood);

        metrics.insert("bic".to_string(), score);
    }
}

