use std::collections::HashMap;
use nalgebra::RowDVector;
use statrs::statistics::Statistics;
use crate::metrics::{EvalData, Metric};
use crate::params::thin::{hard_assignment, MixtureParams, SuperMixtureParams, ThinParams};

pub fn aic(
    dim: usize,
    n_params: usize,
    avg_log_likelihood: f64,
) -> f64 {
    -2.0 * avg_log_likelihood * dim as f64 + 2.0 * n_params as f64
}

pub fn bic(
    dim: usize,
    n_params: usize,
    avg_log_likelihood: f64,
) -> f64 {
    -2.0 * avg_log_likelihood * dim as f64 + n_params as f64 * (dim as f64).ln()
}


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

