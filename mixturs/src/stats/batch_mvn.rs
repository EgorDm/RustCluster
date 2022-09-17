use nalgebra::{DefaultAllocator, DMatrix, DVector, Dynamic, Matrix, StorageMut};
use nalgebra::allocator::Allocator;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use statrs::distribution::MultivariateNormal;
use crate::utils::{col_broadcast_sub};

/// Allows implementation of batchwise probability density functions.
/// As the calculations are batched together it is usually faster than calculating the pdf for each sample individually.
pub trait ContinuousBatchwise<K> {
    /// Returns the probability density function for each `x` (column) in `xs` for a given
    /// distribution.
    /// May panic depending on the implementor.
    fn batchwise_pdf(
        &self,
        xs: K,
    ) -> DVector<f64>;

    /// Returns the log of the probability density function calculated for each `x` (column) in `xs`
    /// for a given distribution.
    /// May panic depending on the implementor.
    fn batchwise_ln_pdf(
        &self,
        xs: K,
    ) -> DVector<f64>;
}

impl<S> ContinuousBatchwise<Matrix<f64, Dynamic, Dynamic, S>> for MultivariateNormal
where
    DefaultAllocator: Allocator<f64, Dynamic, Dynamic>,
    DefaultAllocator: Allocator<f64, Dynamic>,
    S: StorageMut<f64, Dynamic, Dynamic>,
{
    /// Calculates the probability density function for the multivariate
    /// normal distribution for each `x` (column) in `xs`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (2 * π) ^ (-k / 2) * det(Σ) ^ (1 / 2) * e ^ ( -(1 / 2) * transpose(x - μ) * inv(Σ) * (x - μ))
    /// ```
    ///
    /// where `μ` is the mean, `inv(Σ)` is the precision matrix, `det(Σ)` is the determinant
    /// of the covariance matrix, and `k` is the dimension of the distribution
    fn batchwise_pdf(&self, xs: Matrix<f64, Dynamic, Dynamic, S>) -> DVector<f64> {
        let n_points = xs.ncols();
        let dvs = col_broadcast_sub(xs, self.mu()); // broadcast subtract?

        let mut left = self.precision() * &dvs;
        left.component_mul_assign(&dvs);

        let exp_term = DVector::from_iterator(
            n_points,
            left.column_iter().map(|col| (-0.5 * col.sum()).exp() )
        );
        exp_term * self.pdf_const()
    }

    /// Calculates the log probability density function for the multivariate
    /// normal distribution for each `x` (column) in `xs`. Equivalent to pdf(x).ln().
    fn batchwise_ln_pdf(&self, xs: Matrix<f64, Dynamic, Dynamic, S>) -> DVector<f64> {
        let n_points = xs.ncols();
        let dvs = col_broadcast_sub(xs, self.mu()); // broadcast subtract?

        let mut left = self.precision() * &dvs;
        left.component_mul_assign(&dvs);

        let pdf_const = self.pdf_const().ln();
        let exp_term = DVector::from_iterator(
            n_points,
            left.column_iter().map(|col| -0.5 * col.sum() + pdf_const)
        );
        exp_term
    }
}

#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
#[serde(remote = "MultivariateNormal")]
struct MultivariateNormalDef {
    #[serde(getter = "MultivariateNormal::mu")]
    mean: DVector<f64>,
    #[serde(getter = "MultivariateNormal::cov")]
    cov: DMatrix<f64>,
}

#[cfg(feature = "serde")]
impl From<MultivariateNormalDef> for MultivariateNormal {
    fn from(def: MultivariateNormalDef) -> Self {
        MultivariateNormal::new(
            def.mean.data.into(),
            def.cov.data.into()
        ).unwrap()
    }
}



#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};
    use statrs::distribution::MultivariateNormal;
    use crate::stats::batch_mvn::{ContinuousBatchwise};
    use crate::stats::tests::test_almost_mat;

    #[test]
    fn test_pdf() {
        let mvn = MultivariateNormal::new(
            DVector::from_vec(vec![0.0, 0.0]).data.into(),
            DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]).data.into(),
        ).unwrap();

        let data = DMatrix::from_vec(2, 4, vec![
            1., 1.,
            1., 2.,
            1., 10.,
            10., 10.,
        ]);

        let actual = mvn.batchwise_pdf(data);

        let expected = DVector::from_vec(vec![
            0.05854983152431917,
            0.013064233284684921,
            1.8618676045881531e-23,
            5.920684802611216e-45,
        ]);

        test_almost_mat(&actual, &expected, 1e-6);
    }

    #[test]
    fn test_ln_pdf() {
        let mvn = MultivariateNormal::new(
            DVector::from_vec(vec![0.0, 0.0]).data.into(),
            DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]).data.into(),
        ).unwrap();

        let data = DMatrix::from_vec(2, 4, vec![
            1., 1.,
            1., 2.,
            1., 10.,
            10., 10.,
        ]);

        let actual = mvn.batchwise_ln_pdf(data);

        let expected = DVector::from_vec(vec![
            (0.05854983152431917f64   ).ln(),
            (0.013064233284684921f64  ).ln(),
            (1.8618676045881531e-23f64).ln(),
            (5.920684802611216e-45f64 ).ln(),
        ]);

        test_almost_mat(&actual, &expected, 1e-6);
    }
}