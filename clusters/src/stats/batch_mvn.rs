use std::ops::{AddAssign, MulAssign};
use nalgebra::{DefaultAllocator, Dim, DVector, Dynamic, Matrix, Storage, StorageMut};
use nalgebra::allocator::Allocator;
use statrs::distribution::MultivariateNormal;
use statrs::statistics::MeanN;
use crate::utils::{col_broadcast_add, col_broadcast_sub};

pub trait ContinuousBatchwise<K> {
    fn batchwise_pdf(
        &self,
        xs: K,
    ) -> DVector<f64>;

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
    fn batchwise_pdf(&self, xs: Matrix<f64, Dynamic, Dynamic, S>) -> DVector<f64> {
        let n_points = xs.ncols();
        let dvs = col_broadcast_sub(xs, self.mu()); // broadcast subtract?

        let mut left = self.precision() * &dvs;
        left.component_mul_assign(&dvs);

        let mut exp_term = DVector::from_iterator(
            n_points,
            left.column_iter().map(|col| (-0.5 * col.sum()).exp() )
        );
        exp_term * self.pdf_const()
    }

    fn batchwise_ln_pdf(&self, xs: Matrix<f64, Dynamic, Dynamic, S>) -> DVector<f64> {
        let n_points = xs.ncols();
        let dvs = col_broadcast_sub(xs, self.mu()); // broadcast subtract?

        let mut left = self.precision() * &dvs;
        left.component_mul_assign(&dvs);

        let pdf_const = self.pdf_const().ln();
        let mut exp_term = DVector::from_iterator(
            n_points,
            left.column_iter().map(|col| -0.5 * col.sum() + pdf_const)
        );
        exp_term
    }
}


#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};
    use statrs::distribution::MultivariateNormal;
    use crate::stats::batch_mvn::ContinuousBatchwise;
    use crate::stats::tests::test_almost_mat;

    #[test]
    fn test_pdf() {
        let mvn = MultivariateNormal::new(
            DVector::from_vec(vec![0.0, 0.0]),
            DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
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
            DVector::from_vec(vec![0.0, 0.0]),
            DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
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