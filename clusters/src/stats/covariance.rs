use nalgebra::{DefaultAllocator, Dim, Matrix, OMatrix, RawStorage, RawStorageMut, Scalar, Storage, U1};
use nalgebra::allocator::Allocator;
use simba::scalar::{Field, SupersetOf};
use statrs::assert_almost_eq;

pub trait Covariance<T, R: Dim, C: Dim> {
    fn row_cov(&self) -> OMatrix<T, C, C>
        where DefaultAllocator: Allocator<T, C, C>;

    fn col_cov(&self) -> OMatrix<T, R, R>
        where DefaultAllocator: Allocator<T, R, R>;
}

pub trait CovarianceMut<T, R: Dim, C: Dim> {
    fn row_cov_mut(self) -> OMatrix<T, C, C>
        where DefaultAllocator: Allocator<T, C, C>;

    fn col_cov_mut(self) -> OMatrix<T, R, R>
        where DefaultAllocator: Allocator<T, R, R>;
}

impl<
    T: Scalar, R: Dim, C: Dim, S: RawStorage<T, R, C> + Storage<T, R, C>
> Covariance<T, R, C> for Matrix<T, R, C, S>
    where T: Field + SupersetOf<f64>,
          DefaultAllocator: Allocator<T, U1, R>,
          DefaultAllocator: Allocator<T, U1, C>,
          DefaultAllocator: Allocator<T, R, U1>,
          DefaultAllocator: Allocator<T, C, U1>,
          DefaultAllocator: Allocator<T, C, R>,
          DefaultAllocator: Allocator<T, R, C>,
{
    fn row_cov(&self) -> OMatrix<T, C, C>
        where DefaultAllocator: Allocator<T, C, C>
    {
        self.transpose().col_cov_mut()
    }

    fn col_cov(&self) -> OMatrix<T, R, R>
        where DefaultAllocator: Allocator<T, R, R>
    {
        self.clone_owned().col_cov_mut()
    }
}

impl<
    T: Scalar, R: Dim, C: Dim, S: RawStorage<T, R, C> + RawStorageMut<T, R, C> + Storage<T, R, C>
> CovarianceMut<T, R, C> for Matrix<T, R, C, S>
    where T: Field + SupersetOf<f64>,
          DefaultAllocator: Allocator<T, U1, C>,
          DefaultAllocator: Allocator<T, R, U1>,
          DefaultAllocator: Allocator<T, C, R>,
          DefaultAllocator: Allocator<T, R, C>,
{
    fn row_cov_mut(mut self) -> OMatrix<T, C, C>
        where DefaultAllocator: Allocator<T, C, C>
    {
        let mean = self.row_mean();
        for mut row in self.row_iter_mut() {
            row -= &mean;
        }
        let n = self.nrows();
        self.transpose() * self / T::from_subset(&(n as f64))
    }

    fn col_cov_mut(mut self) -> OMatrix<T, R, R>
        where DefaultAllocator: Allocator<T, R, R>
    {
        let mean = self.column_mean();
        for mut col in self.column_iter_mut() {
            col -= &mean;
        }
        let n = self.ncols();
        let self_t = self.transpose();
        self * self_t / T::from_subset(&(n as f64))
    }
}

pub fn test_almost_mat<R1: Dim, C1: Dim, R2: Dim, C2: Dim>(
    value: &OMatrix<f64, R1, C1>,
    expected: &OMatrix<f64, R2, C2>,
    acc: f64,
)
    where DefaultAllocator: nalgebra::allocator::Allocator<f64, R1, C1>,
          DefaultAllocator: nalgebra::allocator::Allocator<f64, R2, C2> {

    assert_eq!(value.nrows(), expected.nrows());
    assert_eq!(value.ncols(), expected.ncols());
    for i in 0..value.nrows() {
        for j in 0..value.ncols() {
            assert_almost_eq!(expected[(i, j)], value[(i, j)], acc);
        }
    }
}


#[cfg(test)]
pub mod tests {
    use nalgebra::{DefaultAllocator, Dim, DMatrix, OMatrix};
    use statrs::assert_almost_eq;
    use crate::stats::covariance::Covariance;

    pub fn test_almost_mat<R1: Dim, C1: Dim, R2: Dim, C2: Dim>(
        value: &OMatrix<f64, R1, C1>,
        expected: &OMatrix<f64, R2, C2>,
        acc: f64,
    )
        where DefaultAllocator: nalgebra::allocator::Allocator<f64, R1, C1>,
              DefaultAllocator: nalgebra::allocator::Allocator<f64, R2, C2> {

        assert_eq!(value.nrows(), expected.nrows());
        assert_eq!(value.ncols(), expected.ncols());
        for i in 0..value.nrows() {
            for j in 0..value.ncols() {
                assert_almost_eq!(expected[(i, j)], value[(i, j)], acc);
            }
        }
    }

    pub fn points1() -> DMatrix<f64> {
        DMatrix::from_row_slice(10, 3, &[
            0.0303, 0.1105, 0.0289,
            0.3770, 0.0281, 0.1693,
            0.8688, 0.1841, 0.0224,
            0.5387, 0.9276, 0.4369,
            0.6116, 0.8197, 0.4987,
            0.4687, 0.2254, 0.7995,
            0.0860, 0.7231, 0.2202,
            0.2485, 0.0035, 0.7435,
            0.3800, 0.3961, 0.7620,
            0.4416, 0.1462, 0.1969,
        ]).transpose()
    }

    #[test]
    fn test_covariance() {
        test_almost_mat(
            &points1().col_cov(),
            &DMatrix::from_row_slice(3, 3, &[
                0.0549, 0.0115, 0.0008,
                0.0115, 0.1061, 0.0116,
                0.0008, 0.0116, 0.0825,
            ]),
            0.0001,
        );

        test_almost_mat(
            &points1().transpose().row_cov(),
            &DMatrix::from_row_slice(3, 3, &[
                0.0549, 0.0115, 0.0008,
                0.0115, 0.1061, 0.0116,
                0.0008, 0.0116, 0.0825,
            ]),
            0.0001,
        );
    }
}