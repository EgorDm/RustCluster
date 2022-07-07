use nalgebra::{DefaultAllocator, Dim, DMatrix, Field, Matrix, RawStorage, Scalar, U1};
use simba::scalar::{SupersetOf};
use nalgebra::allocator::Allocator;

pub fn row_covariance(data: &DMatrix<f64>) -> DMatrix<f64> {
    let mean = data.row_mean();
    let mut centered = data.clone();
    centered.row_iter_mut().for_each(|mut row| {
        row -= &mean;
    });
    centered.transpose() * centered / data.nrows() as f64
}

pub fn col_covariance(data: &DMatrix<f64>) -> DMatrix<f64> {
    let mean = data.column_mean();
    let mut centered = data.clone();
    centered.column_iter_mut().for_each(|mut col| {
        col -= &mean;
    });
    &centered * centered.transpose() / data.ncols() as f64
}


#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use nalgebra::{DefaultAllocator, Dim, DMatrix, OMatrix};
    use statrs::assert_almost_eq;
    use crate::stats::{col_covariance, row_covariance};

    fn test_almost_mat<R1: Dim, C1: Dim, R2: Dim, C2: Dim>(
        value: &OMatrix<f64, R1, C1>,
        expected: &OMatrix<f64, R2, C2>,
        acc: f64,
    )
        where DefaultAllocator: nalgebra::allocator::Allocator<f64, R1, C1>,
              DefaultAllocator: nalgebra::allocator::Allocator<f64, R2, C2> {
        for i in 0..value.nrows() {
            for j in 0..value.ncols() {
                assert_almost_eq!(expected[(i, j)], value[(i, j)], acc);
            }
        }
    }

    fn points1() -> DMatrix<f64> {
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
        ])
    }

    #[test]
    fn test_covariance() {
        dbg!(&row_covariance(&points1()));

        test_almost_mat(
            &row_covariance(&points1()),
            &DMatrix::from_row_slice(3, 3, &[
                0.0549, 0.0115, 0.0008,
                0.0115, 0.1061, 0.0116,
                0.0008, 0.0116, 0.0825,
            ]),
            0.0001,
        );

        test_almost_mat(
            &col_covariance(&points1().transpose()),
            &DMatrix::from_row_slice(3, 3, &[
                0.0549, 0.0115, 0.0008,
                0.0115, 0.1061, 0.0116,
                0.0008, 0.0116, 0.0825,
            ]),
            0.0001,
        );
    }
}