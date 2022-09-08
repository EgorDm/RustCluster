use std::collections::hash_map::Entry;
use std::mem::MaybeUninit;
use nalgebra::{DefaultAllocator, Dim, DMatrix, Dynamic, Matrix, OMatrix, OVector, Scalar, Storage, StorageMut, U1, Vector};
use std::collections::HashMap;
use num_traits::{FromPrimitive, One, PrimInt};
use std::hash::Hash;
use itertools::izip;
use nalgebra::allocator::Allocator;
use nalgebra::constraint::{SameNumberOfRows, ShapeConstraint};
use simba::scalar::RealField;

pub fn each_ref<T, const N: usize>(data: &[T; N]) -> [&T; N] {
    // Unlike in `map`, we don't need a guard here, as dropping a reference
    // is a noop.
    let mut out = [MaybeUninit::uninit(); N];
    for (src, dst) in data.iter().zip(&mut out) {
        dst.write(src);
    }

    // SAFETY: All elements of `dst` are properly initialized and
    // `MaybeUninit<T>` has the same layout as `T`, so this cast is valid.
    unsafe { (&mut out as *mut _ as *mut [&T; N]).read() }
}


pub fn unique_with_indices<T: Copy + Hash + Eq + Ord>(data: &[T], sorted: bool) -> (Vec<T>, Vec<usize>) {
    let mut index = HashMap::new();
    let mut unique = Vec::new();

    for u in data {
        if !index.contains_key(u) {
            unique.push(*u);
            index.insert(*u, 0);
        }
    }

    if sorted {
        unique.sort();
    }
    for (i, u) in unique.iter().enumerate() {
        index.insert(*u, i);
    }

    let mut unique_index = Vec::with_capacity(data.len());
    for d in data {
        unique_index.push(index[d]);
    }

    (unique, unique_index)
}

pub fn bincount<T: Copy + Hash + Eq>(data: &[T]) -> HashMap<T, usize> {
    let mut counts = HashMap::new();
    for &u in data {
        match counts.entry(u) {
            Entry::Occupied(mut e) => {
                *e.get_mut() += 1;
            }
            Entry::Vacant(mut e) => {
                e.insert(1);
            }
        }
    }
    counts
}

pub fn row_normalize_log_weights(
    weights: &mut DMatrix<f64>
) {
    for mut row in weights.row_iter_mut() {
        let max = row.max();
        for x in row.iter_mut() {
            *x = (*x - max).exp();
        }
    }
}


pub fn col_normalize_log_weights(
    weights: &mut DMatrix<f64>
) {
    for mut col in weights.column_iter_mut() {
        let max = col.max();
        for x in col.iter_mut() {
            *x = (*x - max).exp();
        }
    }
}

pub fn col_broadcast_add<Real, R, C, SM, SV>(
    mut arr: Matrix<Real, R, C, SM>,
    vec: &Matrix<Real, C, U1, SV>,
) -> Matrix<Real, R, C, SM>
    where
        Real: RealField,
        R: Dim,
        C: Dim,
        DefaultAllocator: Allocator<Real, R, C>,
        DefaultAllocator: Allocator<Real, C>,
        SM: StorageMut<Real, R, C>,
        SV: Storage<Real, C>,
{
    assert_eq!(arr.nrows(), vec.len());

    let mut nrows = arr.nrows();
    let mut out = arr;

    for mut col in out.column_iter_mut() {
        for i in 0..nrows {
            unsafe { // Rows are already checked
                *col.get_unchecked_mut(i) += vec.get_unchecked(i).clone();
            }
        }
    }

    out
}

pub fn col_broadcast_sub<Real, R, C, SM, SV>(
    mut arr: Matrix<Real, R, C, SM>,
    vec: &Matrix<Real, C, U1, SV>,
) -> Matrix<Real, R, C, SM>
    where
        Real: RealField,
        R: Dim,
        C: Dim,
        DefaultAllocator: Allocator<Real, R, C>,
        DefaultAllocator: Allocator<Real, C>,
        SM: StorageMut<Real, R, C>,
        SV: Storage<Real, C>,
{
    assert_eq!(arr.nrows(), vec.len());

    let mut nrows = arr.nrows();
    let mut out = arr;

    for mut col in out.column_iter_mut() {
        for i in 0..nrows {
            unsafe { // Rows are already checked
                *col.get_unchecked_mut(i) -= vec.get_unchecked(i).clone();
            }
        }
    }

    out
}

pub fn col_scatter<T, R, CO, CI, SO, SI>(
    out: &mut Matrix<T, R, CO, SO>,
    indices: &[usize],
    values: &Matrix<T, R, CI, SI>,
)
    where
        T: Scalar,
        R: Dim,
        CO: Dim,
        CI: Dim,
        DefaultAllocator: Allocator<T, R, CO>,
        DefaultAllocator: Allocator<T, R, CI>,
        SO: StorageMut<T, R, CO>,
        SI: Storage<T, R, CI>,
{
    assert_eq!(out.nrows(), values.nrows());
    assert_eq!(values.ncols(), indices.len());

    for (i, &idx) in indices.iter().enumerate() {
        out.column_mut(idx).copy_from(&values.column(i));
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};
    use num_traits::real::Real;
    use crate::stats::tests::test_almost_mat;
    use crate::utils::col_broadcast_sub;
    use crate::utils::data::{col_broadcast_add};

    #[test]
    fn test_unique_with_indices() {
        let data = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1];
        let (unique, unique_index) = super::unique_with_indices(&data, false);
        assert_eq!(unique, vec![1, 2, 3, 4, 5]);
        assert_eq!(unique_index, vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0]);
    }

    #[test]
    fn test_bincount() {
        let data = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5];
        let counts = super::bincount(&data);
        let mut bincounts: Vec<_> = counts.into_iter().collect();
        bincounts.sort();
        assert_eq!(bincounts, vec![(1, 2), (2, 2), (3, 2), (4, 2), (5, 2)]);
    }

    #[test]
    fn test_normalize_log_weights() {
        let mut weights = DMatrix::from_row_slice(3, 3, &[
            1.0f64, 2.0, 4.0,
            1.0, 2.0, 4.0,
            1.0, 2.0, 4.0,
        ]);
        weights.iter_mut().for_each(|x| *x = x.ln());
        super::row_normalize_log_weights(&mut weights);
        test_almost_mat(&weights, &DMatrix::from_row_slice(3, 3, &[
            0.25, 0.5, 1.0,
            0.25, 0.5, 1.0,
            0.25, 0.5, 1.0,
        ]), 1e-4);

        let mut weights = DMatrix::from_row_slice(3, 3, &[
            1.0f64, 2.0, 4.0,
            1.0, 2.0, 4.0,
            1.0, 2.0, 4.0,
        ]);
        weights.iter_mut().for_each(|x| *x = x.ln());
        super::col_normalize_log_weights(&mut weights);
        test_almost_mat(&weights, &DMatrix::from_row_slice(3, 3, &[
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ]), 1e-4);
    }

    #[test]
    fn test_broadcast_add() {
        let x = DMatrix::<f64>::from_vec(3, 4, vec![
            1.0, 5.0, 100.0,
            2.0, 6.0, 200.0,
            3.0, 7.0, 300.0,
            4.0, 8.0, 400.0,
        ]);
        let v = DVector::<f64>::from_vec(vec![-3.0, -4.0, -5.0]);
        let actual = col_broadcast_add(x, &v);

        let expected = DMatrix::<f64>::from_vec(3, 4, vec![
            -2.0, 1.0, 95.0,
            -1.0, 2.0, 195.0,
            0.0, 3.0, 295.0,
            1.0,  4.0, 395.0,
        ]);

        assert!(actual == expected);
    }

    #[test]
    fn test_broadcast_sub() {
        let x = DMatrix::<f64>::from_vec(3, 4, vec![
            1.0, 5.0, 100.0,
            2.0, 6.0, 200.0,
            3.0, 7.0, 300.0,
            4.0, 8.0, 400.0,
        ]);
        let v = DVector::<f64>::from_vec(vec![3.0, 4.0, 5.0]);
        let actual = col_broadcast_sub(x, &v);

        let expected = DMatrix::<f64>::from_vec(3, 4, vec![
            -2.0, 1.0, 95.0,
            -1.0, 2.0, 195.0,
            0.0, 3.0, 295.0,
            1.0,  4.0, 395.0,
        ]);

        assert!(actual == expected);
    }

    #[test]
    fn test_col_scatter() {
        let mut out = DMatrix::<f64>::from_vec(3, 4, vec![
            1.0, 5.0, 100.0,
            2.0, 6.0, 200.0,
            3.0, 7.0, 300.0,
            4.0, 8.0, 400.0,
        ]);
        let indices = vec![0, 2, 3];
        let values = DMatrix::<f64>::from_vec(3, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        super::col_scatter(&mut out, &indices, &values);
        let expected = DMatrix::<f64>::from_vec(3, 4, vec![
            1.0, 2.0, 3.0,
            2.0, 6.0, 200.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        println!("{} {}", out, expected);
        test_almost_mat(&out, &expected, 1e-10);
    }
}