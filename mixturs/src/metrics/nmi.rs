#![allow(clippy::ptr_arg)]

use std::collections::HashMap;
use std::hash::Hash;
use itertools::Itertools;
use crate::metrics::{EvalData, Metric};
use crate::params::thin::{MixtureParams, SuperMixtureParams, ThinParams};
use crate::utils::{unique_with_indices};


/// It takes two vectors of labels, one for the true labels and one for the predicted labels, and returns a matrix where the
/// rows are the true labels and the columns are the predicted labels. The value at each row/column intersection is the
/// number of times that true label was predicted as that predicted label
///
/// # Arguments:
///
/// * `labels_true`: The true labels of the data.
/// * `labels_pred`: The predicted labels
///
/// # Returns:
///
/// A contingency matrix.
///
/// # Example:
/// ```
/// use mixturs::metrics::contingency_matrix;
/// let labels_true = vec![1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3];
/// let labels_pred = vec![1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 2, 2];
///
/// let contingency = contingency_matrix(&labels_true, &labels_pred);
/// assert_eq!(contingency, vec![vec![5, 1, 0], vec![1, 4, 1], vec![0, 2, 3]]);
pub fn contingency_matrix<T: Copy + Hash + Eq + Ord>(
    labels_true: &[T],
    labels_pred: &[T],
) -> Vec<Vec<usize>> {
    let (classes, class_idx) = unique_with_indices(labels_true, true);
    let (clusters, cluster_idx) = unique_with_indices(labels_pred, true);

    let mut contingency_matrix = vec![vec![0; clusters.len()]; classes.len()];
    for i in 0..class_idx.len() {
        contingency_matrix[class_idx[i]][cluster_idx[i]] += 1;
    }

    contingency_matrix
}

pub fn entropy<T: Copy + Hash + Eq>(data: &[T]) -> Option<f64> {
    let bincounts = data.iter().cloned().counts();
    let sum = bincounts.values().cloned().sum::<usize>() as f64;

    let mut entropy = 0.0;
    for &c in bincounts.values() {
        if c > 0 {
            let pi = c as f64;
            entropy -= (pi / sum) * (pi.ln() - sum.ln());
        }
    }

    Some(entropy)
}

/// Calculates the mutual information score of two variables, given a contingency table
///
/// # Arguments:
///
/// * `contingency`: The contingency table.
///
/// # Returns:
///
/// The mutual information score.
///
/// # Example:
/// ```
/// use statrs::assert_almost_eq;
/// use mixturs::metrics::mutual_info_score;
///
/// let contingency = vec![vec![5, 1, 0], vec![1, 4, 1], vec![0, 2, 3]];
/// let mi = mutual_info_score(&contingency);
/// assert_almost_eq!(mi, 0.41022, 1e-4);
/// ```
pub fn mutual_info_score(
    contingency: &[Vec<usize>]
) -> f64 {
    let mut contingency_sum = 0;
    let mut pi = vec![0; contingency.len()];
    let mut pj = vec![0; contingency[0].len()];
    let (mut nzx, mut nzy, mut nz_val) = (Vec::new(), Vec::new(), Vec::new());

    for r in 0..pi.len() {
        for c in 0..pj.len() {
            contingency_sum += contingency[r][c];
            pi[r] += contingency[r][c];
            pj[c] += contingency[r][c];
            if contingency[r][c] > 0 {
                nzx.push(r);
                nzy.push(c);
                nz_val.push(contingency[r][c]);
            }
        }
    }

    let contingency_sum = contingency_sum as f64;
    let contingency_sum_ln = contingency_sum.ln();
    let pi_sum_l = (pi.iter().sum::<usize>() as f64).ln();
    let pj_sum_l = (pj.iter().sum::<usize>() as f64).ln();

    let log_contingency_nm: Vec<f64> = nz_val
        .iter()
        .map(|v| (*v as f64).ln())
        .collect();
    let contingency_nm: Vec<f64> = nz_val
        .iter()
        .map(|v| (*v as f64) / contingency_sum)
        .collect();
    let outer: Vec<usize> = nzx
        .iter()
        .zip(nzy.iter())
        .map(|(&x, &y)| pi[x] * pj[y])
        .collect();
    let log_outer: Vec<f64> = outer
        .iter()
        .map(|&o| -(o as f64).ln() + pi_sum_l + pj_sum_l)
        .collect();

    let mut result = 0.0;
    for i in 0..log_outer.len() {
        result += (contingency_nm[i] * (log_contingency_nm[i] - contingency_sum_ln)) + contingency_nm[i] * log_outer[i]
    }

    result.max(0.0)
}

/// It computes the mutual information between the true and predicted labels, and then normalizes it by the average entropy
/// of the true and predicted labels
///
/// # Arguments:
///
/// * `labels_true`: The true labels of the data.
/// * `labels_pred`: The predicted labels
///
/// # Returns:
///
/// The normalized mutual information score.
///
/// # Example:
/// ```
/// use statrs::assert_almost_eq;
/// use mixturs::metrics::normalized_mutual_info_score;
///
/// let labels_true = vec![1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3];
/// let labels_pred = vec![1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 2, 2];
///
/// let nmi = normalized_mutual_info_score(&labels_true, &labels_pred);
/// assert_almost_eq!(nmi, 0.378349, 1e-4);
/// ```
pub fn normalized_mutual_info_score<T: Copy + Hash + Eq + Ord>(
    labels_true: &[T],
    labels_pred: &[T],
) -> f64 {
    let mi = mutual_info_score(&contingency_matrix(labels_true, labels_pred));
    if mi == 0.0 {
        return 0.0;
    }

    let h_true = entropy(labels_true).unwrap();
    let h_pred = entropy(labels_pred).unwrap();

    2.0 * mi / (h_true + h_pred)
}

/// Normalized mutual information measure
#[derive(Clone)]
pub struct NMI;

impl<P: ThinParams> Metric<P> for NMI {
    fn compute(
        &mut self,
        _i: usize,
        data: &EvalData,
        params: &P,
        metrics: &mut HashMap<String, f64>,
    ) {
        if data.labels.is_none() {
            return;
        }

        let (_, labels) = SuperMixtureParams(params).predict(data.points.clone_owned());
        let score = normalized_mutual_info_score(
            data.labels.as_ref().unwrap().as_slice(),
            labels.as_slice(),
        );

        metrics.insert("nmi".to_string(), score);
    }
}

#[cfg(test)]
mod tests {
    use statrs::assert_almost_eq;
    use super::*;

    #[test]
    fn contingency_matrix_test() {
        let v1 = vec![0, 0, 1, 1, 2, 0, 4];
        let v2 = vec![1, 0, 0, 0, 0, 1, 0];

        assert_eq!(
            vec!(vec!(1, 2), vec!(2, 0), vec!(1, 0), vec!(1, 0)),
            contingency_matrix(&v1, &v2)
        );
    }

    #[test]
    fn entropy_test() {
        let v1 = vec![0, 0, 1, 1, 2, 0, 4];

        assert!((1.2770f64 - entropy(&v1).unwrap()).abs() < 1e-4);
    }

    #[test]
    fn mutual_info_score_test() {
        let v1 = vec![0, 0, 1, 1, 2, 0, 4];
        let v2 = vec![1, 0, 0, 0, 0, 1, 0];
        let s: f64 = mutual_info_score(&contingency_matrix(&v1, &v2));

        assert_almost_eq!(0.3254, s, 1e-4);
    }

    #[test]
    fn normalized_mutual_info_score_test() {
        let v1 = vec![0, 0, 1, 1, 2, 0, 4];
        let v2 = vec![1, 0, 0, 0, 0, 1, 0];
        let s: f64 = normalized_mutual_info_score::<usize>(&v1, &v2);

        assert_almost_eq!(0.34712007071429435, s, 1e-4);
    }
}
