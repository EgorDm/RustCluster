#![allow(clippy::ptr_arg)]

use std::collections::HashMap;
use std::hash::Hash;
use itertools::Itertools;
use num_traits::{FromPrimitive, PrimInt};
use num_traits::real::Real;
use simba::scalar::SupersetOf;
use crate::metrics::{EvaluationData, Metric};
use crate::params::thin::{MixtureParams, SuperMixtureParams, ThinParams};
use crate::utils::{unique_with_indices};

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

pub struct NMI;

impl<P: ThinParams> Metric<P> for NMI {
    fn compute(
        &mut self,
        data: &EvaluationData,
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
