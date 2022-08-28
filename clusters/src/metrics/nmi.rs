#![allow(clippy::ptr_arg)]
use std::collections::HashMap;
use num_traits::{FromPrimitive, PrimInt};
use num_traits::real::Real;
use simba::scalar::SupersetOf;

fn unique_with_indices<T: PrimInt>(data: &[T]) -> (Vec<T>, Vec<usize>) {
    let mut unique = data.to_vec();
    unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique.dedup();

    let mut index = HashMap::with_capacity(unique.len());
    for (i, u) in unique.iter().enumerate() {
        index.insert(u.to_i64().unwrap(), i);
    }

    let mut unique_index = Vec::with_capacity(data.len());
    for idx in 0..data.len() {
        unique_index.push(index[&data.get(idx).unwrap().to_i64().unwrap()]);
    }

    (unique, unique_index)
}

pub fn contingency_matrix<T: PrimInt>(
    labels_true: &[T],
    labels_pred: &[T],
) -> Vec<Vec<usize>> {
    let (classes, class_idx) = unique_with_indices(labels_true);
    let (clusters, cluster_idx) = unique_with_indices(labels_pred);

    let mut contingency_matrix = Vec::with_capacity(classes.len());

    for _ in 0..classes.len() {
        contingency_matrix.push(vec![0; clusters.len()]);
    }

    for i in 0..class_idx.len() {
        contingency_matrix[class_idx[i]][cluster_idx[i]] += 1;
    }

    contingency_matrix
}

pub fn entropy<T: PrimInt + FromPrimitive>(data: &[T]) -> Option<f64> {
    let mut bincounts = HashMap::with_capacity(data.len());

    for e in data.iter() {
        let k = e.to_i64().unwrap();
        bincounts.insert(k, bincounts.get(&k).unwrap_or(&0.0) + 1.0);
    }

    let mut entropy = 0.0;
    let sum: f64 = bincounts.values().cloned().sum();

    for &c in bincounts.values() {
        if c > 0.0 {
            let pi = c;
            entropy -= (pi / sum) * (pi.ln() - sum.ln());
        }
    }

    Some(entropy)
}

pub fn mutual_info_score<T: PrimInt + FromPrimitive>(contingency: &[Vec<usize>]) -> f64 {
    let mut contingency_sum = 0;
    let mut pi = vec![0; contingency.len()];
    let mut pj = vec![0; contingency[0].len()];
    let (mut nzx, mut nzy, mut nz_val) = (Vec::new(), Vec::new(), Vec::new());

    for r in 0..contingency.len() {
        for (c, pj_c) in pj.iter_mut().enumerate().take(contingency[0].len()) {
            contingency_sum += contingency[r][c];
            pi[r] += contingency[r][c];
            *pj_c += contingency[r][c];
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
        result += (contingency_nm[i] * (log_contingency_nm[i] - contingency_sum_ln))
            + contingency_nm[i] * log_outer[i]
    }

    result.max(0.0)
}

pub fn normalized_mutual_info_score<T: PrimInt + FromPrimitive>(
    labels_true: &[T],
    labels_pred: &[T],
) -> f64 {
    let mi = mutual_info_score::<usize>(&contingency_matrix(labels_true, labels_pred));

    if mi == 0.0 {
        return 0.0;
    }

    let h_true = entropy(labels_true).unwrap();
    let h_pred = entropy(labels_pred).unwrap();

    2.0 * mi / (h_true + h_pred)
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
        let s: f64 = mutual_info_score::<usize>(&contingency_matrix(&v1, &v2));

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
