use nalgebra::{DVector};
use rand::distributions::{Distribution};
use rand::Rng;
use statrs::distribution::{Dirichlet};

/// Samples the dirichlet process using the stick breaking approach.
///
/// # Arguments:
///
/// * `counts`: the number of observations in each cluster
/// * `alpha`: the probability of a new cluster being created
/// * `rng`: a random number generator
///
/// # Returns:
///
/// A vector of weights for each cluster.
///
/// # Example:
/// ```
/// use mixturs::stats::stick_breaking_sample;
///
/// let mut rng = rand::thread_rng();
/// let counts = vec![1.0, 2.0, 3.0];
/// let alpha = 1.0;
/// let weights = stick_breaking_sample(&counts, alpha, &mut rng);
/// ```
pub fn stick_breaking_sample(
    counts: &[f64], alpha: f64, rng: &mut impl Rng
) -> Vec<f64> {
    let cluster_weights = if counts.len() > 1 {
        let dir = Dirichlet::new(counts.to_vec()).unwrap();
        dir.sample(rng)
    } else {
        DVector::from_element(1, 1.0)
    } * (1.0 - alpha);

    if alpha != 0.0 {
        let mut weights = vec![0f64; counts.len() + 1];
        weights[1..counts.len() + 1].copy_from_slice(cluster_weights.as_slice());
        weights[0] = alpha;
        weights
    } else {
        cluster_weights.as_slice().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use statrs::assert_almost_eq;

    #[test]
    fn test_stick_breaking_sample() {
        let counts = vec![1.0, 2.0, 3.0];
        let alpha = 0.5;
        let mut rng = rand::thread_rng();
        let weights = super::stick_breaking_sample(&counts, alpha, &mut rng);
        assert_eq!(weights[0], 0.5);
        assert_almost_eq!(weights.iter().sum::<f64>(), 1.0, 1e-6);

        let counts = vec![2.0, 1000.0];
        let weights = super::stick_breaking_sample(&counts, alpha, &mut rng);
        assert_eq!(weights[0], 0.5);
        assert!(weights[1] < weights[2]);
    }
}