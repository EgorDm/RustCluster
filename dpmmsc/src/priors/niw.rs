use std::ops::Add;
use nalgebra::{DMatrix, DVector, Dynamic, Matrix, Storage};
use rand::distributions::Distribution;
use rand::Rng;
use statrs::consts::LN_PI;
use statrs::distribution::{InverseWishart, MultivariateNormal};
use statrs::function::gamma::mvlgamma;
use crate::priors::{ConjugatePrior, GaussianPrior, PriorHyperParams, SufficientStats};
use crate::stats::row_covariance;

#[derive(Debug, Clone, PartialEq)]
pub struct NIWStats {
    pub n_points: usize,
    pub mean_sum: DVector<f64>,
    pub cov_sum: DMatrix<f64>,
}

impl SufficientStats for NIWStats {
    fn from_data<S: Storage<f64, Dynamic, Dynamic>>(
        data: &Matrix<f64, Dynamic, Dynamic, S>,
    ) -> Self {
        let n_points = data.nrows();
        let mean_sum = data.row_sum().transpose();
        let cov_sum = (data.transpose() * data).symmetric_part();
        Self { n_points, mean_sum, cov_sum }
    }

    fn empty() -> Self {
        Self {
            n_points: 0,
            mean_sum: DVector::zeros(1),
            cov_sum: DMatrix::zeros(1, 1),
        }
    }

    fn n_points(&self) -> usize {
        self.n_points
    }

    fn add(&self, rhs: &Self) -> Self {
        let n_points = self.n_points + rhs.n_points;
        let mean_sum = &self.mean_sum + &rhs.mean_sum;
        let cov_sum = &self.cov_sum + &rhs.cov_sum;
        NIWStats { n_points, mean_sum, cov_sum }
    }
}

impl<'a, 'b> Add<&'b NIWStats> for &'a NIWStats {
    type Output = NIWStats;

    fn add(self, rhs: &'b NIWStats) -> Self::Output {
        let n_points = self.n_points + rhs.n_points;
        let mean_sum = &self.mean_sum + &rhs.mean_sum;
        let cov_sum = &self.cov_sum + &rhs.cov_sum;
        NIWStats { n_points, mean_sum, cov_sum }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NIWParams {
    pub kappa: f64,
    pub mu: DVector<f64>,
    pub nu: f64,
    pub psi: DMatrix<f64>,
}

impl PriorHyperParams for NIWParams {
    fn default(dim: usize) -> Self {
        Self {
            kappa: 1.0,
            mu: DVector::zeros(dim),
            nu: dim as f64 + 3.0,
            psi: DMatrix::identity(dim, dim),
        }
    }
}

impl NIWParams {
    pub fn new(kappa: f64, mu: DVector<f64>, nu: f64, psi: DMatrix<f64>) -> Self {
        NIWParams { kappa, mu, nu, psi }
    }

    pub fn from_data<S: Storage<f64, Dynamic, Dynamic>>(
        kappa: f64,
        nu: f64,
        data: &Matrix<f64, Dynamic, Dynamic, S>,
    ) -> Self {
        let mu = data.row_mean().transpose();
        let psi = row_covariance(data);
        Self { kappa, mu, nu, psi }
    }
}

#[derive(Clone, Debug)]
pub struct NIW;

impl ConjugatePrior for NIW {
    type HyperParams = NIWParams;
    type SuffStats = NIWStats;

    fn posterior(
        prior: &Self::HyperParams,
        stats: &Self::SuffStats,
    ) -> Self::HyperParams {
        let n_points = stats.n_points as f64;
        let kappa = prior.kappa + n_points;
        let nu = prior.nu + n_points;
        let mu = (&prior.mu * prior.kappa + &stats.mean_sum) / kappa;
        let psi =
            (prior.nu * &prior.psi
                + prior.kappa * &prior.mu * &prior.mu.transpose()
                - kappa * &mu * &mu.transpose()
                + &stats.cov_sum
            ) / nu;
        let psi = (&psi + &psi.transpose()) / 2.0;

        NIWParams { kappa, mu, nu, psi }
    }

    fn marginal_log_likelihood(
        prior: &Self::HyperParams,
        post: &Self::HyperParams,
        stats: &Self::SuffStats,
    ) -> f64 {
        let dim = stats.mean_sum.nrows() as f64;
        -(stats.n_points as f64) * dim * 0.5 * LN_PI
            + mvlgamma(dim as i64, post.nu / 2.0)
            - mvlgamma(dim as i64, prior.nu / 2.0)
            + (prior.nu / 2.0) * (dim * prior.nu.ln() + prior.psi.determinant().ln())
            - (post.nu / 2.0) * (dim * post.nu.ln() + post.psi.determinant().ln())
            + (dim / 2.0) * (prior.kappa / post.kappa).ln()
    }

    fn posterior_predictive<S: Storage<f64, Dynamic, Dynamic>>(
        post: &Self::HyperParams,
        data: &Matrix<f64, Dynamic, Dynamic, S>,
    ) -> f64 {
        todo!()
    }
}

impl GaussianPrior for NIW {
    fn sample<R: Rng + ?Sized>(prior: &Self::HyperParams, rng: &mut R) -> MultivariateNormal {
        prior.sample(rng)
    }
}

impl Distribution<MultivariateNormal> for NIWParams {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> MultivariateNormal {
        let w = InverseWishart::new(self.nu, self.nu * &self.psi).unwrap();
        let sigma = w.sample(rng);
        let mv = MultivariateNormal::new(self.mu.clone(), sigma.clone()).unwrap();
        let mu = mv.sample(rng);
        MultivariateNormal::new(mu, sigma).unwrap()
    }
}


#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};
    use rand::distributions::Distribution;
    use rand::prelude::StdRng;
    use rand::SeedableRng;
    use statrs::assert_almost_eq;
    use crate::priors::niw::{NIWParams, NIWStats};
    use crate::priors::{ConjugatePrior, NIW, SufficientStats};
    use crate::stats::tests::{points1, test_almost_mat};

    fn points0() -> DMatrix<f64> {
        DMatrix::from_row_slice(10, 3, &[
            0.5965, 0.3039, 0.9893,
            0.8451, 0.5469, 0.6594,
            0.1359, 0.4946, 0.5687,
            0.5601, 0.3144, 0.7613,
            0.9956, 0.1559, 0.3861,
            0.7791, 0.2035, 0.5857,
            0.8643, 0.2279, 0.7180,
            0.9132, 0.9476, 0.0753,
            0.1983, 0.2940, 0.6786,
            0.0247, 0.4744, 0.8099,
        ])
    }

    #[test]
    fn test_stats() {
        let points = points1();
        let stats = NIWStats::from_data(&points);

        assert_eq!(stats.n_points, 10);
        test_almost_mat(&stats.mean_sum, &DVector::from_row_slice(&[
            4.05119994468987, 3.5643000551499426, 3.878299990668893,
        ]), 1e-4);
        test_almost_mat(&stats.cov_sum, &DMatrix::from_row_slice(3, 3, &[
            2.1903512039472037, 1.5586958104640931, 1.5794594489964737,
            1.5586958104640931, 2.331203265432532, 1.498778618943676,
            1.5794594489964737, 1.498778618943676, 2.3294769049360573,
        ]), 1e-4);
    }

    #[test]
    fn test_posterior() {
        let prior = NIWParams::from_data(1.0, 4.0, &points0());
        let stats = NIWStats::from_data(&points0());
        let post = NIW::posterior(&prior, &stats);

        assert_almost_eq!(post.kappa, 11.0, 1e-15);
        assert_almost_eq!(post.nu, 14.0, 1e-15);
        test_almost_mat(&post.mu, &DVector::from_row_slice(&[
            0.5912800012744557, 0.39630999875588846, 0.623230000532107
        ]), 1e-5);
        test_almost_mat(&post.psi, &DMatrix::from_row_slice(3, 3, &[
            0.112516, -0.00104276, -0.0349345,
            -0.00104276, 0.0491408, -0.0289923,
            -0.0349345, -0.0289923, 0.0562112,
        ]), 1e-5);
    }

    #[test]
    fn test_sample() {
        let prior = NIWParams::from_data(1.0, 4.0, &points0());

        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            prior.sample(&mut rng);
            // let l_prob = w.ln_pdf(sample);
            // assert!(l_prob.is_finite());
        }
    }

    #[test]
    fn test_aggregate() {
        let stats1 = NIWStats::from_data(&points0());
        let stats2 = NIWStats::from_data(&points1());
        let stats = &stats1 + &stats2;

        assert_eq!(stats.n_points, 20);
        test_almost_mat(&stats.mean_sum, &DVector::from_row_slice(&[
            9.963999958708882, 7.527400041464716, 10.110599996522069
        ]), 1e-4);
        test_almost_mat(&stats.cov_sum, &DMatrix::from_row_slice(3, 3, &[
            6.81164, 3.89157, 4.91515,
            3.89157, 4.39323, 3.67878,
            4.91515, 3.67878, 6.77574,
        ]), 1e-4);
    }

    #[test]
    fn test_log_marginal_likelihood() {
        let prior = NIWParams::from_data(1.0, 4.0, &points0());
        let stats = NIWStats::from_data(&points0());
        let post = NIW::posterior(&prior, &stats);
        let lml = NIW::marginal_log_likelihood(&prior, &post, &stats);

        assert_almost_eq!(lml, -6.829891639640866, 1e-4);
    }
}