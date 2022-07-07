use nalgebra::{Cholesky, Dim, DMatrix, DVector, Dynamic, OMatrix, OVector};
use rand::Rng;
use statrs::consts::LN_PI;
use statrs::distribution::{InverseWishart, MultivariateNormal};
use statrs::function::gamma::mvlgamma;
use statrs::statistics::Statistics;
use crate::stats::row_covariance;

#[derive(Debug, Clone, PartialEq)]
pub struct NIWStats {
    N: f64,
    points_sum: DVector<f64>,
    S: DMatrix<f64>,
}

impl NIWStats {
    pub fn new(N: f64, points_sum: DVector<f64>, S: DMatrix<f64>) -> Self {
        Self { N, points_sum, S }
    }

    pub fn from_points(points: &DMatrix<f64>) -> Self <> {
        let N = points.nrows();
        let points_sum = points.row_sum().transpose();
        let S = (points.transpose() * points).symmetric_part();

        NIWStats::new(N as f64, points_sum, S)
    }

    pub fn N(&self) -> f64 {
        self.N
    }

    pub fn points_sum(&self) -> &DVector<f64> {
        &self.points_sum
    }

    pub fn S(&self) -> &DMatrix<f64> {
        &self.S
    }

    pub fn aggregate(&self, other: &Self) -> Self {
        let N = self.N + other.N;
        let points_sum = &self.points_sum + &other.points_sum;
        let S = &self.S + &other.S;

        NIWStats::new(N, points_sum, S)
    }
}


#[derive(Debug, Clone, PartialEq)]
pub struct NIWHyperparams {
    kappa: f64,
    mu: DVector<f64>,
    nu: f64,
    psi: DMatrix<f64>,
}

impl NIWHyperparams {
    pub fn new(kappa: f64, mu: DVector<f64>, nu: f64, psi: DMatrix<f64>) -> Self {
        Self { kappa, mu, nu, psi }
    }

    pub fn from_data(kappa: f64, nu: f64, data: &DMatrix<f64>) -> Self {
        let data_mean = data.row_mean();
        let data_cov = row_covariance(data);
        NIWHyperparams::new(kappa, data_mean.transpose(), nu, data_cov)
    }

    pub fn kappa(&self) -> f64 {
        self.kappa
    }

    pub fn mu(&self) -> &DVector<f64> {
        &self.mu
    }

    pub fn nu(&self) -> f64 {
        self.nu
    }

    pub fn psi(&self) -> &DMatrix<f64> {
        &self.psi
    }

    pub fn posterior(&self, stats: &NIWStats) -> Self {
        let kappa = self.kappa + stats.N;
        let nu = self.nu + stats.N;
        let mu = (&self.mu * self.kappa + &stats.points_sum) / kappa;
        let psi =
            (self.nu * &self.psi
                + self.kappa * &self.mu * &self.mu.transpose()
                - kappa * &mu * &mu.transpose()
                + &stats.S
            ) / nu;
        let psi = (&psi + &psi.transpose()) / 2.0;

        NIWHyperparams { kappa, mu, nu, psi }
    }

    pub fn marginal_log_likelihood(&self, post: &NIWHyperparams, stats: &NIWStats) -> f64 {
        let D = stats.points_sum.nrows() as f64;
        -stats.N * D * 0.5 * LN_PI
            + mvlgamma(D as i64, post.nu / 2.0)
            - mvlgamma(D as i64, self.nu / 2.0)
            + (self.nu / 2.0) * (D * self.nu.ln() + self.psi.determinant().ln())
            - (post.nu / 2.0) * (D * post.nu.ln() + post.psi.determinant().ln())
            + (D / 2.0) * (self.kappa / post.kappa).ln()
    }

    pub fn posterior_predictive(&self, post: &NIWHyperparams, data: &DMatrix<f64>) -> Self {
        unimplemented!()
    }
}

impl ::rand::distributions::Distribution<MultivariateNormal> for NIWHyperparams {
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
    use nalgebra::{DefaultAllocator, Dim, DMatrix, DVector, Matrix, OMatrix};
    use rand::distributions::Distribution;
    use rand::prelude::StdRng;
    use rand::SeedableRng;
    use statrs::assert_almost_eq;
    use crate::priors::niw::{NIWHyperparams, NIWStats};
    use crate::stats::row_covariance;

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

    fn niw_from_data(data: &DMatrix<f64>) -> NIWHyperparams {
        let data_mean = data.row_mean();
        let data_cov = row_covariance(data);
        NIWHyperparams::new(1.0, data_mean.transpose(), 4.0, data_cov)
    }

    #[test]
    fn test_stats() {
        let points = points1();
        let stats = NIWStats::from_points(&points);

        assert_almost_eq!(stats.N, 10.0, 1e-15);
        test_almost_mat(stats.points_sum(), &DVector::from_row_slice(&[
            4.05119994468987, 3.5643000551499426, 3.878299990668893,
        ]), 1e-4);
        test_almost_mat(stats.S(), &DMatrix::from_row_slice(3, 3, &[
            2.1903512039472037, 1.5586958104640931, 1.5794594489964737,
            1.5586958104640931, 2.331203265432532, 1.498778618943676,
            1.5794594489964737, 1.498778618943676, 2.3294769049360573,
        ]), 1e-4);
    }

    #[test]
    fn test_posterior() {
        let niw = niw_from_data(&points0());

        let stats = NIWStats::from_points(&points0());
        let niw_post = niw.posterior(&stats);

        assert_almost_eq!(niw_post.kappa(), 11.0, 1e-15);
        assert_almost_eq!(niw_post.nu(), 14.0, 1e-15);
        test_almost_mat(niw_post.mu(), &DVector::from_row_slice(&[
            0.5912800012744557, 0.39630999875588846, 0.623230000532107
        ]), 1e-5);
        test_almost_mat(niw_post.psi(), &DMatrix::from_row_slice(3, 3, &[
            0.112516, -0.00104276, -0.0349345,
            -0.00104276, 0.0491408, -0.0289923,
            -0.0349345, -0.0289923, 0.0562112,
        ]), 1e-5);
    }

    #[test]
    fn test_sample() {
        let niw = niw_from_data(&points0());

        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let sample = niw.sample(&mut rng);
            // let l_prob = w.ln_pdf(sample);
            // assert!(l_prob.is_finite());
        }
    }

    #[test]
    fn test_aggregate() {
        let stats1 = NIWStats::from_points(&points0());
        let stats2 = NIWStats::from_points(&points1());
        let stats = stats1.aggregate(&stats2);

        assert_almost_eq!(stats.N, 20.0, 1e-15);
        test_almost_mat(stats.points_sum(), &DVector::from_row_slice(&[
            9.963999958708882, 7.527400041464716, 10.110599996522069
        ]), 1e-4);
        test_almost_mat(stats.S(), &DMatrix::from_row_slice(3, 3, &[
            6.81164, 3.89157, 4.91515,
            3.89157, 4.39323, 3.67878,
            4.91515, 3.67878, 6.77574,
        ]), 1e-4);
    }

    #[test]
    fn test_log_marginal_likelihood() {
        let niw = niw_from_data(&points0());
        let stats = NIWStats::from_points(&points0());
        let post = niw.posterior(&stats);
        let lml = niw.marginal_log_likelihood(&post, &stats);

        assert_almost_eq!(lml, -6.829891639640866, 1e-4);
    }
}