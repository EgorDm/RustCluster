use std::marker::PhantomData;
use rand::Rng;
use statrs::function::gamma::ln_gamma;
use crate::clusters::{ClusterParams, SuperClusterParams};
use crate::global::state::{GlobalState};
use crate::options::ModelOptions;
use crate::priors::{GaussianPrior, SufficientStats};
use crate::stats::each_ref;

pub struct GlobalActions<P: GaussianPrior> {
    _phantoms: PhantomData<P>,
}

impl<P: GaussianPrior> GlobalActions<P> {
    pub fn compute_log_h_split(
        prim: &ClusterParams<P>,
        aux: [&ClusterParams<P>; 2],
        alpha: f64,
    ) -> f64 {
        let post = P::posterior(&prim.prior, &prim.stats);
        let post_l = P::posterior(&prim.prior, &aux[0].stats);
        let post_r = P::posterior(&prim.prior, &aux[1].stats);

        let ll = P::marginal_log_likelihood(&prim.prior, &post, &prim.stats);
        let ll_l = P::marginal_log_likelihood(&prim.prior, &post_l, &aux[0].stats);
        let ll_r = P::marginal_log_likelihood(&prim.prior, &post_r, &aux[1].stats);

        alpha.ln()
            + ln_gamma(aux[0].n_points() as f64) + ll_l
            + ln_gamma(aux[1].n_points() as f64) + ll_r
            - ln_gamma(prim.n_points() as f64) - ll
    }

    pub fn should_split<R: Rng>(
        params: &SuperClusterParams<P>,
        alpha: f64,
        rng: &mut R,
    ) -> bool {
        if params.aux.iter().any(|c| c.n_points() == 0) {
            return false;
        }

        let h_split = Self::compute_log_h_split(&params.prim, each_ref(&params.aux), alpha);

        h_split > rng.gen_range(0.0..1.0_f64).ln()
    }


    pub fn compute_log_h_merge(
        prim: &ClusterParams<P>,
        aux: [&ClusterParams<P>; 2],
        alpha: f64,
    ) -> f64 {
        let h_split = Self::compute_log_h_split(prim, aux, alpha);

        -h_split
            + ln_gamma(alpha) - 2.0 * ln_gamma(0.5 * alpha)
            - ln_gamma(prim.n_points() as f64 + alpha)
            + ln_gamma(aux[0].n_points() as f64 + 0.5 * alpha)
            + ln_gamma(aux[1].n_points() as f64 + 0.5 * alpha)
    }

    pub fn should_merge<R: Rng>(
        prim_l: &ClusterParams<P>,
        prim_r: &ClusterParams<P>,
        alpha: f64,
        rng: &mut R,
    ) -> bool {
        let prim_stats = prim_l.stats.add(&prim_r.stats);
        let prim_post = P::posterior(&prim_l.prior, &prim_stats);
        let prim = ClusterParams::new(prim_l.prior.clone(), prim_post, prim_stats, prim_l.dist.clone());

        let h_merge = Self::compute_log_h_merge(&prim, [prim_l, prim_r], alpha);

        h_merge > rng.gen_range(0.0..1.0_f64).ln()
    }

    pub fn check_and_split<R: Rng>(
        global: &mut GlobalState<P>,
        options: &ModelOptions<P>,
        rng: &mut R,
    ) -> Vec<(usize, usize)> {
        let mut decisions = vec![false; global.n_clusters()];
        for (k, cluster) in global.clusters.iter().enumerate() {
            if k == 0 && options.outlier.is_some() {
                continue;
            }

            if cluster.splittable && cluster.n_points() > 1 {
                decisions[k] = Self::should_split(cluster, options.alpha, rng);
            }
        }

        let mut split_idx = Vec::new();
        for (k, split) in decisions.iter().cloned().enumerate() {
            if split {
                let new_idx = global.n_clusters();

                // Split cluster parameters
                global.clusters.push(global.clusters[k].clone());
                let cluster = &global.clusters[k];

                let cluster_l = SuperClusterParams::from_split_params(cluster.aux[0].clone(), options.alpha, options.burnout_period, rng);
                let cluster_r = SuperClusterParams::from_split_params(cluster.aux[1].clone(), options.alpha, options.burnout_period, rng);

                global.clusters[k] = cluster_l;
                global.clusters[new_idx] = cluster_r;

                split_idx.push((k, new_idx));
            }
        }

        split_idx
    }

    pub fn check_and_merge<R: Rng>(
        global: &mut GlobalState<P>,
        options: &ModelOptions<P>,
        rng: &mut R,
    ) -> Vec<(usize, usize)> {
        let mut decisions = Vec::new();
        for ki in 0..global.n_clusters() {
            if ki == 0 && options.outlier.is_some() {
                continue;
            }

            for kj in ki + 1..global.n_clusters() {
                let (cluster_i, cluster_j) = (&global.clusters[ki], &global.clusters[kj]);

                if !cluster_i.splittable || !cluster_j.splittable
                    || cluster_i.n_points() == 0 || cluster_j.n_points() == 0 {
                    continue;
                }

                if !Self::should_merge(&cluster_i.prim, &cluster_j.prim, options.alpha, rng) {
                    continue;
                }

                let cluster = SuperClusterParams::from_merge_params(
                    cluster_i.prim.clone(), cluster_j.prim.clone(),
                    options.alpha, options.burnout_period, rng,
                );
                global.clusters[ki] = cluster;
                global.clusters[kj].prim.stats = P::SuffStats::empty();
                global.clusters[kj].splittable = false;
                decisions.push((ki, kj));
            }
        }

        decisions
    }
}