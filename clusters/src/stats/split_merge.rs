use std::marker::PhantomData;
use rand::Rng;
use statrs::function::gamma::ln_gamma;
use crate::clusters::{ClusterParams, SuperClusterParams};
use crate::stats::NormalConjugatePrior;
use crate::utils::each_ref;

pub struct SplitMerge<P: NormalConjugatePrior>(PhantomData<P>);

impl<P: NormalConjugatePrior> SplitMerge<P> {
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
        let prim_stats = prim_l.stats.clone() + &prim_r.stats;
        let prim_post = P::posterior(&prim_l.prior, &prim_stats);
        let prim = ClusterParams::new(prim_l.prior.clone(), prim_post, prim_stats, prim_l.dist.clone());

        let h_merge = Self::compute_log_h_merge(&prim, [prim_l, prim_r], alpha);

        h_merge > rng.gen_range(0.0..1.0_f64).ln()
    }
}


#[cfg(test)]
mod test {
    #[test]
    fn test_compute_log_h_split() {
        todo!()
    }
}