use criterion::{criterion_group, Criterion};
use nalgebra::{DMatrix, DVector};
use clusters::stats::Covariance;
use statrs::distribution::{Continuous, MultivariateNormal};
use clusters::stats::ContinuousBatchwise;

fn bench_covariance(c: &mut Criterion) {
    let a = DMatrix::<f64>::new_random(100, 100);
    c.bench_function("row_cov", move |bh| bh.iter(|| a.row_cov()));

    let a = DMatrix::<f64>::new_random(100, 100);
    c.bench_function("col_cov", move |bh| bh.iter(|| a.col_cov()));
}

fn bench_mvn(c: &mut Criterion) {
    let mvn = MultivariateNormal::new(
        DVector::from_vec(vec![0.0, 0.0]),
        DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
    ).unwrap();

    let data = DMatrix::new_random(2, 1000);
    c.bench_function("mvn_pdf", |bh| bh.iter(|| mvn.batchwise_pdf(data.clone())));
    c.bench_function("mvn_ln_pdf", |bh| bh.iter(|| mvn.batchwise_ln_pdf(data.clone())));

    c.bench_function("mvn_pdf_single", |bh| bh.iter(|| {
        let mut res = DVector::zeros(1000);
        for (i, v) in res.iter_mut().enumerate() {
            *v = mvn.pdf(&data.column(i).clone_owned());
        }
        res
    }));
    c.bench_function("mvn_ln_pdf_single", |bh| bh.iter(|| {
        let mut res = DVector::zeros(1000);
        for (i, v) in res.iter_mut().enumerate() {
            *v = mvn.ln_pdf(&data.column(i).clone_owned());
        }
        res
    }));

}

criterion_group!(
    stats,
    bench_covariance,
    bench_mvn,
);