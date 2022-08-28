use criterion::{criterion_group, Criterion};
use nalgebra::DMatrix;
use clusters::stats::Covariance;

fn covariance(c: &mut Criterion) {
    let a = DMatrix::<f64>::new_random(100, 100);
    c.bench_function("row_cov", move |bh| bh.iter(|| a.row_cov()));

    let a = DMatrix::<f64>::new_random(100, 100);
    c.bench_function("col_cov", move |bh| bh.iter(|| a.col_cov()));
}

criterion_group!(
    stats,
    covariance,
);