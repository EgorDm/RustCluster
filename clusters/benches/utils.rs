use criterion::{criterion_group, Criterion};
use nalgebra::DMatrix;
use rand::distributions::Standard;
use rand::Rng;
use clusters::utils::{unique_with_indices};

fn bench_unique_with_indices(c: &mut Criterion) {
    let values: Vec<i32> = rand::thread_rng().sample_iter(Standard).take(10000).collect();
    c.bench_function("unique_with_indices", move |bh| bh.iter(|| unique_with_indices(&values, true)));
}

criterion_group!(
    utils,
    bench_unique_with_indices,
);