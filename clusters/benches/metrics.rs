use criterion::{criterion_group, Criterion};
use rand::distributions::Standard;
use rand::Rng;
use clusters::metrics::normalized_mutual_info_score;

fn bench_normalized_mutual_info_score(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let a: Vec<i32> = (0..10000).map(|_| rng.gen_range(0..10)).collect();
    let b: Vec<i32> = (0..10000).map(|_| rng.gen_range(0..10)).take(10000).collect();

    c.bench_function("nmi", move |bh| bh.iter(|| normalized_mutual_info_score(&a, &b)));
}

criterion_group!(
    metrics,
    bench_normalized_mutual_info_score,
);